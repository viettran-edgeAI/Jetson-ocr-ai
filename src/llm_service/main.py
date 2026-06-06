from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from runtime_env import load_default_model_env

load_default_model_env()

MODEL_PATH = Path(
    os.environ.get(
        "LLM_MODEL_PATH",
        Path.home() / "models" / "llm" / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
    )
)
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "18080"))
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}")
LLM_HOST = os.environ.get("LLM_HOST", "0.0.0.0")
LLM_PORT = int(os.environ.get("LLM_PORT", "8081"))

DEFAULT_CTX_SIZE = int(os.environ.get("LLM_CTX_SIZE", "12288"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "160"))
DEFAULT_THINKING_MAX_TOKENS = int(
    os.environ.get("LLM_THINKING_MAX_TOKENS", os.environ.get("LLM_MAX_TOKENS_THINKING", "768"))
)
DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("LLM_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.environ.get("LLM_TOP_K", "40"))
DEFAULT_PARALLEL = int(os.environ.get("LLM_PARALLEL", "1"))
DEFAULT_GPU_LAYERS = os.environ.get("LLM_GPU_LAYERS", "auto")
DEFAULT_MAX_OCR_CHARS = int(os.environ.get("LLM_MAX_OCR_CHARS", "12000"))
DEFAULT_MAX_HISTORY_CHARS = int(os.environ.get("LLM_MAX_HISTORY_CHARS", "4000"))
LLM_DEVICE = os.environ.get("LLM_DEVICE", "").strip()
LLM_FLASH_ATTN = os.environ.get("LLM_FLASH_ATTN", "").strip()
LLM_FIT = os.environ.get("LLM_FIT", "").strip()
LLM_KV_OFFLOAD = os.environ.get("LLM_KV_OFFLOAD", "1").lower() not in {
    "0",
    "false",
    "no",
}
LLM_OP_OFFLOAD = os.environ.get("LLM_OP_OFFLOAD", "1").lower() not in {
    "0",
    "false",
    "no",
}
STARTUP_TIMEOUT_SECONDS = float(os.environ.get("LLM_STARTUP_TIMEOUT_SECONDS", "240"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "300"))

MODEL_ALIAS = os.environ.get("LLM_MODEL_ALIAS", "gemma-4-E2B-it-Q4_K_M")
DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "0").lower() not in {
    "0",
    "false",
    "no",
}
EXTERNAL_LLAMA_SERVER = os.environ.get("LLM_EXTERNAL_LLAMA_SERVER", "0").lower() in {
    "1",
    "true",
    "yes",
}
WARMUP_ON_STARTUP = os.environ.get("LLM_WARMUP_ON_STARTUP", "1").lower() not in {
    "0",
    "false",
    "no",
}

logger = logging.getLogger(__name__)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class AnswerRequest(BaseModel):
    ocr_markdown: str = Field(default="")
    user_request: str = Field(..., min_length=1)
    conversation_history: list[ConversationMessage] = Field(default_factory=list, max_length=40)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    thinking_mode: Literal["fast", "thinking"] = "fast"


class AnswerResponse(BaseModel):
    answer: str
    reasoning_text: str | None = None
    model: str
    elapsed_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    ocr_chars: int
    ocr_truncated: bool
    stopped_due_to_max_tokens: bool = False
    max_tokens_limit: int | None = None


class LlamaServer:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    async def start(self) -> None:
        if EXTERNAL_LLAMA_SERVER:
            await wait_for_llama_ready()
            await maybe_warmup_llama()
            return

        if not MODEL_PATH.exists():
            raise RuntimeError(f"LLM model file does not exist: {MODEL_PATH}")

        self.process = subprocess.Popen(
            build_llama_command(),
            stdout=None,
            stderr=None,
            text=True,
        )
        await wait_for_llama_ready(process=self.process)
        await maybe_warmup_llama()

    async def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            await asyncio.to_thread(self.process.wait, 20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            await asyncio.to_thread(self.process.wait, 10)


server = LlamaServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await server.start()
    try:
        yield
    finally:
        await server.stop()


app = FastAPI(title="Jetson OCR LLM Service", version="0.1.0", lifespan=lifespan)
SYSTEM_PROMPT = (
    "Answer directly and concisely. Do not repeat the question in your answer. "
    "Use OCR Markdown context when relevant. If OCR lacks details, you may use "
    "general knowledge and briefly state that OCR lacked details."
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    if not await is_llama_ready():
        raise HTTPException(status_code=503, detail="llama-server is not ready")
    return {"status": "ok", "model": MODEL_ALIAS}


@app.post("/v1/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse:
    started = time.perf_counter()
    prepared_ocr = prepare_ocr_markdown(request.ocr_markdown)
    payload = build_chat_payload(request, prepared_ocr=prepared_ocr, stream=False)
    max_tokens_limit = resolve_max_tokens(request)
    data = await asyncio.to_thread(post_json, "/v1/chat/completions", payload)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    try:
        first_choice = data["choices"][0]
        message = first_choice["message"]
        answer_text, reasoning_text = extract_message_parts(message)
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected llama-server response") from exc

    if not answer_text:
        raise HTTPException(status_code=502, detail="llama-server returned an empty answer")

    finish_reason = str(first_choice.get("finish_reason") or "").strip().lower()
    stopped_due_to_max_tokens = finish_reason == "length"
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return AnswerResponse(
        answer=answer_text,
        reasoning_text=reasoning_text or None,
        model=str(data.get("model") or MODEL_ALIAS),
        elapsed_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        ocr_chars=prepared_ocr["original_chars"],
        ocr_truncated=prepared_ocr["truncated"],
        stopped_due_to_max_tokens=stopped_due_to_max_tokens,
        max_tokens_limit=max_tokens_limit,
    )


@app.post("/v1/answer/stream")
async def answer_question_stream(request: AnswerRequest) -> StreamingResponse:
    started = time.perf_counter()
    prepared_ocr = prepare_ocr_markdown(request.ocr_markdown)
    payload = build_chat_payload(request, prepared_ocr=prepared_ocr, stream=True)
    max_tokens_limit = resolve_max_tokens(request)

    def stream_events() -> Generator[str, None, None]:
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        model_name = MODEL_ALIAS
        completion_chunks = 0
        finish_reason = ""
        try:
            for chunk in post_json_stream("/v1/chat/completions", payload):
                model_name = str(chunk.get("model") or model_name)
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                chunk_timings = chunk.get("timings")
                if isinstance(chunk_timings, dict):
                    timings = chunk_timings
                chunk_finish_reason = extract_finish_reason(chunk)
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                delta_parts = extract_delta_parts(chunk)
                reasoning_delta = delta_parts["reasoning"]
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    yield sse_event("token", {"delta": reasoning_delta, "kind": "reasoning"})
                answer_delta = delta_parts["answer"]
                if answer_delta:
                    answer_parts.append(answer_delta)
                    completion_chunks += 1
                    yield sse_event("token", {"delta": answer_delta, "kind": "answer"})
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "llama-server stream failed"
            yield sse_event("error", {"detail": detail})
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            yield sse_event("error", {"detail": f"llama-server stream failed: {exc}"})
            return

        answer_text = "".join(answer_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        if not answer_text:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            fallback_payload.pop("stream_options", None)
            try:
                fallback_data = post_json("/v1/chat/completions", fallback_payload)
                model_name = str(fallback_data.get("model") or model_name)
                fallback_usage = fallback_data.get("usage")
                if isinstance(fallback_usage, dict):
                    usage = fallback_usage
                choices = fallback_data.get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        fallback_finish_reason = str(first_choice.get("finish_reason") or "").strip().lower()
                        if fallback_finish_reason:
                            finish_reason = fallback_finish_reason
                        message = first_choice.get("message")
                        if isinstance(message, dict):
                            answer_text, reasoning_text = extract_message_parts(message)
            except HTTPException:
                answer_text = ""
        if not answer_text:
            yield sse_event("error", {"detail": "llama-server returned an empty answer"})
            return

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        prompt_tokens = as_int_or_none(usage.get("prompt_tokens"))
        completion_tokens = as_int_or_none(usage.get("completion_tokens"))
        total_tokens = as_int_or_none(usage.get("total_tokens"))
        if completion_tokens is None:
            completion_tokens = as_int_or_none(timings.get("predicted_n"))
        if prompt_tokens is None:
            prompt_tokens = as_int_or_none(timings.get("prompt_n"))
        if completion_tokens is None and completion_chunks > 0:
            completion_tokens = completion_chunks
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        stopped_due_to_max_tokens = finish_reason == "length"
        yield sse_event(
            "done",
            {
                "answer": answer_text,
                "reasoning_text": reasoning_text or None,
                "model": model_name,
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "ocr_chars": prepared_ocr["original_chars"],
                "ocr_truncated": prepared_ocr["truncated"],
                "stopped_due_to_max_tokens": stopped_due_to_max_tokens,
                "max_tokens_limit": max_tokens_limit,
            },
        )

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def build_llama_command() -> list[str]:
    command = [
        LLAMA_SERVER_BIN,
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "--model",
        str(MODEL_PATH),
        "--alias",
        MODEL_ALIAS,
        "--ctx-size",
        str(DEFAULT_CTX_SIZE),
        "--parallel",
        str(DEFAULT_PARALLEL),
        "--gpu-layers",
        DEFAULT_GPU_LAYERS,
        "--temp",
        str(DEFAULT_TEMPERATURE),
        "--top-p",
        str(DEFAULT_TOP_P),
        "--top-k",
        str(DEFAULT_TOP_K),
        "--no-ui",
        "--offline",
    ]
    if DISABLE_THINKING:
        command.extend(
            [
                "--chat-template-kwargs",
                json.dumps({"enable_thinking": False}, separators=(",", ":")),
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
            ]
        )
    if LLM_DEVICE:
        command.extend(["--device", LLM_DEVICE])
    if LLM_FLASH_ATTN:
        command.extend(["--flash-attn", LLM_FLASH_ATTN])
    if LLM_FIT:
        command.extend(["--fit", LLM_FIT])
    if not LLM_KV_OFFLOAD:
        command.append("--no-kv-offload")
    if not LLM_OP_OFFLOAD:
        command.append("--no-op-offload")
    return command


def build_chat_payload(
    request: AnswerRequest,
    prepared_ocr: dict[str, Any] | None = None,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    if prepared_ocr is None:
        prepared_ocr = prepare_ocr_markdown(request.ocr_markdown)

    has_ocr_context = bool(prepared_ocr["text"])
    truncation_note = ""
    if prepared_ocr["truncated"]:
        truncation_note = (
            "\n\nNote: OCR Markdown was truncated to fit the configured context cap."
        )
    system_content = SYSTEM_PROMPT
    if has_ocr_context:
        ocr_context = (
            "OCR Markdown appended to this session:\n"
            "```markdown\n"
            f"{prepared_ocr['text']}\n"
            "```"
            f"{truncation_note}"
        )
        system_content = f"{SYSTEM_PROMPT}\n\n{ocr_context}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    messages.extend(prepare_conversation_history(request.conversation_history))
    messages.append({"role": "user", "content": request.user_request.strip()})

    payload: dict[str, Any] = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "max_tokens": resolve_max_tokens(request),
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    apply_thinking_mode(payload, request.thinking_mode)
    return payload


def build_warmup_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "Warm the model and reply with one short token."}],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": False,
    }
    apply_thinking_mode(payload, "fast")
    return payload


def prepare_conversation_history(
    conversation_history: list[ConversationMessage],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_chars = 0

    for message in reversed(conversation_history):
        content = message.content.strip()
        if not content:
            continue

        remaining_chars = DEFAULT_MAX_HISTORY_CHARS - used_chars
        if remaining_chars <= 0:
            break

        if len(content) > remaining_chars:
            suffix = "\n[Conversation history truncated]"
            if remaining_chars > len(suffix):
                content_budget = remaining_chars - len(suffix)
                content = f"{content[:content_budget].rstrip()}{suffix}"
            else:
                content = content[:remaining_chars].rstrip()

        selected.append({"role": message.role, "content": content})
        used_chars += len(content)

    selected.reverse()
    return selected


def prepare_ocr_markdown(ocr_markdown: str) -> dict[str, Any]:
    text = ocr_markdown.strip()
    original_chars = len(text)
    if original_chars <= DEFAULT_MAX_OCR_CHARS:
        return {"text": text, "original_chars": original_chars, "truncated": False}

    head_budget = max(DEFAULT_MAX_OCR_CHARS - 160, 1)
    trimmed = text[:head_budget].rstrip()
    trimmed = (
        f"{trimmed}\n\n[OCR Markdown truncated after {head_budget} "
        f"of {original_chars} characters]"
    )
    return {"text": trimmed, "original_chars": original_chars, "truncated": True}


async def wait_for_llama_ready(process: subprocess.Popen[str] | None = None) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        if await is_llama_ready():
            return
        await asyncio.sleep(1)
    raise RuntimeError(f"llama-server did not become ready: {last_error}")


async def maybe_warmup_llama() -> None:
    if not WARMUP_ON_STARTUP:
        return
    try:
        await asyncio.to_thread(post_json, "/v1/chat/completions", build_warmup_payload())
    except Exception as exc:
        logger.warning("llama warmup failed: %s", exc)


async def is_llama_ready() -> bool:
    try:
        await asyncio.to_thread(get_json, "/health")
        return True
    except Exception:
        return False


def get_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{LLAMA_SERVER_URL}{path}", method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{LLAMA_SERVER_URL}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"llama-server error: {detail}") from exc
    return json.loads(raw)


def post_json_stream(path: str, payload: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{LLAMA_SERVER_URL}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            for data in iter_sse_data_frames(response):
                if not data or data == "[DONE]":
                    continue
                try:
                    decoded = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    yield decoded
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"llama-server error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"llama-server is unavailable: {exc.reason}") from exc


def iter_sse_data_frames(response: Any) -> Generator[str, None, None]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def extract_delta_parts(chunk: dict[str, Any]) -> dict[str, str]:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"answer": "", "reasoning": ""}
    first = choices[0]
    if not isinstance(first, dict):
        return {"answer": "", "reasoning": ""}
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return {"answer": "", "reasoning": ""}
    return {
        "answer": normalize_text_content(delta.get("content")),
        "reasoning": extract_text_from_fields(delta, ("reasoning", "reasoning_content")),
    }


def extract_finish_reason(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    return str(first.get("finish_reason") or "").strip().lower()


def extract_delta_content(chunk: dict[str, Any]) -> str:
    parts = extract_delta_parts(chunk)
    return parts["answer"] or parts["reasoning"]


def extract_message_text(message: dict[str, Any]) -> str:
    return extract_text_from_fields(message, ("content", "reasoning", "reasoning_content"))


def extract_message_parts(message: dict[str, Any]) -> tuple[str, str]:
    answer_text = normalize_text_content(message.get("content")).strip()
    reasoning_text = extract_text_from_fields(message, ("reasoning", "reasoning_content")).strip()
    if answer_text:
        return answer_text, reasoning_text
    if reasoning_text:
        split_answer, split_reasoning = split_reasoning_output(reasoning_text)
        return split_answer, split_reasoning
    return "", ""


def extract_text_from_fields(payload: dict[str, Any], field_order: tuple[str, ...]) -> str:
    for field in field_order:
        value = payload.get(field)
        text = normalize_text_content(value)
        if text:
            return text
    return ""


def normalize_text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item:
                parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"text", "output_text", "reasoning", "reasoning_text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def split_reasoning_output(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return "", ""

    think_match = re.search(r"<think>(.*?)</think>(.*)", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning_text = think_match.group(1).strip()
        answer_text = think_match.group(2).strip()
        if answer_text:
            return answer_text, reasoning_text

    labeled_match = re.search(
        r"(.*?)(?:^|\n)(?:final answer|answer|response|conclusion)\s*:\s*(.+)$",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    if labeled_match:
        reasoning_text = labeled_match.group(1).strip()
        answer_text = labeled_match.group(2).strip()
        if answer_text:
            return answer_text, reasoning_text

    numbered_match = re.search(
        r"(.*(?:^|\n)\d+\.\s+[^\n]+:\s.*?)(\n+[A-Z][\s\S]+)$",
        cleaned,
        flags=re.DOTALL | re.MULTILINE,
    )
    if numbered_match:
        reasoning_text = numbered_match.group(1).strip()
        answer_text = numbered_match.group(2).strip()
        if answer_text:
            return answer_text, reasoning_text

    return cleaned, ""


def resolve_max_tokens(request: AnswerRequest) -> int:
    if request.max_tokens is not None:
        return request.max_tokens
    if request.thinking_mode == "thinking" and not DISABLE_THINKING:
        return DEFAULT_THINKING_MAX_TOKENS
    return DEFAULT_MAX_TOKENS


def apply_thinking_mode(payload: dict[str, Any], thinking_mode: str) -> None:
    if DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return
    if thinking_mode == "thinking":
        payload.pop("chat_template_kwargs", None)
        return
    payload["chat_template_kwargs"] = {"enable_thinking": False}


def as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sse_event(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def main() -> None:
    import uvicorn

    uvicorn.run("llm_service.main:app", host=LLM_HOST, port=LLM_PORT, reload=False)


if __name__ == "__main__":
    main()
