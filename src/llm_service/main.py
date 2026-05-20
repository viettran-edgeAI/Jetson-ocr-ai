from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.environ.get(
        "LLM_MODEL_PATH",
        APP_ROOT / "models" / "llm" / "gemma-4-E2B-it-Q4_K_M.gguf",
    )
)
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "18080"))
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}")
LLM_HOST = os.environ.get("LLM_HOST", "0.0.0.0")
LLM_PORT = int(os.environ.get("LLM_PORT", "8081"))

DEFAULT_CTX_SIZE = int(os.environ.get("LLM_CTX_SIZE", "4096"))
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "160"))
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
DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "1").lower() not in {
    "0",
    "false",
    "no",
}
EXTERNAL_LLAMA_SERVER = os.environ.get("LLM_EXTERNAL_LLAMA_SERVER", "0").lower() in {
    "1",
    "true",
    "yes",
}


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class AnswerRequest(BaseModel):
    ocr_markdown: str = Field(..., min_length=1)
    user_request: str = Field(..., min_length=1)
    conversation_history: list[ConversationMessage] = Field(default_factory=list, max_length=40)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)


class AnswerResponse(BaseModel):
    answer: str
    model: str
    elapsed_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    ocr_chars: int
    ocr_truncated: bool


class LlamaServer:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None

    async def start(self) -> None:
        if EXTERNAL_LLAMA_SERVER:
            await wait_for_llama_ready()
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


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    if not await is_llama_ready():
        raise HTTPException(status_code=503, detail="llama-server is not ready")
    return {"status": "ok", "model": MODEL_ALIAS}


@app.post("/v1/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest) -> AnswerResponse:
    started = time.perf_counter()
    prepared_ocr = prepare_ocr_markdown(request.ocr_markdown)
    payload = build_chat_payload(request, prepared_ocr=prepared_ocr)
    data = await asyncio.to_thread(post_json, "/v1/chat/completions", payload)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    try:
        message = data["choices"][0]["message"]
        answer_text = str(message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected llama-server response") from exc

    if not answer_text:
        raise HTTPException(status_code=502, detail="llama-server returned an empty answer")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return AnswerResponse(
        answer=answer_text,
        model=str(data.get("model") or MODEL_ALIAS),
        elapsed_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        ocr_chars=prepared_ocr["original_chars"],
        ocr_truncated=prepared_ocr["truncated"],
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
) -> dict[str, Any]:
    if prepared_ocr is None:
        prepared_ocr = prepare_ocr_markdown(request.ocr_markdown)

    system_prompt = (
        "You answer user requests using only the OCR Markdown provided. "
        "If the OCR text does not contain enough evidence, say "
        "\"insufficient evidence in OCR text\". Keep answers concise. "
        "For multiple-choice questions, give the selected option and a brief reason. "
        "Use the conversation history only to resolve follow-up references within "
        "this same OCR session. Do not include hidden reasoning, chain-of-thought, "
        "or <think> text."
    )
    truncation_note = ""
    if prepared_ocr["truncated"]:
        truncation_note = (
            "\n\nNote: OCR Markdown was truncated to fit the configured one-shot context cap."
        )
    ocr_context = (
        "OCR Markdown for this file session:\n"
        "```markdown\n"
        f"{prepared_ocr['text']}\n"
        "```"
        f"{truncation_note}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"{system_prompt}\n\n{ocr_context}"}
    ]
    messages.extend(prepare_conversation_history(request.conversation_history))
    messages.append({"role": "user", "content": request.user_request.strip()})

    payload: dict[str, Any] = {
        "model": MODEL_ALIAS,
        "messages": messages,
        "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "stream": False,
    }
    if DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
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


def main() -> None:
    import uvicorn

    uvicorn.run("llm_service.main:app", host=LLM_HOST, port=LLM_PORT, reload=False)


if __name__ == "__main__":
    main()
