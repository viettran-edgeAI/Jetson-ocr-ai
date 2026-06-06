# LLM Service

This document describes the internal language-model service in `src/llm_service/`: the runtime layout, request and response contracts, prompt assembly, and streaming behavior.

## Purpose

`llm-service` is the private answer-generation backend for the Jetson stack.

It accepts:

- user prompts
- conversation history
- optional OCR Markdown from `ocr-service`

It returns grounded answers through a local `llama-server` process backed by a GGUF model.

The service is intended to be called by `web-app` over the internal Docker network. It is not meant to be exposed publicly.

## Service Components

### `src/llm_service/main.py`

FastAPI entrypoint and the main implementation file.

Responsibilities:

- launches or connects to `llama-server`
- waits for the model backend to become ready
- optionally performs a startup warmup call
- exposes `GET /healthz`
- exposes `POST /v1/answer`
- exposes `POST /v1/answer/stream`
- builds the final chat payload sent to `llama-server`
- parses normal and streaming responses
- applies OCR truncation, thinking-mode rules, and conversation-history limits

### `src/llm_service/__init__.py`

Package marker only.

## Runtime Model and Backend

The default model path is:

- `/home/viettran_orin/models/llm/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`

This can be overridden with:

- `LLM_MODEL_PATH`

Local host runs read `configs/models.host.env` by default. Docker Compose loads `configs/models.container.env` and mounts the same external model root at `/models`.

The service starts `llama-server` unless an external server is configured.

Key runtime settings:

- `LLAMA_SERVER_BIN` defaults to `llama-server`
- `LLAMA_HOST` defaults to `127.0.0.1`
- `LLAMA_PORT` defaults to `18080`
- `LLAMA_SERVER_URL` defaults to `http://127.0.0.1:18080`
- `LLM_HOST` defaults to `0.0.0.0`
- `LLM_PORT` defaults to `8081`
- `LLM_MODEL_ALIAS` defaults to `gemma-4-E2B-it-Q4_K_M`

External-server mode:

- `LLM_EXTERNAL_LLAMA_SERVER=1` makes the service connect to an already-running `llama-server` instead of spawning one

## API Surface

### `GET /healthz`

Returns a small health response once `llama-server` is ready:

```json
{
  "status": "ok",
  "model": "gemma-4-E2B-it-Q4_K_M"
}
```

If the backend is not ready, the service returns `503`.

### `POST /v1/answer`

Synchronous answer endpoint.

Request body uses the `AnswerRequest` model:

- `ocr_markdown`: optional OCR Markdown string
- `user_request`: required user prompt
- `conversation_history`: list of prior `user` / `assistant` messages
- `max_tokens`: optional override
- `thinking_mode`: `fast` or `thinking`

Response body uses the `AnswerResponse` model:

- `answer`
- `reasoning_text`
- `model`
- `elapsed_ms`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `ocr_chars`
- `ocr_truncated`

### `POST /v1/answer/stream`

Streaming answer endpoint.

It returns server-sent events with:

- `token` events for answer or reasoning deltas
- `error` events on failure
- a final `done` event with the assembled answer and usage metadata

The response includes:

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `X-Accel-Buffering: no`

## Request Models

### `ConversationMessage`

Each history item contains:

- `role`: `user` or `assistant`
- `content`: non-empty string

### `AnswerRequest`

Fields:

- `ocr_markdown`
- `user_request`
- `conversation_history`
- `max_tokens`
- `thinking_mode`

Important limits:

- message content is capped at 8000 characters per history item
- conversation history is capped at 40 items
- `max_tokens` is limited to 2048 when provided

## Prompt Assembly

The service builds the final chat payload in `build_chat_payload()`.

### OCR handling

`prepare_ocr_markdown()` trims the OCR Markdown and caps it at `LLM_MAX_OCR_CHARS`:

- default cap: `12000` characters
- if truncated, the prompt includes a note explaining that the OCR text was cut down

If OCR text is present, the service inserts it into the system prompt as fenced Markdown and instructs the model to use it as the primary document context.

### No OCR context

If no OCR Markdown is attached, the system prompt tells the model that no document context is available yet and that the user should attach a document if they are asking about one.

### Conversation history

`prepare_conversation_history()` walks history from newest to oldest, keeps the most recent usable messages, and truncates at `LLM_MAX_HISTORY_CHARS`:

- default cap: `4000` characters

This keeps follow-up context while protecting the model context window.

### Thinking mode

The service supports two modes:

- `fast`
- `thinking`

`thinking` is only active when `LLM_DISABLE_THINKING` is not set.

When thinking mode is enabled:

- the prompt asks for concise, user-facing reasoning
- visible reasoning updates may stream through the `/v1/answer/stream` endpoint
- hidden chain-of-thought is still suppressed from the final prompt contract

When fast mode is active:

- thinking is explicitly disabled in the chat template kwargs

## Model Execution

The service builds a `llama-server` command with:

- model path
- alias
- context size
- parallelism
- GPU layer count
- temperature
- top-p
- top-k
- offline mode
- no UI mode

Relevant knobs:

- `LLM_CTX_SIZE` defaults to `12288`
- `LLM_MAX_TOKENS` defaults to `160`
- `LLM_THINKING_MAX_TOKENS` defaults to `768`
- `LLM_TEMPERATURE` defaults to `0.2`
- `LLM_TOP_P` defaults to `0.95`
- `LLM_TOP_K` defaults to `40`
- `LLM_PARALLEL` defaults to `1`
- `LLM_GPU_LAYERS` defaults to `auto`

Additional optional flags:

- `LLM_DEVICE`
- `LLM_FLASH_ATTN`
- `LLM_FIT`
- `LLM_KV_OFFLOAD`
- `LLM_OP_OFFLOAD`

## Startup Behavior

The lifespan handler does the following:

1. Start the `llama-server` process unless external mode is enabled.
2. Wait for `GET /health` to report ready.
3. Optionally send a warmup request.

Startup controls:

- `LLM_WARMUP_ON_STARTUP` defaults to on
- `LLM_STARTUP_TIMEOUT_SECONDS` defaults to `240`
- `LLM_REQUEST_TIMEOUT_SECONDS` defaults to `300`

If the backend process exits early or never becomes ready, startup fails.

## Response Parsing

The service handles both JSON and streaming SSE responses from `llama-server`.

### Normal response parsing

`extract_message_parts()` returns:

- answer text from `content`
- reasoning text from `reasoning` or `reasoning_content`

If only reasoning text is present, the service tries to split structured output into answer and reasoning using simple heuristics.

### Streaming parsing

`extract_delta_parts()` handles streaming chunks by reading:

- `delta.content`
- `delta.reasoning`
- `delta.reasoning_content`

The streaming handler assembles:

- answer text
- reasoning text
- usage counts when present
- timing metadata from the backend when available

If no answer text is produced, the service falls back to a non-streaming request before failing.

## Error Handling

The service converts backend failures into `502` responses when:

- `llama-server` returns an HTTP error
- the backend is unavailable
- the response shape is unexpected
- the answer is empty

Health checks return `503` when the backend is not ready.

## Downstream Contract

The web app sends the current user request plus any OCR Markdown to `POST /v1/answer` or `POST /v1/answer/stream`.

The LLM service returns:

- the final answer text
- optional reasoning text for UI display
- token usage metadata when available
- OCR context size and truncation flags

That allows the web app to render the answer, keep the OCR context visible, and optionally show a Thinking panel without losing the final response.
