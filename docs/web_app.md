# Web App

`web-app` is the public browser-facing orchestration layer for the OCR-plus-LLM stack. It serves a full-screen, dark-first unified chat workspace, owns sessions and authentication, queues attachments for OCR, and calls the private `ocr-service` and `llm-service` containers.

## Public boundary

The public application is `https://jetsonocrai.cc`. Cloudflare Tunnel `jetson-ocr-ai` routes it to `web-app` at `http://localhost:8080`. Only `web-app` is public; `ocr-service` and `llm-service` remain private on the Docker network.

The browser entry point is `GET /`. UI HTML and `/static/*` responses disable caching, and the entry point adds a version query to CSS and JavaScript URLs so a deployment does not leave an older shell in the browser.

## Interface

The current workspace fills the viewport and has three coordinated areas:

- A far-left session rail contains the brand, `New chat`, recent sessions, selection/bulk-delete controls, account actions, help, and theme controls.
- The center is one conversation surface. OCR results and later user/assistant messages share the transcript; OCR is rendered as synthetic assistant messages, preserving the source whitespace and line structure.
- A right action rail stacks `Answer` and `Translate` under Quick actions, then `Run OCR` and `Search web` under Tools. `Run OCR` uses the translucent green treatment; `Search web` uses the blue treatment. The composer includes attachment, Fast/Thinking, prompt, send, and web-search chip controls.

The upload drop zone accepts file-picker selection, drag-and-drop, and pasted clipboard images. Supported uploads are PNG, JPG, JPEG, and PDF. The picker allows multiple files.

## Attachment and OCR flow

1. Files are queued in the composer; the queue count and per-file preview are shown before processing.
2. `Run OCR` uploads the queue sequentially to one session through `POST /sessions/upload?session_id=...`.
3. The session stores attachments with an explicit zero-based `position`. Session serialization returns the attachment table in that order, including filename, type, status, OCR Markdown, page count, elapsed time, and image preview URL where applicable.
4. Existing pre-attachment sessions are backfilled as one legacy attachment, so older sessions continue to open through the ordered attachment API.
5. Each completed attachment is displayed as its own OCR assistant message. The cumulative OCR context joins attachment artifacts in order with deterministic separators and does not rewrite OCR whitespace. It is supplied to subsequent LLM requests.

Upload and OCR failures retain the session and attachment state where possible. Hourly OCR limits are tiered (guest 10, free 50, pro 2000; owner is unlimited).

## Chat, translation, and web grounding

`Answer` and `Translate` are quick prompt builders. Translation language is selected in the action rail. Chat can start without an attachment, while an empty prompt is allowed only at the beginning of an OCR session. Fast and Thinking modes are forwarded to the LLM service.

`Search web` is a next-prompt toggle represented in the composer by a `Web search on` chip. The browser sends only `search_web: true` in the ask request. The backend calls Brave's `/v1/llm/context` endpoint with bounded query/context parameters, sends normalized snippets and URLs to `llm-service`, and appends Markdown source links to the assistant response. `BRAVE_SEARCH_API_KEY` is backend-only and is never exposed to the browser. If the process environment does not provide it, the client can read that same variable from the sibling `F1_fact_checker/.env` fallback; it does not read or log other credentials.

Image attachments are forwarded to `llm-service` only when the vision gate is enabled. Forwarding is limited to raster images, at most six images, and bounded per-image/aggregate byte sizes; PDFs are OCR context, not multimodal image inputs.

## Public API

All session and account endpoints are scoped to the signed guest identity or authenticated user cookie.

- `GET /healthz` — web-app health.
- `GET /auth/me`, `POST /auth/signup/start`, `POST /auth/signup/verify-email`, `POST /auth/signup/complete`, `POST /auth/login`, `POST /auth/logout` — identity and account entry points. Authenticated accounts retain email verification, TOTP-capable account controls, and tier usage limits.
- `GET /sessions/recent?include_all=...` — recent or full session summaries.
- `POST /sessions/chat` — create a chat-only session.
- `POST /sessions/upload` — accept one multipart file (`image` is used for the internal OCR call); pass `session_id` to append an attachment.
- `GET /sessions/{id}` — restore a session, messages, ordered attachments, and cumulative OCR Markdown.
- `POST /sessions/{id}/ask` and `POST /sessions/{id}/ask/stream` — submit an `AskRequest` (`prompt`, optional `mode`, `thinking_mode`, and `search_web`). Streaming responses use SSE `token`, `error`, and `done` events and disable intermediary buffering.
- `GET /sessions/{id}/original`, `GET /sessions/{id}/attachments/{attachment_id}/original`, `DELETE /sessions/{id}/attachments/{attachment_id}` — inline previews and attachment deletion.
- `PATCH /sessions/{id}`, `DELETE /sessions/{id}`, `POST /sessions/bulk-delete` — rename and delete sessions.

The web app calls private `POST /v1/ocr` on `ocr-service` and `POST /v1/answer` or `POST /v1/answer/stream` on `llm-service`.

## Environment

Core service and storage settings:

- `WEB_APP_DATA_DIR` (default `data/web_app`), `OCR_SERVICE_URL` (default `http://ocr:8000`), `LLM_SERVICE_URL` (default `http://llm:8081`), `WEB_HOST` (`0.0.0.0`), `WEB_PORT` (`8080`), and `WEB_REQUEST_TIMEOUT_SECONDS` (`360`).
- `WEB_APP_SECRET_KEY` signs guest/auth cookies and must be changed from the development default. `WEB_APP_COOKIE_SECURE` controls the secure cookie flag.
- `WEB_APP_OWNER_EMAIL` identifies the owner tier. SMTP settings are `WEB_APP_SMTP_HOST`, `WEB_APP_SMTP_PORT`, `WEB_APP_SMTP_USERNAME`, `WEB_APP_SMTP_PASSWORD`, `WEB_APP_SMTP_FROM`, and `WEB_APP_SMTP_STARTTLS`.
- `BRAVE_SEARCH_API_KEY`, `BRAVE_LLM_CONTEXT_ENDPOINT`, `BRAVE_CONTEXT_COUNT`, `BRAVE_CONTEXT_MAX_URLS`, `BRAVE_CONTEXT_MAX_SNIPPETS`, `BRAVE_CONTEXT_MAX_TOKENS`, and `BRAVE_SEARCH_TIMEOUT` configure optional backend search. The key may fall back to `../F1_fact_checker/.env` (or the equivalent sibling path) when absent from the process environment.
- `WEB_LLM_VISION_ENABLED` explicitly enables image forwarding. If unset, the gate defaults on only when `LLM_MMPROJ_PATH` is set; `LLM_MMPROJ_PATH` identifies the optional compatible multimodal projector used by `llm-service`.

## Storage and deployment behavior

The app stores uploads and OCR Markdown under `WEB_APP_DATA_DIR` and session/auth/message metadata in SQLite. Session ownership is enforced for guest and registered identities. The public deployment should continue to expose only port 8080 through the tunnel; internal service URLs are not public APIs.

## Limitations

OCR quality and latency vary with scan quality, layout, and PDF size. OCR is performed one attachment at a time, so a large queue increases waiting time. LLM OCR and web context are bounded before prompt assembly. Web grounding depends on a configured and reachable Brave API and is supporting evidence rather than a guarantee of correctness. Image forwarding is deliberately gated and size-limited; on the current host, no compatible mmproj weights are available, so actual visual inference requires supplying compatible weights.
