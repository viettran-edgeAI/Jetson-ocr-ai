# Project progress

## Completed plan — 2026-08-10

Goal: replace the split OCR/chat workspace with one full-screen, dark-first chat application and add multi-image OCR, Brave Search LLM context, and multimodal Gemma request plumbing.

Scope:

- Web UI: left session rail; one central conversation; stacked top-right quick actions; stacked translucent bottom-right Run OCR and Search web tools; responsive/mobile behavior; image attachment previews; search tool indicator inside the composer.
- Web orchestration/storage: multiple ordered attachments per session, cumulative layout-preserving OCR Markdown, artifact cleanup/restore, backend-only Brave context lookup, grounded citations, and image forwarding to the LLM.
- LLM service: accept bounded image data URLs and web context, build llama.cpp multimodal message content, and expose optional multimodal-projector configuration without changing private-service boundaries.
- Operations: reuse only `BRAVE_SEARCH_API_KEY` from `../F1_fact_checker/.env` when the OCR project does not define it, preserve no-cache asset behavior, and validate local plus public deployment.

Acceptance criteria:

1. The default browser view is a visually coherent full-screen dark chat, with sessions at far left and controls in the requested stacked right-side positions.
2. Users can paste/select/drop several PNG/JPEG images in one session, run OCR on the queued batch, restore the session, and see each ordered OCR result as an assistant response with whitespace/layout preserved.
3. Search web inserts a visible search tool indicator in the composer; the next submitted prompt calls Brave `/v1/llm/context`, grounds the model response, and exposes source links without leaking the API key.
4. Attached images are forwarded through the web app to the LLM service as bounded OpenAI-compatible `image_url` content; text-only requests remain compatible. If no compatible projector exists, configuration and error reporting must be explicit.
5. Existing auth, tier limits, session rename/delete/select, chat streaming/thinking, OCR API, and cache-busting behavior continue to work.
6. Automated tests cover schema migration/attachments, OCR aggregation and cleanup, Brave request/normalization/error paths, multimodal payload construction, and key browser contracts; syntax/compile checks and a local UI smoke test pass.
7. The stack is launched and both local and `https://jetsonocrai.cc/` are checked; any external tunnel/model-weight blocker is reported accurately.

Test matrix:

- Unit: Brave query validation, request headers/limits, response normalization, API failure sanitization.
- Unit: attachment table migration/backfill, ordered retrieval, cascade deletion, legacy-session compatibility.
- Unit: LLM text-only vs image-content payloads, web-context truncation, invalid/non-image data URL rejection, optional `--mmproj` command flags.
- API/integration: create chat, upload two mocked OCR images into the same session, cumulative Markdown/order, attachment URLs, rate-limit accounting, ask with and without web search, cleanup.
- Browser/static: required layout landmarks, default dark state, multi-file input, stacked action/tool controls, search chip, and multi-paste queue behavior.
- Operational: Python compile, JavaScript syntax check, pytest, Compose config, local health/static/API smoke, responsive screenshot inspection where browser tooling is available, public endpoint check.

Completion evidence:

- The full-screen dark chat shell, left session rail, stacked quick actions, and stacked translucent OCR/search tools are deployed at `https://jetsonocrai.cc/`; the public HTML exposes the new shell and the origin/tunnel are healthy.
- A production smoke test uploaded two different bundled images to one guest session. Both attachments completed OCR in stable positions `0` and `1`, and cumulative OCR Markdown contained both results. The temporary session was deleted afterward.
- A production Brave LLM-context request confirmed that the backend-only key is configured and returned grounded sources using the application's real 3,000-token limit.
- Focused tests: 17 passed. Full suite: 36 passed. JavaScript syntax, Python compilation, shell syntax, Compose configuration, and whitespace checks all passed.
- Documentation now describes the unified interface, ordered multi-image OCR, Brave grounding, image request path, environment variables, and operational limits.

Known limitation:

- The web and LLM services now support bounded OpenAI-compatible image content and optional llama.cpp `--mmproj` configuration. The current host still has no compatible Gemma multimodal projector file, so `WEB_LLM_VISION_ENABLED` remains safely disabled in production. OCR-grounded image chat works; native Gemma pixel inspection becomes available after compatible projector/model weights are supplied and the gate is enabled.
- No functioning headless browser runtime is installed on the host, so visual verification used the deployed HTML/CSS contracts, responsive static tests, and public-origin smoke checks rather than an automated screenshot. Manual browser review remains the recommended final aesthetic check.

Next action: obtain a compatible Gemma multimodal projector/model pair, set `LLM_MMPROJ_PATH` and `WEB_LLM_VISION_ENABLED=true`, then repeat the multimodal smoke test.
