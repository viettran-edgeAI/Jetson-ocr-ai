# Directory structure

Keep this document brief and update it whenever folders change.

| Path | Purpose |
| --- | --- |
| `/` | Project root for runtime entrypoints, Docker files, and the top-level README. |
| `docs/` | Project documentation, including structure notes, design history, and the web-app design spec. |
| `data/` | Runtime uploads, OCR outputs, SQLite state, and other local artifacts generated during development or app usage. |
| `data/web_app/` | Web-app runtime data, including uploads, OCR Markdown artifacts, auth outbox, and the session database. |
| `models/` | Local OCR and LLM model snapshots and metadata. |
| `models/PP-OCRv5_mobile_det/` | PP-OCRv5 text detection files. |
| `models/PP-OCRv5_mobile_rec/` | PP-OCRv5 text recognition files. |
| `models/PP-LCNet_x0_25_textline_ori_infer/` | PP-LCNet text line orientation files. |
| `models/PP-LCNet_x1_0_doc_ori_infer/` | PP-LCNet document orientation files. |
| `models/UVDoc_infer/` | UVDoc document layout analysis files. |
| `models/llm/` | Local GGUF model storage for `llm-service`. |
| `wheels/` | Jetson-compatible PaddlePaddle wheel storage and notes. |
| `third_party/` | Packaged third-party runtime dependencies bundled into images. |
| `third_party/llama-bin/bin/` | Validated host `llama.cpp` runtime artifacts copied into `Dockerfile.llm`. |
| `src/` | Application source code. |
| `src/ocr_service/` | OCR pipeline and internal OCR API service. |
| `src/ocr_service/pipeline.py` | High-level PP-OCRv5 pipeline wrapper that uses the local OCR models. |
| `src/ocr_service/local_infer.py` | Local OCR runner for explicit image or PDF inputs. |
| `src/ocr_service/module_benchmark.py` | Local benchmark runner for the OCR component models. |
| `src/ocr_service/main.py` | FastAPI app for upload-and-read OCR requests. |
| `src/llm_service/` | LLM assistant API service grounded in OCR Markdown. |
| `src/llm_service/main.py` | FastAPI wrapper that starts `llama-server` and exposes `/v1/answer`. |
| `src/web_app/` | Public browser UI and session orchestration layer. |
| `src/web_app/auth.py` | Local account, password hashing, signed cookie, and identity helpers. |
| `src/web_app/main.py` | FastAPI app for uploads, session restore, OCR calls, and LLM calls. |
| `src/web_app/store.py` | SQLite session and message persistence helper. |
| `src/web_app/static/` | Static HTML, CSS, and JavaScript for the OCR AI Assistant interface. |
| `Dockerfile` | Jetson OCR service image recipe. |
| `Dockerfile.llm` | Jetson LLM service image recipe. |
| `Dockerfile.web` | Web-app image recipe. |
| `docker-compose.yml` | Main local service wiring. |
| `docker-compose.gpu-test.yml` | GPU-specific Compose overrides used by helper scripts. |
| `requirements.txt` | Python dependencies for `ocr-service`. |
| `requirements-llm.txt` | Python dependencies for `llm-service`. |
| `requirements-web.txt` | Python dependencies for `web-app`. |
| `start_app.sh` | Start helper for the multi-container runtime with readiness checks. |
| `stop_app.sh` | Stop helper for the running stack. |
| `latest_run_log.txt` | Local captured build or runtime log; disposable and not part of the application runtime contract. |
