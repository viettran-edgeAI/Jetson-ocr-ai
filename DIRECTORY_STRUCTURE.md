# Directory structure

Keep this document brief and update it whenever folders change.

| Folder | Purpose |
| --- | --- |
| `/` | Project root; central docs and future app files. |
| `.dockerignore` | Keeps local models, data, virtualenvs, and test assets out of Docker build contexts. |
| `models/` | Local OCR model snapshots and metadata. |
| `models/PP-OCRv5_mobile_det/` | PP-OCRv5 text detection files. |
| `models/PP-OCRv5_mobile_rec/` | PP-OCRv5 text recognition files. |
| `models/PP-LCNet_x0_25_textline_ori_infer/` | PP-LCNet text line orientation files. |
| `models/PP-LCNet_x1_0_doc_ori_infer/` | PP-LCNet document orientation files. |
| `models/UVDoc_infer/` | UVDoc document layout analysis files. |
| `data/` | Runtime uploads and OCR results. |
| `data/ocr_local/` | Local PP-OCRv5 demo outputs: Markdown plus annotated images. |
| `ocr_img_test/` | Local image fixtures for OCR smoke tests and demos. |
| `wheels/` | Jetson-compatible PaddlePaddle wheel and related notes. |
| `Dockerfile` | Jetson container image recipe that installs the local Jetson-compatible PaddlePaddle wheel. |
| `Dockerfile.llm` | Jetson LLM service image recipe for the FastAPI wrapper around `llama.cpp`. |
| `Dockerfile.web` | Lightweight web-app image recipe for the public browser/session layer. |
| `docker-compose.yml` | Local service and GPU/CPU runtime wiring. |
| `WEB_APP_SESSION_INTERFACE_AND_SYSTEM_DESIGN.md` | Dedicated design spec for the web-app session interface and orchestration layer. |
| `requirements.txt` | Python dependencies for the service. |
| `requirements-llm.txt` | Minimal Python dependencies for the LLM service. |
| `requirements-web.txt` | Minimal Python dependencies for the web-app service. |
| `src/` | Application source code. |
| `src/ocr_service/` | OCR pipeline and API service. |
| `src/ocr_service/pipeline.py` | High-level PP-OCRv5 pipeline wrapper that uses all five local OCR models. |
| `src/ocr_service/local_infer.py` | Local runner for batch OCR + annotated image output. |
| `src/ocr_service/module_benchmark.py` | Local benchmark runner that times the five PP-OCRv5 component models separately. |
| `src/ocr_service/main.py` | FastAPI app for upload-and-read. |
| `src/llm_service/` | LLM assistant API service that answers user requests grounded in OCR Markdown. |
| `src/llm_service/main.py` | FastAPI wrapper that starts `llama-server` and exposes `/v1/answer`. |
| `src/llm_service/smoke_test.py` | Local smoke test client for `question_0.md` and a simulated user request. |
| `src/web_app/` | Public browser UI and session orchestration layer. |
| `src/web_app/main.py` | FastAPI app for uploads, session restore, OCR calls, and LLM calls. |
| `src/web_app/store.py` | SQLite session and message persistence helper. |
| `src/web_app/static/` | Static HTML, CSS, and JavaScript for the OCR AI Assistant interface. |
| `tests/` | Lightweight unit tests for service configuration and request payload behavior. |
| `third_party/llama-bin/bin/` | Ignored local staging area for validated host `llama.cpp` runtime artifacts copied into `Dockerfile.llm`. |
