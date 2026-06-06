# Directory structure

Keep this document brief and update it whenever folders change.

| Path | Purpose |
| --- | --- |
| `/` | Project root for runtime entrypoints, Compose wiring, helper scripts, and the short landing `README.md`. |
| `docs/` | Project documentation, including structure notes, design history, service/interface docs, and the archived base README. |
| `configs/` | Runtime configuration files, including model path pointers for host and container runs. |
| `configs/models.host.env` | Host-side model path pointers for local Python runs outside Docker. |
| `configs/models.container.env` | Container-side model path pointers used by `docker-compose.yml`. |
| `docs/base_doc.md` | Archived project README content moved out of the root to keep the landing page lightweight. |
| `docs/ocr_service.md` | OCR service architecture, API, and pipeline document. |
| `docs/llm_service.md` | LLM service architecture, API, and prompt contract document. |
| `docs/web_app.md` | Web-app session interface and system design document. |
| `data/` | Runtime uploads, OCR outputs, SQLite state, and other local artifacts generated during development or app usage. |
| `data/web_app/` | Web-app runtime data, including uploads, OCR Markdown artifacts, auth outbox, and the session database. |
| `/home/viettran_orin/models/` | External OCR and LLM model storage kept outside the repository. |
| `/home/viettran_orin/models/PP-OCRv5_mobile_det_infer/` | PP-OCRv5 text detection files for the replacement OCR pipeline. |
| `/home/viettran_orin/models/PP-OCRv5_mobile_rec_infer/` | PP-OCRv5 text recognition files for the replacement OCR pipeline. |
| `/home/viettran_orin/models/PP-LCNet_x0_25_textline_ori_infer/` | PP-LCNet text line orientation files. |
| `/home/viettran_orin/models/PP-LCNet_x1_0_doc_ori_infer/` | PP-LCNet document orientation files. |
| `/home/viettran_orin/models/UVDoc_infer/` | UVDoc document layout analysis files. |
| `/home/viettran_orin/models/PP-DocLayout_plus-L_infer/` | Layout detection files used by the replacement OCR pipeline. |
| `/home/viettran_orin/models/PP-DocBlockLayout_infer/` | Region/block detection files used by the replacement OCR pipeline. |
| `/home/viettran_orin/models/PP-FormulaNet_plus-S_infer/` | Formula recognition files used by the replacement OCR pipeline. |
| `/home/viettran_orin/models/llm/` | Local GGUF model storage for `llm-service`. |
| `wheels/` | Jetson-compatible PaddlePaddle wheel storage and notes. |
| `third_party/` | Packaged third-party runtime dependencies bundled into images. |
| `third_party/llama-bin/bin/` | Validated host `llama.cpp` runtime artifacts copied into `docker/Dockerfile.llm`. |
| `docker/` | Container build recipes and supporting Docker assets. |
| `docker/Dockerfile` | Jetson OCR service image recipe. |
| `docker/Dockerfile.llm` | Jetson LLM service image recipe. |
| `docker/Dockerfile.web` | Web-app image recipe. |
| `docker/paddleocr-l4t-base/` | Minimal OCR image package based on `l4t-base` plus the PaddleOCR runtime. |
| `requirements/` | Python dependency sets grouped by service. |
| `requirements/ocr.txt` | Python dependencies for `ocr-service`. |
| `requirements/llm.txt` | Python dependencies for `llm-service`. |
| `requirements/web.txt` | Python dependencies for `web-app`. |
| `scripts/` | Local helper scripts for OCR diagnostics, benchmarking, and visualization. |
| `scripts/measure_pipeline_baseline.py` | End-to-end baseline timing script for OCR pipeline runs on local fixtures. |
| `src/` | Application source code. |
| `src/ocr_service/` | OCR pipeline and internal OCR API service. |
| `src/ocr_service/pipeline.py` | Document-level OCR wrapper for image/PDF loading, OCR stages, and Markdown assembly. |
| `src/ocr_service/paddle_adapter.py` | Lazy PaddleOCR module loader for the replacement OCR pipeline. |
| `src/ocr_service/image_ops.py` | Shared image and geometry helpers for the OCR pipeline. |
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
| `docker-compose.yml` | Main local service wiring. |
| `start_app.sh` | Start helper for the multi-container runtime with readiness checks. |
| `stop_app.sh` | Stop helper for the running stack. |
| `latest_run_log.txt` | Local captured build or runtime log; disposable and not part of the application runtime contract. |
