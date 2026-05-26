# Design history

Use this file to record design changes, rejected ideas, and why they were replaced. When a new design is accepted, update the main document in `README.md` and the relevant detailed document under `docs/`.

## Current baseline

- Single-node Jetson OCR service
- L4T-based container on Jetson Orin Nano Super with a Jetson-compatible prebuilt PaddlePaddle wheel and CPU fallback
- PP-OCRv5 local model directories with two runtime profiles: `fast` for lower latency and `full` for maximum preprocessing
- Explicit `paddle_static` default engine with a larger recognition batch for text-dense pages
- Public access via tunnel or reverse proxy with auth

## Implementation decisions

- Started with a small `src/ocr_service` package instead of splitting into many services.
- Chose a FastAPI upload endpoint so the first vertical slice can be tested quickly.
- Kept model paths configurable so the same code works on the host and in a container.

## Rejected or replaced designs

- Carrying the full Paddle source tree in-repo for local builds. It consumed too much space and is no longer needed now that a JetPack 6.2-compatible wheel is available.

## Change log

| Date | Change | Why |
| --- | --- | --- |
| 2026-05-26 | Replaced the legacy `ocr-service` internals with the separate Jetson OCR pipeline while preserving the existing upload API and Markdown response contract. | The older OCR service path was no longer reliable enough, but `web-app` and `llm-service` already depended on the current service boundary, so the internal swap kept the integration stable. |
| 2026-05-23 | Switched LLM thinking from globally disabled to opt-in, raised the default context window to 12288 tokens, and made the browser update streamed thinking traces incrementally. | Thinking mode should be clickable and visibly streaming without freezing the chat UI, while Fast mode should keep the lightweight no-thinking path. |
| 2026-05-22 | Collapsed the split Compose setup into a single `docker-compose.yml` and removed the GPU override file. | The helper scripts always consumed the override, so keeping a second Compose file only added indirection without providing a separate supported launch mode. |
| 2026-05-22 | Moved secondary documentation into `docs/` and removed in-repo test/demo fixtures from the main project layout. | The root directory had accumulated documentation drift and disposable assets that were no longer part of the runtime contract. |
| 2026-05-20 | Added cache-busted static asset URLs at `/`, inline thumbnail delivery, frontend compatibility guards for mixed old/new page shells, and regression tests for those deployment behaviors. | The public hostname briefly served an older browser interface after iterative rebuilds; deployment needed stronger guarantees that previews and UI assets would refresh correctly. |
| 2026-05-20 | Added a dedicated web-app session/interface design document and condensed the main README into a short project overview. | The browser UI and session orchestration now need their own source-of-truth document instead of living in the top-level summary. |
| 2026-05-20 | Tuned `llm-service` for OCR QA with explicit Gemma no-thinking flags, container health checks, and unit/smoke tests. The context window later increased to 8096 tokens for chat plus attached OCR context. | The LLM service should answer grounded document requests and normal chat turns without spending tokens on hidden reasoning. |
| 2026-05-20 | Switched `Dockerfile.llm` from source-building `llama.cpp` to packaging validated host `llama-server` artifacts from `third_party/llama-bin/bin`. | Full CUDA source builds inside Docker take too long on Jetson and failed late on CUDA driver-library linking; packaging the known-good host runtime is faster and matches the current pragmatic deployment path. |
| 2026-05-19 | Replaced OCR semantic post-processing with coordinate-arranged Markdown output for the API and local runner. | The web app needs a display-ready OCR result, and guessing questions, options, tables, cells, or page types in the OCR service was adding brittle behavior that is not required. |
| 2026-05-19 | Reworked OCR post-processing into a document-level schema with page blocks, normalized text, question/table extraction, warnings, and PDF-capable API responses. | The OCR container needs a stable contract that downstream web and LLM services can consume without reparsing raw line dumps. |
| 2026-05-19 | Made `fast` + `paddle_static` + recognition batch size 4 the default runtime and kept TensorRT opt-in. | Warmed benchmark probes showed TensorRT only marginally improves this fixture, while batch size 4 was the best steady-state setting for the current local OCR sample. |
| 2026-05-19 | Added a guarded Jetson TensorRT path with module-specific dynamic shapes, lower TRT workspace defaults, legacy Paddle-TRT mode, and local CLI switches for TRT/cache probes. | The upstream HPI path is not supported for Jetson AArch64 GPU, and the first TRT probe hit memory pressure with the upstream 4000-pixel detection shape range. |
| 2026-05-19 | Added local OCR warmup runs for benchmark-style probes and patched PaddleX's already-imported TRT mode flags when enabling TensorRT later in a process. | Cold TensorRT probes were being compared with warmed Paddle runs, and PaddleX latches its PIR/legacy TRT flag at import time. |
| 2026-05-19 | Added a component-level benchmark runner for the five PP-OCRv5 models. | We need direct per-module timings on Jetson to see whether optimization effort should focus on preprocessing, detection, or recognition. |
| 2026-05-19 | Added fast and full runtime profiles, made the reduced-module fast path the default, and switched the default engine to `paddle_static` with a larger recognition batch size. | On this Jetson setup, the effective tuning levers are module selection, batching, and the standard Paddle static runtime; they cut latency more reliably than the full five-stage path. |
| 2026-05-19 | Switched the runtime and container path to the JetPack 6.2-compatible prebuilt PaddlePaddle wheel, enabled the high-level five-stage PP-OCRv5 pipeline, and removed the stale source-build path. | This matches the newly found Jetson wheel, simplifies deployment, and aligns the local runner with the official OCR pipeline docs. |
| 2026-05-18 | Added a local OCR runner for batch images and annotated outputs. | Lets us verify the two-model pipeline on Jetson without standing up the API layer. |
| 2026-05-18 | Reworked the OCR pipeline into an explicit detect -> crop -> recognize flow for local experimentation. | This keeps the first slice focused on the two PP-OCRv5 mobile models and makes the inference path easier to reason about. |
| 2026-05-17 | Chose a pipeline-first OCR design over a heavily split microservice design. | Simpler, lighter on memory, and better suited to Jetson. |
| 2026-05-17 | Started a minimal FastAPI + PaddleOCR implementation. | Fastest path to an end-to-end working OCR slice. |
| 2026-05-17 | Switched the container runtime to a source-built PaddlePaddle wheel and added CPU fallback in the pipeline. | The Jetson runtime needed to work without a public ARM64 GPU wheel, and inference still needs to run when CUDA is unavailable. |
