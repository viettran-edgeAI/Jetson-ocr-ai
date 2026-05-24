# OCR Service

This document describes the internal OCR service in `src/ocr_service/`: what it contains, how the pipeline is assembled, and what the API returns.

## Purpose

`ocr-service` is the private OCR backend for the Jetson stack. It accepts a single uploaded document, runs PaddleOCR-based inference locally, converts the result into Markdown, and returns that Markdown to the caller.

It is designed to be used by `web-app` over the internal Docker network, not directly from the public internet.

## Service Components

### `src/ocr_service/main.py`

FastAPI entrypoint for the OCR service.

Responsibilities:

- creates the runtime directories under `data/`
- lazily initializes the OCR pipeline
- optionally preloads and warms up the pipeline during startup
- exposes `GET /healthz`
- exposes `POST /v1/ocr`

Runtime directories created on startup:

- `data/uploads/` for copied input files
- `data/results/` for generated Markdown
- `data/.ocr_warmup.png` as a temporary warmup artifact

### `src/ocr_service/pipeline.py`

Core OCR implementation.

This file defines:

- the OCR result data model
- PaddleOCR pipeline configuration
- document parsing and post-processing
- line filtering and duplicate suppression
- block reconstruction
- Markdown rendering
- structured document payload assembly

### `src/ocr_service/local_infer.py`

Offline CLI for running OCR on a file or directory of files.

It reuses the same pipeline as the API service and writes:

- document-level Markdown
- page-level Markdown
- optional annotated preview images

### `src/ocr_service/module_benchmark.py`

Benchmark runner for the individual OCR modules.

It measures the time spent in:

- document orientation classification
- document unwarping
- text detection
- textline orientation classification
- text recognition

### `src/ocr_service/__init__.py`

Package marker only.

## Runtime Models

The pipeline loads local PaddleOCR model snapshots from `models/` by default:

- `models/PP-LCNet_x1_0_doc_ori_infer/`
- `models/UVDoc_infer/`
- `models/PP-LCNet_x0_25_textline_ori_infer/`
- `models/PP-OCRv5_mobile_det/`
- `models/PP-OCRv5_mobile_rec/`

These can be overridden with environment variables:

- `OCR_DOC_ORI_MODEL_DIR`
- `OCR_DOC_UNWARP_MODEL_DIR`
- `OCR_TEXTLINE_ORI_MODEL_DIR`
- `OCR_DET_MODEL_DIR`
- `OCR_REC_MODEL_DIR`

## API Surface

### `GET /healthz`

Returns startup state:

```json
{
  "status": "ok",
  "startup_ready": true,
  "startup_error": null
}
```

If startup fails or is still in progress, `status` changes accordingly.

### `POST /v1/ocr`

Accepts multipart upload with a single file field named `image`.

Accepted inputs:

- image uploads whose `content-type` starts with `image/`
- PDF uploads with `content-type == application/pdf`
- PDF uploads whose filename ends in `.pdf`

Rejected inputs:

- any non-image, non-PDF upload returns `400`

Response:

- body: Markdown text
- content type: `text/markdown; charset=utf-8`
- extra header: `X-OCR-Job-ID`

The endpoint also copies the original upload to `data/uploads/` and writes the Markdown output to `data/results/`.

## Pipeline Structure

The main runtime path is:

1. Resolve the uploaded path and verify that it exists.
2. Load image size metadata when applicable.
3. Call PaddleOCR through `self._ocr.predict(str(image_path))`.
4. Extract one or more page payloads from the PaddleOCR response.
5. Convert each page payload into line objects.
6. Apply line filters and warnings.
7. Reconstruct layout rows into blocks.
8. Convert the page to fenced Markdown.
9. Combine pages into document-level Markdown.

### Optional preprocessing stages

The pipeline can enable or disable these stages:

- document orientation classification
- document image unwarping
- textline orientation classification

The default profile is `fast`, which disables the optional stages unless environment flags override them. The `full` profile enables them by default.

### TensorRT and acceleration

The pipeline has a TensorRT path for Jetson-style deployment. It can:

- append the TensorRT Python path when needed
- patch PaddleX TensorRT defaults for the selected submodules
- use TRT engine configs for detection, recognition, textline orientation, document orientation, and document unwarping

Relevant environment variables include:

- `OCR_USE_TENSORRT`
- `OCR_TENSORRT_PYTHON_PATH`
- `OCR_TRT_PROFILE`
- `OCR_TRT_MODULES`
- `OCR_TRT_WORKSPACE_MB`
- `OCR_TRT_DET_MAX_SIDE`
- `OCR_TRT_DET_OPT_SIDE`
- `OCR_TRT_REC_MAX_WIDTH`

## OCR Result Model

`OCRResult` is the main per-page result object.

### Fields

- `raw_text`
- `full_text`
- `normalized_text`
- `markdown_text`
- `lines`
- `blocks`
- `warnings`
- `timings_ms`
- `meta`

### Line model

Each `OCRLine` contains:

- `order`
- `text`
- `normalized_text`
- `det_score`
- `rec_score`
- `polygon`
- `bbox`
- `page_index`
- `accepted`
- `flags`

### Block model

Each `OCRBlock` contains:

- `id`
- `order`
- `kind`
- `text`
- `normalized_text`
- `page_index`
- `line_orders`
- `bbox`
- `confidence`
- `cells`

### Warning model

Warnings are emitted as dictionaries with:

- `code`
- `severity`
- `message`
- `page_index`
- `line_orders`

## Post-processing Rules

The pipeline does more than just forward raw OCR output.

### Line extraction

The pipeline reads these payload fields when present:

- `rec_texts`
- `rec_scores`
- `dt_scores`
- `rec_polys` or `dt_polys`
- `page_index`
- `doc_preprocessor_res`
- `model_settings`
- `text_det_params`

If `rec_texts` is missing, it still builds empty text lines from detected polygons.

### Line filtering

The service applies a few heuristics before block reconstruction:

- empty lines are rejected
- recognition scores below `0.8` trigger a low-confidence warning
- short marker-like strings such as `o`, `x`, `v`, `□`, `○`, `●` trigger a marker warning
- duplicate lines are suppressed when the normalized text matches a recent accepted line and the boxes overlap strongly

### Layout reconstruction

Accepted lines are grouped into rows using:

- polygon geometry when available
- page skew estimation
- median line height to set the row grouping tolerance

Within each row, text is reassembled by estimating horizontal gaps so the Markdown stays closer to the document layout.

## Markdown Output

Per page, the service renders the OCR result as a fenced Markdown text block.

Rules:

- one page -> a single fenced block
- multiple pages -> each page is prefixed with `## Page N`
- pages are separated with `---`
- the original filename is emitted as an HTML comment when available

Example shape:

````markdown
<!-- source: sample.pdf -->

## Page 1

```text
page one text
```

---

## Page 2

```text
page two text
```
````

The actual fence length is increased automatically if the text contains backticks.

## Structured Document Payload

`build_document_payload()` assembles a document-wide structure for downstream consumers.

Top-level keys:

- `raw_text`
- `full_text`
- `normalized_text`
- `markdown_text`
- `pages`
- `lines`
- `blocks`
- `warnings`
- `timings_ms`
- `meta`

Important nested fields:

- `pages`: list of per-page `OCRResult.to_dict()` objects
- `lines`: flattened line list across pages
- `blocks`: flattened block list across pages
- `warnings`: flattened warning list across pages
- `timings_ms.document_total`: sum of per-page totals
- `timings_ms.page_count`: number of pages
- `meta.page_count`
- `meta.original_filename`
- `meta.content_type`

## Timing Metadata

Per page, `timings_ms` contains:

- `preprocess`
- `pipeline`
- `postprocess`
- `total`

The document-wide payload also records aggregate totals.

## Startup Behavior

Startup is controlled by two boolean environment flags:

- `OCR_PRELOAD_PIPELINE_ON_STARTUP`
- `OCR_WARMUP_ON_STARTUP`

Default behavior:

- preload the pipeline
- run a small synthetic warmup inference
- fail startup if initialization or warmup fails

## Local CLI Behavior

`local_infer.py` is useful for offline validation and debugging.

Supported input types:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`
- `.pdf`

Output layout:

- `output/documents/` for document-level Markdown
- `output/markdown/` for page-level Markdown
- `output/images/` for annotated previews of non-PDF images

Useful CLI options include:

- `--profile`
- `--engine`
- `--device`
- `--use-tensorrt`
- `--trt-profile`
- `--trt-modules`
- `--clear-trt-cache`
- `--warmup-runs`

## Benchmark Utility

`module_benchmark.py` exists to measure the OCR stack in a more granular way than the API service.

It reports metrics such as:

- module latency in milliseconds
- number of detected boxes
- number of recognized lines
- recognition batch counts
- crop ratio statistics
- recognition input width statistics

## Operational Notes

- The API returns Markdown, not JSON.
- The structured `OCRResult` objects are internal, but the code already supports converting them to dictionaries if a JSON payload is needed later.
- `web-app` is the intended caller for `POST /v1/ocr`.
- The service remains private; only `web-app` should be exposed publicly.
