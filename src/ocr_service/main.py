from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .pipeline import OCRPipeline


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("OCR_DATA_DIR", APP_ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"

for path in (UPLOAD_DIR, RESULT_DIR):
    path.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Jetson OCR", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: OCRPipeline | None = None


def get_pipeline() -> OCRPipeline:
    global pipeline
    if pipeline is None:
        pipeline = OCRPipeline()
    return pipeline


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ocr", response_class=PlainTextResponse)
async def ocr_image(image: UploadFile = File(...)) -> PlainTextResponse:
    content_type = image.content_type or ""
    is_image = content_type.startswith("image/")
    is_pdf = content_type == "application/pdf" or (image.filename or "").lower().endswith(".pdf")
    if not is_image and not is_pdf:
        raise HTTPException(status_code=400, detail="Only image and PDF uploads are supported.")

    job_id = uuid.uuid4().hex
    suffix = Path(image.filename or "upload").suffix or (".pdf" if is_pdf else ".png")
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    result_path = RESULT_DIR / f"{job_id}.md"

    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    ocr = get_pipeline()
    page_results = ocr.predict_document(upload_path)
    markdown = ocr.build_document_markdown(
        page_results,
        original_filename=image.filename,
        content_type=image.content_type,
    )

    result_path.write_text(markdown, encoding="utf-8")
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"X-OCR-Job-ID": job_id},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "ocr_service.main:app",
        host=os.environ.get("OCR_HOST", "0.0.0.0"),
        port=int(os.environ.get("OCR_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
