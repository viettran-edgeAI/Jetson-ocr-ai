from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

import ocr_service.main as ocr_main
from ocr_service.models import OCRResult
from ocr_service.pipeline import OCRPipeline


class StubPipeline:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.last_path: Path | None = None

    def predict_document(self, image_path: Path) -> list[OCRResult]:
        self.last_path = Path(image_path)
        return [
            OCRResult(
                raw_text=self.markdown,
                full_text=self.markdown,
                normalized_text=self.markdown,
                markdown_text=self.markdown,
                lines=[],
                blocks=[],
                warnings=[],
                timings_ms={},
                meta={"page_index": 0, "page_number": 1},
                regions=[],
                formulas=[],
            )
        ]

    def build_document_markdown(
        self,
        results: list[OCRResult],
        *,
        original_filename: str | None = None,
        content_type: str | None = None,
    ) -> str:
        del content_type
        body = results[0].markdown_text
        if original_filename:
            return f"<!-- source: {original_filename} -->\n\n{body}\n"
        return f"{body}\n"


class OCRServiceApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.upload_dir = self.data_root / "uploads"
        self.result_dir = self.data_root / "results"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.env_patch = patch.dict(
            os.environ,
            {
                "OCR_PRELOAD_PIPELINE_ON_STARTUP": "0",
                "OCR_WARMUP_ON_STARTUP": "0",
                "OCR_DATA_DIR": self.temp_dir.name,
            },
            clear=False,
        )
        self.path_patch = patch.multiple(
            ocr_main,
            DATA_DIR=self.data_root,
            UPLOAD_DIR=self.upload_dir,
            RESULT_DIR=self.result_dir,
            WARMUP_IMAGE_PATH=self.data_root / ".ocr_warmup.png",
        )
        self.env_patch.start()
        self.path_patch.start()

    def tearDown(self) -> None:
        self.path_patch.stop()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_image_upload_returns_markdown_and_job_id(self) -> None:
        image = Image.new("RGB", (8, 8), color=(255, 255, 255))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        payload.seek(0)
        upload = UploadFile(
            file=io.BytesIO(payload.getvalue()),
            filename="sample.png",
            headers=Headers({"content-type": "image/png"}),
        )

        stub = StubPipeline("image markdown")
        with patch("ocr_service.main.get_pipeline", return_value=stub):
            response = asyncio.run(ocr_main.ocr_image(upload))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body.decode("utf-8"), "<!-- source: sample.png -->\n\nimage markdown\n")
        self.assertTrue(response.headers.get("x-ocr-job-id"))
        self.assertIsNotNone(stub.last_path)
        self.assertEqual(stub.last_path.suffix, ".png")
        self.assertTrue(stub.last_path.exists())

    def test_pdf_upload_returns_markdown(self) -> None:
        upload = UploadFile(
            file=io.BytesIO(b"%PDF-1.4\n%"),
            filename="sample.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )
        stub = StubPipeline("pdf markdown")
        with patch("ocr_service.main.get_pipeline", return_value=stub):
            response = asyncio.run(ocr_main.ocr_image(upload))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body.decode("utf-8"), "<!-- source: sample.pdf -->\n\npdf markdown\n")
        self.assertIsNotNone(stub.last_path)
        self.assertEqual(stub.last_path.suffix, ".pdf")

    def test_unsupported_upload_type_returns_400(self) -> None:
        upload = UploadFile(
            file=io.BytesIO(b"not supported"),
            filename="notes.txt",
            headers=Headers({"content-type": "text/plain"}),
        )
        with self.assertRaises(HTTPException) as context:
            asyncio.run(ocr_main.ocr_image(upload))

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn(
            "Only image and PDF uploads are supported",
            str(context.exception.detail),
        )


class OCRPipelineMarkdownTests(unittest.TestCase):
    def test_build_document_markdown_formats_multi_page_output(self) -> None:
        pipeline = OCRPipeline.__new__(OCRPipeline)
        results = [
            OCRResult(
                raw_text="first",
                full_text="first",
                normalized_text="first",
                markdown_text="First page",
                lines=[],
                blocks=[],
                warnings=[],
                timings_ms={},
                meta={"page_index": 0},
                regions=[],
                formulas=[],
            ),
            OCRResult(
                raw_text="second",
                full_text="second",
                normalized_text="second",
                markdown_text="Second page",
                lines=[],
                blocks=[],
                warnings=[],
                timings_ms={},
                meta={"page_index": 1},
                regions=[],
                formulas=[],
            ),
        ]

        markdown = OCRPipeline.build_document_markdown(
            pipeline,
            results,
            original_filename="multi.pdf",
            content_type="application/pdf",
        )

        self.assertEqual(
            markdown,
            "<!-- source: multi.pdf -->\n\n## Page 1\n\nFirst page\n\n---\n\n## Page 2\n\nSecond page\n",
        )


if __name__ == "__main__":
    unittest.main()
