from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("WEB_APP_DATA_DIR", tempfile.mkdtemp(prefix="ocr-web-test-"))

from fastapi import HTTPException

from web_app import main as web_main
from web_app.store import SessionStore


class WebAppSessionTests(unittest.TestCase):
    def test_upload_validation_accepts_only_supported_formats(self) -> None:
        web_main.validate_upload("scan.png", "image/png")
        web_main.validate_upload("scan.jpg", "image/jpeg")
        web_main.validate_upload("scan.jpeg", "image/jpeg")
        web_main.validate_upload("scan.pdf", "application/pdf")

        with self.assertRaises(HTTPException):
            web_main.validate_upload("notes.txt", "text/plain")

    def test_multipart_body_matches_ocr_service_contract(self) -> None:
        body = web_main.build_multipart_file_body(
            field_name="image",
            filename="scan.png",
            content_type="image/png",
            body=b"png-bytes",
            boundary="boundary",
        )

        self.assertIn(b'name="image"; filename="scan.png"', body)
        self.assertIn(b"Content-Type: image/png", body)
        self.assertTrue(body.endswith(b"\r\n--boundary--\r\n"))

    def test_index_uses_versioned_static_asset_urls(self) -> None:
        response = asyncio.run(web_main.index())
        html = response.body.decode("utf-8")

        self.assertIn('/static/styles.css?v=', html)
        self.assertIn('/static/app.js?v=', html)

    def test_original_file_is_served_inline(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            image_path = Path(tmpdir) / "scan.png"
            image_path.write_bytes(b"png-bytes")
            now = "2026-05-20T00:00:00+00:00"
            temp_store.create_session(
                session_id="session-1",
                filename="scan.png",
                content_type="image/png",
                original_path=image_path,
                created_at=now,
            )
            web_main.store = temp_store
            try:
                response = asyncio.run(web_main.get_original("session-1"))
            finally:
                web_main.store = original_store

        self.assertEqual(response.media_type, "image/png")
        self.assertIn('inline; filename="scan.png"', response.headers["content-disposition"])

    def test_session_store_persists_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            store.create_session(
                session_id="session-1",
                filename="scan.pdf",
                content_type="application/pdf",
                original_path=Path(tmpdir) / "scan.pdf",
                created_at=now,
            )
            store.update_session(
                "session-1",
                now,
                status="ocr_complete",
                page_count=2,
                ocr_markdown_path=str(Path(tmpdir) / "scan.md"),
            )
            store.add_message(
                session_id="session-1",
                role="assistant",
                content="Answer",
                created_at=now,
                elapsed_ms=42,
            )

            session = store.get_session("session-1")

        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session["status"], "ocr_complete")
        self.assertEqual(session["page_count"], 2)
        self.assertEqual(session["messages"][0]["content"], "Answer")

    def test_session_store_renames_and_deletes_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            store.create_session(
                session_id="session-1",
                filename="scan.pdf",
                content_type="application/pdf",
                original_path=Path(tmpdir) / "scan.pdf",
                created_at=now,
            )

            store.rename_session("session-1", "renamed.pdf", now)
            renamed = store.get_session("session-1")
            deleted = store.delete_session("session-1")
            after_delete = store.get_session("session-1")

        self.assertIsNotNone(renamed)
        assert renamed is not None
        self.assertEqual(renamed["filename"], "renamed.pdf")
        self.assertIsNotNone(deleted)
        self.assertIsNone(after_delete)


if __name__ == "__main__":
    unittest.main()
