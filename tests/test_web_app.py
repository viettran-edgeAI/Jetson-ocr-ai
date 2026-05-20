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

    def test_bulk_delete_sessions_removes_multiple_entries(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            for session_id in ("session-1", "session-2"):
                image_path = Path(tmpdir) / f"{session_id}.png"
                image_path.write_bytes(b"png-bytes")
                markdown_path = Path(tmpdir) / f"{session_id}.md"
                markdown_path.write_text("content", encoding="utf-8")
                temp_store.create_session(
                    session_id=session_id,
                    filename=f"{session_id}.png",
                    content_type="image/png",
                    original_path=image_path,
                    created_at=now,
                )
                temp_store.update_session(
                    session_id,
                    now,
                    status="ocr_complete",
                    ocr_markdown_path=str(markdown_path),
                )

            web_main.store = temp_store
            try:
                response = asyncio.run(
                    web_main.bulk_delete_sessions(
                        web_main.BulkDeleteSessionsRequest(session_ids=["session-1", "session-2", "session-2"])
                    )
                )
            finally:
                web_main.store = original_store

            remaining = [temp_store.get_session("session-1"), temp_store.get_session("session-2")]

        self.assertEqual(response["deleted_count"], 2)
        self.assertEqual(response["missing_ids"], [])
        self.assertEqual(remaining, [None, None])

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

    def test_message_history_for_llm_keeps_only_chat_messages(self) -> None:
        history = web_main.message_history_for_llm(
            [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": " First question "},
                {"role": "assistant", "content": "First answer"},
                {"role": "assistant", "content": ""},
            ]
        )

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
        )

    def test_ask_session_sends_prior_messages_to_llm(self) -> None:
        original_store = web_main.store
        original_post_answer = web_main.post_answer_request
        original_to_thread = web_main.asyncio.to_thread
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            markdown_path = Path(tmpdir) / "scan.md"
            markdown_path.write_text("Invoice total is $42.", encoding="utf-8")
            temp_store.create_session(
                session_id="session-1",
                filename="scan.png",
                content_type="image/png",
                original_path=Path(tmpdir) / "scan.png",
                created_at=now,
            )
            temp_store.update_session(
                "session-1",
                now,
                status="answered",
                ocr_markdown_path=str(markdown_path),
            )
            temp_store.add_message(
                session_id="session-1",
                role="user",
                content="What is the total?",
                created_at=now,
            )
            temp_store.add_message(
                session_id="session-1",
                role="assistant",
                content="The total is $42.",
                created_at=now,
            )
            captured: dict[str, object] = {}

            def fake_post_answer(
                markdown: str,
                prompt: str,
                conversation_history: list[dict[str, str]] | None = None,
            ) -> dict[str, object]:
                captured["markdown"] = markdown
                captured["prompt"] = prompt
                captured["conversation_history"] = conversation_history
                return {"answer": "Because the OCR text says so.", "elapsed_ms": 1}

            async def immediate_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            web_main.store = temp_store
            web_main.post_answer_request = fake_post_answer
            web_main.asyncio.to_thread = immediate_to_thread
            try:
                response = asyncio.run(
                    web_main.ask_session("session-1", web_main.AskRequest(prompt="Why?"))
                )
            finally:
                web_main.store = original_store
                web_main.post_answer_request = original_post_answer
                web_main.asyncio.to_thread = original_to_thread

        self.assertEqual(captured["markdown"], "Invoice total is $42.")
        self.assertEqual(captured["prompt"], "Why?")
        self.assertEqual(
            captured["conversation_history"],
            [
                {"role": "user", "content": "What is the total?"},
                {"role": "assistant", "content": "The total is $42."},
            ],
        )
        self.assertEqual(response["messages"][-1]["content"], "Because the OCR text says so.")

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
