from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("WEB_APP_DATA_DIR", tempfile.mkdtemp(prefix="ocr-web-test-"))

from fastapi import HTTPException, Response

from web_app import main as web_main
from web_app.auth import Identity
from web_app.store import SessionStore


class FakeUploadFile:
    def __init__(self, filename: str, content_type: str, body: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self.body = body

    async def read(self) -> bytes:
        return self.body


class WebAppSessionTests(unittest.TestCase):
    def solve_bot_question(self, question: str) -> str:
        left, right = [int(part.strip()) for part in question.split("+", 1)]
        return str(left + right)

    def complete_signup(
        self,
        *,
        email: str = "user@example.com",
        username: str = "tester",
        password: str = "password123",
        avatar_key: str | None = None,
    ) -> dict[str, object]:
        sent_codes: dict[str, str] = {}
        original_send_email = web_main.send_verification_email

        def fake_send_email(target_email: str, code: str) -> None:
            sent_codes[target_email] = code

        web_main.send_verification_email = fake_send_email
        try:
            start = asyncio.run(
                web_main.signup_start(
                    web_main.SignupStartRequest(
                        email=email,
                        username=username,
                        password=password,
                        avatar_key=avatar_key,
                    )
                )
            )
            asyncio.run(
                web_main.signup_verify_email(
                    web_main.SignupVerifyEmailRequest(
                        pending_id=start["pending_id"],
                        code=sent_codes[email],
                    )
                )
            )
            completed = asyncio.run(
                web_main.signup_complete(
                    web_main.SignupCompleteRequest(
                        pending_id=start["pending_id"],
                    ),
                    Response(),
                )
            )
        finally:
            web_main.send_verification_email = original_send_email
        return completed

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

    def test_create_chat_session_without_document(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            try:
                response = asyncio.run(web_main.create_chat_session())
            finally:
                web_main.store = original_store

        self.assertEqual(response["filename"], "Untitled chat")
        self.assertEqual(response["status"], "chat_ready")
        self.assertFalse(response["has_document"])
        self.assertEqual(response["ocr_markdown"], "")
        self.assertEqual(response["messages"], [])

    def test_signup_activates_owner_placeholder(self) -> None:
        original_store = web_main.store
        original_owner_email = web_main.OWNER_EMAIL
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            web_main.OWNER_EMAIL = "owner@example.com"
            temp_store.ensure_owner_placeholder(
                user_id="owner-user",
                email="owner@example.com",
                created_at="2026-05-20T00:00:00+00:00",
            )
            try:
                response = self.complete_signup(
                    email="owner@example.com",
                    username="owner_admin",
                    password="password123",
                )
            finally:
                web_main.store = original_store
                web_main.OWNER_EMAIL = original_owner_email

        self.assertEqual(response["user"]["tier"], "owner")
        self.assertEqual(response["user"]["username"], "owner_admin")
        self.assertEqual(response["rate_limit"]["unlimited"], True)

    def test_signup_defaults_avatar_when_omitted(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            try:
                response = self.complete_signup(email="avatar@example.com", username="avataruser")
            finally:
                web_main.store = original_store

        self.assertEqual(response["user"]["avatar_key"], web_main.DEFAULT_AVATAR_KEY)

    def test_signup_fails_without_smtp_when_debug_codes_disabled(self) -> None:
        original_smtp_host = web_main.SMTP_HOST
        original_debug_codes = web_main.AUTH_DEBUG_CODES
        try:
            web_main.SMTP_HOST = ""
            web_main.AUTH_DEBUG_CODES = False
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    web_main.signup_start(
                        web_main.SignupStartRequest(
                            email="smtp@example.com",
                            username="smtpuser",
                            password="password123",
                        )
                    )
                )
        finally:
            web_main.SMTP_HOST = original_smtp_host
            web_main.AUTH_DEBUG_CODES = original_debug_codes

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("SMTP", raised.exception.detail)

    def test_signup_restart_allowed_for_incomplete_account_without_password_match(self) -> None:
        original_store = web_main.store
        original_send_email = web_main.send_verification_email
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-21T00:00:00+00:00"
            temp_store.create_user(
                user_id="user-1",
                email="owner@example.com",
                username="owner_start",
                password_hash=web_main.hash_password("old-password"),
                avatar_key=web_main.DEFAULT_AVATAR_KEY,
                tier="free",
                disabled=False,
                created_at=now,
            )
            sent_codes: dict[str, str] = {}

            def fake_send_email(target_email: str, code: str) -> None:
                sent_codes[target_email] = code

            web_main.store = temp_store
            web_main.send_verification_email = fake_send_email
            try:
                response = asyncio.run(
                    web_main.signup_start(
                        web_main.SignupStartRequest(
                            email="owner@example.com",
                            username="owner_start",
                            password="new-password-123",
                        )
                    )
                )
            finally:
                web_main.store = original_store
                web_main.send_verification_email = original_send_email

        self.assertEqual(response["status"], "verification_sent")
        self.assertIn("owner@example.com", sent_codes)

    def test_identity_serialization_uses_username_avatar_and_hides_email(self) -> None:
        identity = Identity(
            owner_type="user",
            owner_id="user-1",
            tier="pro",
            email="hidden@example.com",
            username="displayname",
            avatar_key="sage",
            is_authenticated=True,
        )

        response = web_main.serialize_identity(identity)

        self.assertNotIn("email", response["identity"])
        self.assertEqual(response["identity"]["username"], "displayname")
        self.assertEqual(response["identity"]["avatar_key"], "sage")
        self.assertEqual(response["identity"]["tier_color"], "dark-green")

    def test_account_details_include_email_and_usage_for_authenticated_user(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            try:
                signup = self.complete_signup(email="account@example.com", username="accountuser")
                identity = Identity(
                    owner_type="user",
                    owner_id=str(signup["user"]["id"]),
                    tier="free",
                    email="account@example.com",
                    username="accountuser",
                    is_authenticated=True,
                )
                response = asyncio.run(web_main.account_details(identity=identity))
            finally:
                web_main.store = original_store

        self.assertEqual(response["account"]["email"], "account@example.com")
        self.assertEqual(response["account"]["tier"], "free")
        self.assertEqual(response["account"]["usage"]["limit"], web_main.OCR_UPLOAD_LIMITS["free"])

    def test_login_succeeds_without_two_factor(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            try:
                self.complete_signup(email="login@example.com", username="loginuser")
                login = asyncio.run(
                    web_main.login(
                        web_main.AuthRequest(email="login@example.com", password="password123"),
                        Response(),
                    )
                )
            finally:
                web_main.store = original_store

        self.assertEqual(login["requires_two_factor"], False)
        self.assertEqual(login["user"]["username"], "loginuser")

    def test_signup_allows_duplicate_usernames(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            web_main.store = temp_store
            try:
                first = self.complete_signup(email="dup1@example.com", username="same_name")
                second = self.complete_signup(email="dup2@example.com", username="same_name")
            finally:
                web_main.store = original_store

        self.assertEqual(first["user"]["username"], "same_name")
        self.assertEqual(second["user"]["username"], "same_name")
        self.assertNotEqual(first["user"]["id"], second["user"]["id"])

    def test_session_routes_are_scoped_to_identity(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            owner_a = Identity(owner_type="user", owner_id="user-a", tier="free", is_authenticated=True)
            owner_b = Identity(owner_type="user", owner_id="user-b", tier="free", is_authenticated=True)
            temp_store.create_session(
                session_id="session-a",
                owner_type=owner_a.owner_type,
                owner_id=owner_a.owner_id,
                filename="a.png",
                content_type="image/png",
                original_path=Path(tmpdir) / "a.png",
                created_at=now,
            )
            temp_store.create_session(
                session_id="session-b",
                owner_type=owner_b.owner_type,
                owner_id=owner_b.owner_id,
                filename="b.png",
                content_type="image/png",
                original_path=Path(tmpdir) / "b.png",
                created_at=now,
            )
            web_main.store = temp_store
            try:
                recent = asyncio.run(web_main.recent_sessions(identity=owner_a))
                with self.assertRaises(HTTPException):
                    asyncio.run(web_main.get_session("session-b", identity=owner_a))
            finally:
                web_main.store = original_store

        self.assertEqual([session["id"] for session in recent["sessions"]], ["session-a"])

    def test_ocr_upload_rate_limit_blocks_guest_after_hourly_limit(self) -> None:
        original_store = web_main.store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            identity = Identity(owner_type="guest", owner_id="guest-a", tier="guest")
            now = datetime.now(timezone.utc)
            for index in range(web_main.OCR_UPLOAD_LIMITS["guest"]):
                temp_store.add_rate_limit_event(
                    owner_type=identity.owner_type,
                    owner_id=identity.owner_id,
                    action=web_main.OCR_UPLOAD_ACTION,
                    created_at=(now - timedelta(minutes=index)).isoformat(),
                )
            web_main.store = temp_store
            try:
                response = asyncio.run(
                    web_main.upload_document(
                        file=FakeUploadFile("scan.png", "image/png", b"png-bytes"),
                        identity=identity,
                    )
                )
            finally:
                web_main.store = original_store

        self.assertEqual(response.status_code, 429)

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

    def test_ask_session_allows_chat_without_ocr_markdown(self) -> None:
        original_store = web_main.store
        original_post_answer = web_main.post_answer_request
        original_to_thread = web_main.asyncio.to_thread
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            temp_store.create_session(
                session_id="session-1",
                filename="Untitled chat",
                content_type=web_main.CHAT_CONTENT_TYPE,
                original_path="",
                created_at=now,
                status="chat_ready",
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
                return {"answer": "General chat response.", "elapsed_ms": 1}

            async def immediate_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            web_main.store = temp_store
            web_main.post_answer_request = fake_post_answer
            web_main.asyncio.to_thread = immediate_to_thread
            try:
                response = asyncio.run(
                    web_main.ask_session("session-1", web_main.AskRequest(prompt="Hello"))
                )
            finally:
                web_main.store = original_store
                web_main.post_answer_request = original_post_answer
                web_main.asyncio.to_thread = original_to_thread

        self.assertEqual(captured["markdown"], "")
        self.assertEqual(captured["prompt"], "Hello")
        self.assertEqual(captured["conversation_history"], [])
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["messages"][-1]["content"], "General chat response.")

    def test_upload_can_attach_document_to_existing_chat_session(self) -> None:
        original_store = web_main.store
        original_post_ocr = web_main.post_ocr_request
        original_to_thread = web_main.asyncio.to_thread
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_store = SessionStore(Path(tmpdir) / "sessions.sqlite3")
            now = "2026-05-20T00:00:00+00:00"
            temp_store.create_session(
                session_id="session-1",
                filename="Untitled chat",
                content_type=web_main.CHAT_CONTENT_TYPE,
                original_path="",
                created_at=now,
                status="answered",
            )
            temp_store.add_message(
                session_id="session-1",
                role="user",
                content="Hello before upload",
                created_at=now,
            )
            captured: dict[str, object] = {}

            def fake_post_ocr(filename: str, content_type: str, body: bytes) -> str:
                captured["filename"] = filename
                captured["content_type"] = content_type
                captured["body"] = body
                return "OCR text from attached document."

            async def immediate_to_thread(function, *args, **kwargs):
                return function(*args, **kwargs)

            upload = FakeUploadFile(
                filename="scan.png",
                content_type="image/png",
                body=b"png-bytes",
            )

            web_main.store = temp_store
            web_main.post_ocr_request = fake_post_ocr
            web_main.asyncio.to_thread = immediate_to_thread
            try:
                response = asyncio.run(web_main.upload_document(file=upload, session_id="session-1"))
            finally:
                web_main.store = original_store
                web_main.post_ocr_request = original_post_ocr
                web_main.asyncio.to_thread = original_to_thread

        self.assertEqual(captured["filename"], "scan.png")
        self.assertEqual(captured["content_type"], "image/png")
        self.assertEqual(captured["body"], b"png-bytes")
        self.assertEqual(response["id"], "session-1")
        self.assertTrue(response["has_document"])
        self.assertEqual(response["filename"], "scan.png")
        self.assertEqual(response["ocr_markdown"], "OCR text from attached document.")
        self.assertEqual(response["messages"][0]["content"], "Hello before upload")

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

    def test_session_store_migrates_legacy_sessions_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE sessions (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        original_path TEXT NOT NULL,
                        ocr_markdown_path TEXT,
                        status TEXT NOT NULL,
                        error TEXT,
                        page_count INTEGER,
                        ocr_elapsed_ms INTEGER,
                        answer_elapsed_ms INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

            store = SessionStore(db_path)
            now = "2026-05-21T00:00:00+00:00"
            store.create_session(
                session_id="legacy-1",
                owner_type="user",
                owner_id="owner-1",
                filename="legacy.pdf",
                content_type="application/pdf",
                original_path=Path(tmpdir) / "legacy.pdf",
                created_at=now,
            )
            session = store.get_session("legacy-1", owner_type="user", owner_id="owner-1")

        self.assertIsNotNone(session)

    def test_session_store_backfills_legacy_user_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE users (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        password_hash TEXT,
                        tier TEXT NOT NULL DEFAULT 'free',
                        disabled INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    INSERT INTO users (
                        id, email, password_hash, tier, disabled, created_at, updated_at
                    )
                    VALUES (
                        'user-1', 'legacy.user@example.com', 'hash', 'free', 0,
                        '2026-05-20T00:00:00+00:00', '2026-05-20T00:00:00+00:00'
                    );
                    """
                )

            store = SessionStore(db_path)
            user = store.get_user_by_email("legacy.user@example.com")

        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user["username"], "legacy_user")
        self.assertEqual(user["avatar_key"], "default")


if __name__ == "__main__":
    unittest.main()
