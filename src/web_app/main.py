from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import os
import re
import secrets
import smtplib
import time
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from urllib import error, request

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import (
    AUTH_COOKIE_NAME,
    AUTH_SESSION_DAYS,
    GUEST_COOKIE_NAME,
    Identity,
    auth_cookie_max_age,
    auth_token_from_request,
    clear_cookie,
    guest_cookie_max_age,
    guest_id_from_request,
    hash_password,
    hash_token,
    new_guest_id,
    new_numeric_code,
    new_token,
    normalize_email,
    set_signed_cookie,
    utc_expiry,
    validate_password,
    validate_secret_key,
    verify_password,
)
from .store import SessionStore


APP_ROOT = Path(__file__).resolve().parents[2]
WEB_DATA_DIR = Path(os.environ.get("WEB_APP_DATA_DIR", APP_ROOT / "data" / "web_app"))
UPLOAD_DIR = WEB_DATA_DIR / "uploads"
OCR_DIR = WEB_DATA_DIR / "ocr_markdown"
DB_PATH = WEB_DATA_DIR / "sessions.sqlite3"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_TEMPLATE_PATH = STATIC_DIR / "index.html"

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr:8000")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm:8081")
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("WEB_REQUEST_TIMEOUT_SECONDS", "360"))
SECRET_KEY = os.environ.get("WEB_APP_SECRET_KEY", "dev-insecure-change-me")
OWNER_EMAIL = normalize_email(os.environ.get("WEB_APP_OWNER_EMAIL", ""))
COOKIE_SECURE = os.environ.get("WEB_APP_COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes"}
DEFAULT_AVATAR_KEY = os.environ.get("WEB_APP_DEFAULT_AVATAR_KEY", "atlas").strip() or "atlas"
AUTH_OUTBOX_PATH = Path(os.environ.get("WEB_APP_AUTH_OUTBOX_PATH", WEB_DATA_DIR / "auth_outbox.jsonl"))
AUTH_DEBUG_CODES = os.environ.get("WEB_APP_AUTH_DEBUG_CODES", "0").strip().lower() in {"1", "true", "yes"}
SMTP_HOST = os.environ.get("WEB_APP_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("WEB_APP_SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("WEB_APP_SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("WEB_APP_SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("WEB_APP_SMTP_FROM", SMTP_USERNAME or "no-reply@jetsonocrai.cc").strip()
SMTP_STARTTLS = os.environ.get("WEB_APP_SMTP_STARTTLS", "1").strip().lower() in {"1", "true", "yes"}

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "application/pdf",
}
CHAT_CONTENT_TYPE = "application/x-chat-session"
CHAT_SESSION_FILENAME = "Untitled chat"
ASKABLE_STATUSES = {"chat_ready", "ocr_complete", "answered", "llm_failed", "ocr_failed"}
RECENT_SESSIONS_DEFAULT_LIMIT = 8
MAX_STORED_SESSIONS = 50
OCR_UPLOAD_ACTION = "ocr_upload"
OCR_UPLOAD_LIMITS = {
    "guest": 10,
    "free": 50,
    "pro": 2000,
}
AVATAR_KEYS = {"atlas", "nova", "sage", "ember", "orbit", "pixel"}
RESERVED_USERNAMES = {"admin", "api", "auth", "guest", "owner", "root", "support", "system"}
PENDING_SIGNUP_HOURS = 2
BOT_CHALLENGE_MINUTES = 10
AUTH_ATTEMPT_LIMIT = 30
AUTH_ATTEMPT_ACTION = "auth_attempt"

for path in (UPLOAD_DIR, OCR_DIR):
    path.mkdir(parents=True, exist_ok=True)

validate_secret_key(SECRET_KEY)
store = SessionStore(DB_PATH)
app = FastAPI(title="OCR AI Assistant", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def disable_cache_for_ui(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class AskRequest(BaseModel):
    prompt: str = Field(..., min_length=0, max_length=2000)
    mode: str | None = Field(default=None, max_length=64)
    thinking_mode: Literal["fast", "thinking"] = "fast"


class RenameSessionRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=160)


class BulkDeleteSessionsRequest(BaseModel):
    session_ids: list[str] = Field(..., min_length=1, max_length=200)


class BotChallengeProof(BaseModel):
    challenge_id: str = Field(..., min_length=1, max_length=128)
    answer: str = Field(..., min_length=1, max_length=32)
    website: str | None = Field(default="", max_length=200)


class SignupStartRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=256)
    avatar_key: str | None = Field(default=None, max_length=32)
    website: str | None = Field(default="", max_length=200)
    bot_challenge: BotChallengeProof | None = None


class SignupVerifyEmailRequest(BaseModel):
    pending_id: str = Field(..., min_length=1, max_length=128)
    code: str = Field(..., min_length=4, max_length=32)


class SignupCompleteRequest(BaseModel):
    pending_id: str = Field(..., min_length=1, max_length=128)


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=256)


async def current_identity(request: Request, response: Response) -> Identity:
    token = auth_token_from_request(request, SECRET_KEY)
    if token:
        user = store.get_user_by_auth_token_hash(hash_token(token), utc_now())
        if user is not None:
            return identity_from_user(user)
        clear_cookie(response, AUTH_COOKIE_NAME, secure=COOKIE_SECURE)

    guest_id = guest_id_from_request(request, SECRET_KEY)
    if not guest_id:
        guest_id = new_guest_id()
        set_signed_cookie(
            response,
            name=GUEST_COOKIE_NAME,
            value=guest_id,
            secret_key=SECRET_KEY,
            max_age=guest_cookie_max_age(),
            secure=COOKIE_SECURE,
        )
    return Identity(
        owner_type="guest",
        owner_id=guest_id,
        tier="guest",
        username="Guest",
        avatar_key="guest",
        is_authenticated=False,
    )


@app.get("/")
async def index() -> HTMLResponse:
    asset_version = static_asset_version()
    html = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("/static/styles.css", f"/static/styles.css?v={asset_version}")
    html = html.replace("/static/app.js", f"/static/app.js?v={asset_version}")
    return HTMLResponse(html)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/me")
async def auth_me(
    response: Response,
    identity: Identity = Depends(current_identity),
) -> dict[str, Any]:
    identity = coerce_identity(identity)
    return serialize_identity(identity)


@app.get("/auth/bot-challenge")
async def bot_challenge() -> dict[str, Any]:
    return create_bot_challenge_response()


@app.post("/auth/signup/start")
async def signup_start(signup: SignupStartRequest) -> dict[str, Any]:
    email = normalize_signup_email(signup.email)
    username = normalize_username(signup.username)
    avatar_key = normalize_avatar_key(signup.avatar_key)
    validate_password(signup.password)
    assert_auth_attempt_allowed(email)
    verify_signup_honeypot(signup.website)
    if signup.bot_challenge is not None:
        verify_bot_challenge(signup.bot_challenge)

    now = utc_now()
    store.prune_pending_signups(now)
    existing_user = store.get_user_by_email(email)
    activate_user_id: str | None = None
    if existing_user is not None:
        if can_start_upgrade_for_user(existing_user, signup.password):
            activate_user_id = str(existing_user["id"])
        else:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

    code = new_numeric_code()
    tier = "owner" if OWNER_EMAIL and email == OWNER_EMAIL else "free"
    if existing_user is not None:
        tier = "owner" if OWNER_EMAIL and email == OWNER_EMAIL else str(existing_user.get("tier") or tier)
    pending = store.upsert_pending_signup(
        pending_id=uuid.uuid4().hex,
        email=email,
        username=username,
        password_hash=hash_password(signup.password),
        avatar_key=avatar_key,
        tier=tier,
        email_code_hash=hash_token(code),
        activate_user_id=activate_user_id,
        bot_passed_at=now,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=PENDING_SIGNUP_HOURS)).isoformat(),
        created_at=now,
    )
    send_verification_email(email, code)
    response: dict[str, Any] = {
        "pending_id": pending["id"],
        "status": "verification_sent",
        "expires_at": pending["expires_at"],
    }
    if AUTH_DEBUG_CODES:
        response["verification_code"] = code
    return response


@app.post("/auth/signup/verify-email")
async def signup_verify_email(request_body: SignupVerifyEmailRequest) -> dict[str, Any]:
    pending = require_pending_signup(request_body.pending_id)
    if not hmac_safe_hash_match(request_body.code, str(pending["email_code_hash"])):
        raise HTTPException(status_code=400, detail="Invalid verification code.")
    updated = store.mark_pending_email_verified(pending_id=pending["id"], verified_at=utc_now())
    return {"pending_id": updated["id"], "status": "email_verified"}


@app.post("/auth/signup/complete")
async def signup_complete(request_body: SignupCompleteRequest, response: Response) -> dict[str, Any]:
    pending = require_pending_signup(request_body.pending_id)
    if not pending.get("email_verified_at"):
        raise HTTPException(status_code=409, detail="Verify your email before completing signup.")

    user = finalize_pending_signup(pending)
    store.delete_pending_signup(pending["id"])
    issue_auth_cookie(response, user["id"])
    return {
        "user": serialize_user(user),
        "rate_limit": rate_limit_status(identity_from_user(user)),
    }


@app.post("/auth/login")
async def login(auth: AuthRequest, response: Response) -> dict[str, Any]:
    email = normalize_signup_email(auth.email)
    assert_auth_attempt_allowed(email)
    user = store.get_user_by_email(email)
    if (
        user is None
        or user.get("disabled")
        or not user.get("password_hash")
        or not verify_password(auth.password, str(user["password_hash"]))
    ):
        record_auth_attempt(email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if OWNER_EMAIL and user["email"] == OWNER_EMAIL and user["tier"] != "owner":
        user = store.update_user_tier(user_id=user["id"], tier="owner", updated_at=utc_now())
    if account_requires_upgrade(user):
        raise HTTPException(
            status_code=403,
            detail="Account requires email verification. Start signup again with this email.",
        )
    now = utc_now()
    store.record_successful_login(user_id=user["id"], logged_in_at=now)
    issue_auth_cookie(response, user["id"])
    return {"requires_two_factor": False, "user": serialize_user(user), "rate_limit": rate_limit_status(identity_from_user(user))}


@app.post("/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, str]:
    token = auth_token_from_request(request, SECRET_KEY)
    if token:
        store.revoke_auth_session(hash_token(token), utc_now())
    clear_cookie(response, AUTH_COOKIE_NAME, secure=COOKIE_SECURE)
    return {"status": "logged_out"}


@app.get("/account/rate-limit")
async def account_rate_limit(identity: Identity = Depends(current_identity)) -> dict[str, Any]:
    identity = coerce_identity(identity)
    return rate_limit_status(identity)


@app.get("/account")
async def account_details(identity: Identity = Depends(current_identity)) -> dict[str, Any]:
    identity = coerce_identity(identity)
    user = require_authenticated_user(identity)
    return {"account": serialize_account(user)}


@app.get("/sessions/recent")
async def recent_sessions(
    include_all: bool = Query(default=False),
    identity: Identity = Depends(current_identity),
) -> dict[str, list[dict[str, Any]]]:
    identity = coerce_identity(identity)
    prune_sessions_for_identity(identity)
    limit = MAX_STORED_SESSIONS if include_all else RECENT_SESSIONS_DEFAULT_LIMIT
    return {
        "sessions": [
            serialize_session_summary(row)
            for row in store.recent_sessions(
                limit=limit,
                owner_type=identity.owner_type,
                owner_id=identity.owner_id,
            )
        ]
    }


@app.post("/sessions/chat")
async def create_chat_session(identity: Identity = Depends(current_identity)) -> dict[str, Any]:
    identity = coerce_identity(identity)
    session_id = uuid.uuid4().hex
    now = utc_now()
    session = store.create_session(
        session_id=session_id,
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
        filename=CHAT_SESSION_FILENAME,
        content_type=CHAT_CONTENT_TYPE,
        original_path="",
        created_at=now,
        status="chat_ready",
    )
    prune_sessions_for_identity(identity)
    return serialize_session_detail(session)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, identity: Identity = Depends(current_identity)) -> dict[str, Any]:
    identity = coerce_identity(identity)
    session = get_owned_session_or_404(session_id, identity)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return serialize_session_detail(session)


@app.get("/sessions/{session_id}/original")
async def get_original(
    session_id: str,
    identity: Identity = Depends(current_identity),
) -> FileResponse:
    identity = coerce_identity(identity)
    session = get_owned_session_or_404(session_id, identity)
    if not has_session_document(session):
        raise HTTPException(status_code=404, detail="No original document is attached.")
    original_path = Path(session["original_path"])
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found.")
    return FileResponse(
        original_path,
        media_type=session["content_type"],
        filename=session["filename"],
        content_disposition_type="inline",
    )


@app.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    request_body: RenameSessionRequest,
    identity: Identity = Depends(current_identity),
) -> dict[str, Any]:
    identity = coerce_identity(identity)
    get_owned_session_or_404(session_id, identity)
    filename = sanitize_filename(request_body.filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Session name cannot be empty.")
    store.rename_owned_session(session_id, identity.owner_type, identity.owner_id, filename, utc_now())
    updated = store.get_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Session vanished after rename.")
    return serialize_session_detail(updated)


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    identity: Identity = Depends(current_identity),
) -> dict[str, str]:
    identity = coerce_identity(identity)
    session = store.delete_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    delete_session_artifacts(session)
    return {"status": "deleted"}


@app.post("/sessions/bulk-delete")
async def bulk_delete_sessions(
    request_body: BulkDeleteSessionsRequest,
    identity: Identity = Depends(current_identity),
) -> dict[str, Any]:
    identity = coerce_identity(identity)
    seen: set[str] = set()
    deleted_count = 0
    missing_ids: list[str] = []

    for session_id in request_body.session_ids:
        session_id = session_id.strip()
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        session = store.delete_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
        if session is None:
            missing_ids.append(session_id)
            continue
        delete_session_artifacts(session)
        deleted_count += 1

    return {
        "status": "deleted",
        "deleted_count": deleted_count,
        "missing_ids": missing_ids,
    }


@app.post("/sessions/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = None,
    identity: Identity = Depends(current_identity),
) -> dict[str, Any]:
    identity = coerce_identity(identity)
    filename = sanitize_filename(file.filename or "upload")
    content_type = normalize_content_type(file.content_type or "", filename)
    validate_upload(filename, content_type)
    session_id = (session_id or "").strip() or None
    existing_session: dict[str, Any] | None = None
    if session_id:
        existing_session = store.get_session(
            session_id,
            owner_type=identity.owner_type,
            owner_id=identity.owner_id,
        )
        if existing_session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        if has_session_document(existing_session):
            raise HTTPException(
                status_code=409,
                detail="This session already has a document. Start again to attach another file.",
            )
        if existing_session["status"] in {"uploading", "ocr_running", "answering"}:
            raise HTTPException(status_code=409, detail="Session is busy.")

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Upload is empty.")

    limit_status = rate_limit_status(identity)
    if not limit_status["unlimited"] and limit_status["remaining"] <= 0:
        return rate_limit_exceeded_response(limit_status)
    record_ocr_upload(identity)

    session_id = session_id or uuid.uuid4().hex
    now = utc_now()
    suffix = Path(filename).suffix.lower()
    original_path = artifact_owner_dir(UPLOAD_DIR, identity) / f"{session_id}{suffix}"
    original_path.write_bytes(body)

    if existing_session is None:
        store.create_session(
            session_id=session_id,
            owner_type=identity.owner_type,
            owner_id=identity.owner_id,
            filename=filename,
            content_type=content_type,
            original_path=original_path,
            created_at=now,
        )
        prune_sessions_for_identity(identity)
        store.update_owned_session(
            session_id,
            identity.owner_type,
            identity.owner_id,
            utc_now(),
            status="ocr_running",
        )
    else:
        store.update_owned_session(
            session_id,
            identity.owner_type,
            identity.owner_id,
            now,
            filename=filename,
            content_type=content_type,
            original_path=str(original_path),
            status="ocr_running",
            error=None,
            ocr_markdown_path=None,
            page_count=None,
            ocr_elapsed_ms=None,
        )

    started = time.perf_counter()
    try:
        markdown = await asyncio.to_thread(
            post_ocr_request,
            filename,
            content_type,
            body,
        )
    except ServiceError as exc:
        store.update_owned_session(
            session_id,
            identity.owner_type,
            identity.owner_id,
            utc_now(),
            status="ocr_failed",
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    ocr_path = artifact_owner_dir(OCR_DIR, identity) / f"{session_id}.md"
    ocr_path.write_text(markdown, encoding="utf-8")
    store.update_owned_session(
        session_id,
        identity.owner_type,
        identity.owner_id,
        utc_now(),
        status="ocr_complete",
        error=None,
        ocr_markdown_path=str(ocr_path),
        page_count=count_pages(markdown, content_type),
        ocr_elapsed_ms=elapsed_ms,
    )

    session = store.get_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
    if session is None:
        raise HTTPException(status_code=500, detail="Session vanished after OCR.")
    data = serialize_session_detail(session)
    data["rate_limit"] = rate_limit_status(identity)
    return data


@app.post("/sessions/{session_id}/ask")
async def ask_session(
    session_id: str,
    ask: AskRequest,
    identity: Identity = Depends(current_identity),
) -> dict[str, Any]:
    identity = coerce_identity(identity)
    session = get_owned_session_or_404(session_id, identity)
    if session["status"] not in ASKABLE_STATUSES:
        raise HTTPException(status_code=409, detail="Session is not ready for chat.")

    markdown = read_session_markdown(session)
    user_prompt = ask.prompt.strip()
    existing_messages = session.get("messages", [])
    validate_empty_prompt_submission(
        user_prompt=user_prompt,
        has_ocr=bool(markdown),
        existing_messages=existing_messages,
    )
    prompt = build_prompt(user_prompt, ask.mode, has_ocr=bool(markdown))
    now = utc_now()
    if user_prompt:
        store.add_message(session_id=session_id, role="user", content=user_prompt, created_at=now)
    store.update_owned_session(
        session_id,
        identity.owner_type,
        identity.owner_id,
        now,
        status="answering",
    )

    started = time.perf_counter()
    try:
        answer = await asyncio.to_thread(
            post_answer_request,
            markdown,
            prompt,
            message_history_for_llm(existing_messages),
            ask.thinking_mode,
        )
    except ServiceError as exc:
        store.update_owned_session(
            session_id,
            identity.owner_type,
            identity.owner_id,
            utc_now(),
            status="llm_failed",
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    answer_text = str(answer.get("answer") or "").strip()
    answer_text = append_max_tokens_notice_if_needed(
        answer_text,
        stopped_due_to_max_tokens=bool(answer.get("stopped_due_to_max_tokens")),
        max_tokens_limit=as_int_or_none(answer.get("max_tokens_limit")),
    )
    store.add_message(
        session_id=session_id,
        role="assistant",
        content=answer_text,
        elapsed_ms=answer.get("elapsed_ms", elapsed_ms),
        prompt_tokens=answer.get("prompt_tokens"),
        completion_tokens=answer.get("completion_tokens"),
        total_tokens=answer.get("total_tokens"),
        created_at=utc_now(),
    )
    store.update_owned_session(
        session_id,
        identity.owner_type,
        identity.owner_id,
        utc_now(),
        status="answered",
        error=None,
        answer_elapsed_ms=elapsed_ms,
    )

    updated = store.get_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Session vanished after answer.")
    return serialize_session_detail(updated)


@app.post("/sessions/{session_id}/ask/stream")
async def ask_session_stream(
    session_id: str,
    ask: AskRequest,
    identity: Identity = Depends(current_identity),
) -> StreamingResponse:
    identity = coerce_identity(identity)
    session = get_owned_session_or_404(session_id, identity)
    if session["status"] not in ASKABLE_STATUSES:
        raise HTTPException(status_code=409, detail="Session is not ready for chat.")

    markdown = read_session_markdown(session)
    user_prompt = ask.prompt.strip()
    existing_messages = session.get("messages", [])
    validate_empty_prompt_submission(
        user_prompt=user_prompt,
        has_ocr=bool(markdown),
        existing_messages=existing_messages,
    )
    prompt = build_prompt(user_prompt, ask.mode, has_ocr=bool(markdown))
    now = utc_now()
    history = message_history_for_llm(existing_messages)
    if user_prompt:
        store.add_message(session_id=session_id, role="user", content=user_prompt, created_at=now)
    store.update_owned_session(
        session_id,
        identity.owner_type,
        identity.owner_id,
        now,
        status="answering",
    )

    def stream_events() -> Generator[str, None, None]:
        started = time.perf_counter()
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_meta: dict[str, Any] = {}
        try:
            for event in post_answer_request_stream(markdown, prompt, history, ask.thinking_mode):
                event_name = str(event.get("event") or "")
                data = event.get("data")
                if not isinstance(data, dict):
                    data = {}
                if event_name == "token":
                    delta = str(data.get("delta") or "")
                    if not delta:
                        continue
                    kind = str(data.get("kind") or "answer")
                    if kind == "reasoning":
                        reasoning_parts.append(delta)
                    else:
                        answer_parts.append(delta)
                    yield sse_event("token", {"delta": delta, "kind": kind})
                    continue
                if event_name == "done":
                    final_meta = data
                    break
                if event_name == "error":
                    detail = str(data.get("detail") or "LLM service failed.")
                    raise ServiceError(detail)
        except ServiceError as exc:
            store.update_owned_session(
                session_id,
                identity.owner_type,
                identity.owner_id,
                utc_now(),
                status="llm_failed",
                error=str(exc),
            )
            yield sse_event("error", {"detail": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            detail = f"LLM service failed: {exc}"
            store.update_owned_session(
                session_id,
                identity.owner_type,
                identity.owner_id,
                utc_now(),
                status="llm_failed",
                error=detail,
            )
            yield sse_event("error", {"detail": detail})
            return

        answer_text = str(final_meta.get("answer") or "").strip()
        if not answer_text:
            answer_text = "".join(answer_parts).strip()
        if not answer_text:
            detail = "LLM service returned an empty answer."
            store.update_owned_session(
                session_id,
                identity.owner_type,
                identity.owner_id,
                utc_now(),
                status="llm_failed",
                error=detail,
            )
            yield sse_event("error", {"detail": detail})
            return
        answer_text = append_max_tokens_notice_if_needed(
            answer_text,
            stopped_due_to_max_tokens=bool(final_meta.get("stopped_due_to_max_tokens")),
            max_tokens_limit=as_int_or_none(final_meta.get("max_tokens_limit")),
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        answer_elapsed_ms = as_int(final_meta.get("elapsed_ms"), fallback=elapsed_ms)
        prompt_tokens = as_int_or_none(final_meta.get("prompt_tokens"))
        completion_tokens = as_int_or_none(final_meta.get("completion_tokens"))
        total_tokens = as_int_or_none(final_meta.get("total_tokens"))

        store.add_message(
            session_id=session_id,
            role="assistant",
            content=answer_text,
            elapsed_ms=answer_elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            created_at=utc_now(),
        )
        store.update_owned_session(
            session_id,
            identity.owner_type,
            identity.owner_id,
            utc_now(),
            status="answered",
            error=None,
            answer_elapsed_ms=elapsed_ms,
        )
        updated = store.get_session(session_id, owner_type=identity.owner_type, owner_id=identity.owner_id)
        if updated is None:
            detail = "Session vanished after answer."
            yield sse_event("error", {"detail": detail})
            return

        payload = {
            "answer": answer_text,
            "reasoning_text": str(final_meta.get("reasoning_text") or "").strip()
            or "".join(reasoning_parts).strip()
            or None,
            "elapsed_ms": answer_elapsed_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stopped_due_to_max_tokens": bool(final_meta.get("stopped_due_to_max_tokens")),
            "max_tokens_limit": as_int_or_none(final_meta.get("max_tokens_limit")),
            "session": serialize_session_detail(updated),
        }
        yield sse_event("done", payload)

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class ServiceError(RuntimeError):
    pass


def post_ocr_request(filename: str, content_type: str, body: bytes) -> str:
    boundary = f"----ocr-web-app-{uuid.uuid4().hex}"
    payload = build_multipart_file_body(
        field_name="image",
        filename=filename,
        content_type=content_type,
        body=body,
        boundary=boundary,
    )
    req = request.Request(
        f"{OCR_SERVICE_URL.rstrip('/')}/v1/ocr",
        data=payload,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ServiceError(f"OCR service failed: {detail}") from exc
    except error.URLError as exc:
        raise ServiceError(f"OCR service is unavailable: {exc.reason}") from exc


def post_answer_request(
    markdown: str,
    prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
    thinking_mode: Literal["fast", "thinking"] = "fast",
) -> dict[str, Any]:
    request_body: dict[str, Any] = {
        "ocr_markdown": markdown,
        "user_request": prompt,
        "thinking_mode": thinking_mode,
    }
    if conversation_history:
        request_body["conversation_history"] = conversation_history
    payload = json.dumps(request_body).encode("utf-8")
    req = request.Request(
        f"{LLM_SERVICE_URL.rstrip('/')}/v1/answer",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ServiceError(f"LLM service failed: {detail}") from exc
    except error.URLError as exc:
        raise ServiceError(f"LLM service is unavailable: {exc.reason}") from exc


def post_answer_request_stream(
    markdown: str,
    prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
    thinking_mode: Literal["fast", "thinking"] = "fast",
) -> Generator[dict[str, Any], None, None]:
    request_body: dict[str, Any] = {
        "ocr_markdown": markdown,
        "user_request": prompt,
        "thinking_mode": thinking_mode,
    }
    if conversation_history:
        request_body["conversation_history"] = conversation_history
    payload = json.dumps(request_body).encode("utf-8")
    req = request.Request(
        f"{LLM_SERVICE_URL.rstrip('/')}/v1/answer/stream",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            for frame in iter_sse_frames(response):
                event_name = str(frame.get("event") or "").strip() or "message"
                data_blob = frame.get("data")
                if not isinstance(data_blob, str) or not data_blob:
                    continue
                try:
                    decoded = json.loads(data_blob)
                except json.JSONDecodeError:
                    continue
                if not isinstance(decoded, dict):
                    continue
                yield {"event": event_name, "data": decoded}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ServiceError(f"LLM service failed: {detail}") from exc
    except error.URLError as exc:
        raise ServiceError(f"LLM service is unavailable: {exc.reason}") from exc


def iter_sse_frames(response: Any) -> Generator[dict[str, str], None, None]:
    event_name = "message"
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield {"event": event_name, "data": "\n".join(data_lines)}


def sse_event(event_name: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n"


def as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any, *, fallback: int) -> int:
    parsed = as_int_or_none(value)
    return parsed if parsed is not None else fallback


def append_max_tokens_notice_if_needed(
    answer_text: str,
    *,
    stopped_due_to_max_tokens: bool,
    max_tokens_limit: int | None,
) -> str:
    text = str(answer_text or "").strip()
    if not stopped_due_to_max_tokens:
        return text
    if max_tokens_limit is not None and max_tokens_limit > 0:
        notice = f"Sorry, answer exceeds max tokens ({max_tokens_limit})."
    else:
        notice = "Sorry, answer exceeds max tokens."
    if notice in text:
        return text
    if not text:
        return notice
    return f"{text}\n\n{notice}"


def message_history_for_llm(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})
    return history


def build_multipart_file_body(
    *,
    field_name: str,
    filename: str,
    content_type: str,
    body: bytes,
    boundary: str,
) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + body + footer


def serialize_session_detail(session: dict[str, Any]) -> dict[str, Any]:
    data = serialize_session_summary(session)
    data["ocr_markdown"] = ""
    if session.get("ocr_markdown_path"):
        path = Path(session["ocr_markdown_path"])
        if path.exists():
            data["ocr_markdown"] = path.read_text(encoding="utf-8")
    data["messages"] = session.get("messages", [])
    return data


def serialize_session_summary(session: dict[str, Any]) -> dict[str, Any]:
    has_document = has_session_document(session)
    return {
        "id": session["id"],
        "filename": session["filename"],
        "content_type": session["content_type"],
        "file_type": file_type_label(session["filename"], session["content_type"]),
        "has_document": has_document,
        "status": session["status"],
        "error": session.get("error"),
        "page_count": session.get("page_count"),
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "ocr_elapsed_ms": session.get("ocr_elapsed_ms"),
        "answer_elapsed_ms": session.get("answer_elapsed_ms"),
        "thumbnail_url": f"/sessions/{session['id']}/original"
        if has_document and str(session["content_type"]).startswith("image/")
        else None,
    }


def read_session_markdown(session: dict[str, Any]) -> str:
    path_value = session.get("ocr_markdown_path")
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="OCR Markdown file not found.")
    markdown = path.read_text(encoding="utf-8").strip()
    if not markdown:
        return ""
    return markdown


def validate_empty_prompt_submission(
    *,
    user_prompt: str,
    has_ocr: bool,
    existing_messages: list[dict[str, Any]],
) -> None:
    if user_prompt:
        return
    if not has_ocr:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty without OCR context.")
    if existing_messages:
        raise HTTPException(status_code=400, detail="Empty prompt is only allowed at the start of a session.")


def build_prompt(prompt: str, mode: str | None, *, has_ocr: bool = True) -> str:
    cleaned = prompt.strip()
    if not cleaned:
        if has_ocr:
            return "Answer the question(s) contained in the OCR text."
        return "No question was provided. Ask the user to provide a question."
    if mode == "answer":
        if cleaned.lower() in {"answer this question", "answer the question(s)"}:
            if not has_ocr:
                return "Answer the question(s). If the question is missing, ask for it briefly."
            return "Answer the question(s) contained in the OCR text."
        if not has_ocr:
            return f"Answer the question(s): {cleaned}"
        return f"Answer the question(s) from the OCR text: {cleaned}"
    if mode and mode.startswith("translate:"):
        language = mode.split(":", 1)[1].strip() or "Vietnamese"
        if cleaned.lower() == f"translate to {language.lower()}":
            if not has_ocr:
                return f"Translate to {language}. If the text is missing, ask for it briefly."
            return f"Translate the OCR text to {language}."
        if not has_ocr:
            return f"Translate to {language}: {cleaned}"
        return f"Translate to {language} using the OCR text: {cleaned}"
    return cleaned


def validate_upload(filename: str, content_type: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS or content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG, JPEG, and PDF uploads are supported.",
        )


def normalize_content_type(content_type: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    guessed = mimetypes.types_map.get(suffix, "")
    if content_type in ALLOWED_CONTENT_TYPES:
        return content_type
    if guessed in ALLOWED_CONTENT_TYPES:
        return guessed
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".pdf":
        return "application/pdf"
    return content_type or "application/octet-stream"


def sanitize_filename(filename: str) -> str:
    cleaned = Path(filename).name.strip().replace("\x00", "")
    if not cleaned:
        return "upload"
    return re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned)[:160]


def count_pages(markdown: str, content_type: str) -> int:
    if content_type != "application/pdf":
        return 1
    page_markers = re.findall(r"(?m)^##\s+Page\s+\d+", markdown)
    return max(len(page_markers), 1)


def file_type_label(filename: str, content_type: str) -> str:
    if content_type == CHAT_CONTENT_TYPE:
        return "CHAT"
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix:
        return suffix.upper()
    if content_type == "application/pdf":
        return "PDF"
    if content_type.startswith("image/"):
        return content_type.split("/", 1)[1].upper()
    return "FILE"


def normalize_signup_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        raise HTTPException(status_code=400, detail="A valid email address is required.")
    return normalized


def normalize_username(username: str) -> str:
    normalized = username.strip().lower().lstrip("@")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,31}", normalized):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 characters using letters, numbers, underscores, or hyphens.",
        )
    if normalized in RESERVED_USERNAMES:
        raise HTTPException(status_code=400, detail="This username is reserved.")
    return normalized


def normalize_avatar_key(avatar_key: str | None) -> str:
    cleaned = (avatar_key or "").strip().lower() or DEFAULT_AVATAR_KEY
    allowed = set(AVATAR_KEYS)
    allowed.add(DEFAULT_AVATAR_KEY)
    if cleaned not in allowed:
        raise HTTPException(status_code=400, detail="Selected avatar is not available.")
    return cleaned


def create_bot_challenge_response() -> dict[str, Any]:
    left = secrets.randbelow(8) + 2
    right = secrets.randbelow(8) + 2
    now = utc_now()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=BOT_CHALLENGE_MINUTES)).isoformat()
    challenge_id = uuid.uuid4().hex
    store.prune_auth_challenges(now)
    store.create_bot_challenge(
        challenge_id=challenge_id,
        answer_hash=hash_token(str(left + right)),
        expires_at=expires_at,
        created_at=now,
    )
    return {
        "challenge_id": challenge_id,
        "question": f"{left} + {right}",
        "expires_at": expires_at,
    }


def verify_bot_challenge(proof: BotChallengeProof) -> None:
    if (proof.website or "").strip():
        raise HTTPException(status_code=400, detail="Bot challenge failed.")
    answer = proof.answer.strip()
    if not store.consume_bot_challenge(
        challenge_id=proof.challenge_id,
        answer_hash=hash_token(answer),
        consumed_at=utc_now(),
    ):
        raise HTTPException(status_code=400, detail="Bot challenge failed.")


def verify_signup_honeypot(website: str | None) -> None:
    if (website or "").strip():
        raise HTTPException(status_code=400, detail="Signup could not be completed.")


def assert_auth_attempt_allowed(email: str) -> None:
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    count = store.count_rate_limit_events(
        owner_type="auth",
        owner_id=email,
        action=AUTH_ATTEMPT_ACTION,
        since=since,
    )
    if count >= AUTH_ATTEMPT_LIMIT:
        raise HTTPException(status_code=429, detail="Too many authentication attempts. Try again later.")


def record_auth_attempt(email: str) -> None:
    store.add_rate_limit_event(
        owner_type="auth",
        owner_id=email,
        action=AUTH_ATTEMPT_ACTION,
        created_at=utc_now(),
    )


def send_verification_email(email: str, code: str) -> None:
    subject = "Verify your OCR AI Assistant account"
    body = f"Your verification code is {code}. It expires in {PENDING_SIGNUP_HOURS} hours."
    if not SMTP_HOST:
        if AUTH_DEBUG_CODES:
            return
        raise HTTPException(
            status_code=503,
            detail="Email verification is not available because SMTP is not configured.",
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = SMTP_FROM
    message["To"] = email
    message.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as client:
            if SMTP_STARTTLS:
                client.starttls()
            if SMTP_USERNAME:
                client.login(SMTP_USERNAME, SMTP_PASSWORD)
            client.send_message(message)
    except OSError as exc:
        raise HTTPException(status_code=502, detail="Verification email could not be sent.") from exc


def require_pending_signup(pending_id: str) -> dict[str, Any]:
    pending = store.get_pending_signup(pending_id, utc_now())
    if pending is None:
        raise HTTPException(status_code=404, detail="Signup session expired.")
    return pending


def can_start_upgrade_for_user(user: dict[str, Any], password: str) -> bool:
    if user.get("disabled") and not user.get("password_hash"):
        return True
    if not account_requires_upgrade(user):
        return False
    # Email verification is the source of truth for account ownership in the
    # simplified auth flow, so incomplete accounts can restart signup.
    return True


def account_requires_upgrade(user: dict[str, Any]) -> bool:
    return not user.get("email_verified_at")


def hmac_safe_hash_match(value: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_token(value.strip()), expected_hash)


def finalize_pending_signup(pending: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    email = str(pending["email"])
    username = str(pending["username"])
    activate_user_id = pending.get("activate_user_id")
    existing_email_user = store.get_user_by_email(email)
    if existing_email_user is not None and existing_email_user["id"] != activate_user_id:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    tier = "owner" if OWNER_EMAIL and email == OWNER_EMAIL else str(pending["tier"])
    if activate_user_id:
        return store.update_user_auth_state(
            user_id=str(activate_user_id),
            username=username,
            password_hash=str(pending["password_hash"]),
            avatar_key=str(pending["avatar_key"]),
            tier=tier,
            email_verified_at=now,
            updated_at=now,
        )
    return store.create_user(
        user_id=uuid.uuid4().hex,
        email=email,
        username=username,
        password_hash=str(pending["password_hash"]),
        avatar_key=str(pending["avatar_key"]),
        tier=tier,
        disabled=False,
        email_verified_at=now,
        created_at=now,
    )


def tier_color(tier: str) -> str:
    return {
        "guest": "gray",
        "free": "light-blue",
        "pro": "dark-green",
        "owner": "red",
    }.get(tier, "gray")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def static_asset_version() -> str:
    paths = (
        STATIC_DIR / "index.html",
        STATIC_DIR / "styles.css",
        STATIC_DIR / "app.js",
    )
    newest_mtime = max(int(path.stat().st_mtime) for path in paths)
    return str(newest_mtime)


def delete_session_artifacts(session: dict[str, Any]) -> None:
    for path_value in (session.get("original_path"), session.get("ocr_markdown_path")):
        if path_value:
            Path(path_value).unlink(missing_ok=True)


def prune_sessions_for_identity(identity: Identity) -> None:
    pruned = store.prune_owned_sessions(
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
        keep_count=MAX_STORED_SESSIONS,
    )
    for session in pruned:
        delete_session_artifacts(session)


def has_session_document(session: dict[str, Any]) -> bool:
    original_path = str(session.get("original_path") or "").strip()
    return bool(original_path) and session.get("content_type") != CHAT_CONTENT_TYPE


def identity_from_user(user: dict[str, Any]) -> Identity:
    tier = str(user.get("tier") or "free")
    if OWNER_EMAIL and user.get("email") == OWNER_EMAIL:
        tier = "owner"
    return Identity(
        owner_type="user",
        owner_id=str(user["id"]),
        tier=tier,
        email=str(user["email"]),
        username=str(user.get("username") or "user"),
        avatar_key=str(user.get("avatar_key") or DEFAULT_AVATAR_KEY),
        is_authenticated=True,
    )


def issue_auth_cookie(response: Response, user_id: str) -> None:
    token = new_token()
    now = utc_now()
    store.create_auth_session(
        user_id=user_id,
        token_hash=hash_token(token),
        expires_at=utc_expiry(AUTH_SESSION_DAYS),
        created_at=now,
    )
    set_signed_cookie(
        response,
        name=AUTH_COOKIE_NAME,
        value=token,
        secret_key=SECRET_KEY,
        max_age=auth_cookie_max_age(),
        secure=COOKIE_SECURE,
    )


def serialize_user(user: dict[str, Any]) -> dict[str, Any]:
    tier = str(user.get("tier") or "free")
    if OWNER_EMAIL and user.get("email") == OWNER_EMAIL:
        tier = "owner"
    return {
        "id": user["id"],
        "username": user.get("username") or "user",
        "avatar_key": user.get("avatar_key") or DEFAULT_AVATAR_KEY,
        "tier": tier,
        "tier_color": tier_color(tier),
        "email_verified": bool(user.get("email_verified_at")),
        "two_factor_enabled": bool(user.get("two_factor_enabled")),
    }


def serialize_account(user: dict[str, Any]) -> dict[str, Any]:
    identity = identity_from_user(user)
    rate_limit = rate_limit_status(identity)
    if rate_limit["unlimited"]:
        usage = {
            "remaining": None,
            "limit": None,
            "unlimited": True,
            "summary": "OCR uploads: unlimited.",
        }
    else:
        usage = {
            "remaining": rate_limit["remaining"],
            "limit": rate_limit["limit"],
            "unlimited": False,
            "summary": f"{rate_limit['remaining']}/{rate_limit['limit']} remaining this hour",
        }
    return {
        "id": user["id"],
        "username": identity.username,
        "email": identity.email,
        "tier": identity.tier,
        "tier_color": tier_color(identity.tier),
        "usage": usage,
        "rate_limit": rate_limit,
    }


def serialize_identity(identity: Identity) -> dict[str, Any]:
    username = identity.username or ("Guest" if identity.owner_type == "guest" else "user")
    avatar_key = identity.avatar_key or ("guest" if identity.owner_type == "guest" else DEFAULT_AVATAR_KEY)
    return {
        "identity": {
            "type": identity.owner_type,
            "id": identity.owner_id,
            "username": username,
            "avatar_key": avatar_key,
            "tier": identity.tier,
            "tier_color": tier_color(identity.tier),
            "authenticated": identity.is_authenticated,
        },
        "rate_limit": rate_limit_status(identity),
    }


def require_authenticated_user(identity: Identity) -> dict[str, Any]:
    if not identity.is_authenticated or not identity.owner_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = store.get_user_by_id(identity.owner_id)
    if user is None or user.get("disabled") or account_requires_upgrade(user):
        raise HTTPException(status_code=401, detail="Authentication required.")
    if OWNER_EMAIL and user.get("email") == OWNER_EMAIL and user.get("tier") != "owner":
        user = store.update_user_tier(user_id=user["id"], tier="owner", updated_at=utc_now())
    return user


def coerce_identity(value: Any) -> Identity:
    if isinstance(value, Identity):
        return value
    return Identity(
        owner_type="user",
        owner_id="legacy-owner",
        tier="free",
        username="user",
        avatar_key=DEFAULT_AVATAR_KEY,
        is_authenticated=True,
    )


def get_owned_session_or_404(session_id: str, identity: Identity) -> dict[str, Any]:
    session = store.get_session(
        session_id,
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def artifact_owner_dir(root: Path, identity: Identity) -> Path:
    path = root / identity.owner_type / sanitize_path_segment(identity.owner_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_path_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:160] or "unknown"


def rate_limit_status(identity: Identity) -> dict[str, Any]:
    limit = OCR_UPLOAD_LIMITS.get(identity.tier)
    now_dt = datetime.now(timezone.utc)
    since_dt = now_dt - timedelta(hours=1)
    if limit is None:
        return {
            "tier": identity.tier,
            "tier_color": tier_color(identity.tier),
            "limit": None,
            "remaining": None,
            "reset_at": None,
            "unlimited": True,
        }

    since = since_dt.isoformat()
    count = store.count_rate_limit_events(
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
        action=OCR_UPLOAD_ACTION,
        since=since,
    )
    oldest = store.oldest_rate_limit_event_since(
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
        action=OCR_UPLOAD_ACTION,
        since=since,
    )
    reset_at = (now_dt + timedelta(hours=1)).isoformat()
    if oldest:
        reset_at = (datetime.fromisoformat(oldest) + timedelta(hours=1)).isoformat()
    return {
        "tier": identity.tier,
        "tier_color": tier_color(identity.tier),
        "limit": limit,
        "remaining": max(limit - count, 0),
        "reset_at": reset_at,
        "unlimited": False,
    }


def record_ocr_upload(identity: Identity) -> None:
    if OCR_UPLOAD_LIMITS.get(identity.tier) is None:
        return
    now_dt = datetime.now(timezone.utc)
    store.prune_rate_limit_events((now_dt - timedelta(hours=24)).isoformat())
    store.add_rate_limit_event(
        owner_type=identity.owner_type,
        owner_id=identity.owner_id,
        action=OCR_UPLOAD_ACTION,
        created_at=now_dt.isoformat(),
    )


def rate_limit_exceeded_response(limit_status: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Hourly OCR upload limit reached.",
            "limit": limit_status["limit"],
            "remaining": 0,
            "reset_at": limit_status["reset_at"],
            "tier": limit_status["tier"],
        },
    )


def configure_owner_account() -> None:
    if not OWNER_EMAIL:
        return
    now = utc_now()
    owner_id = uuid.uuid5(uuid.NAMESPACE_URL, f"ocr-owner:{OWNER_EMAIL}").hex
    owner = store.ensure_owner_placeholder(user_id=owner_id, email=OWNER_EMAIL, created_at=now)
    store.assign_legacy_sessions_to_owner(str(owner["id"]))


configure_owner_account()


def main() -> None:
    import uvicorn

    uvicorn.run("web_app.main:app", host=WEB_HOST, port=WEB_PORT, reload=False)


if __name__ == "__main__":
    main()
