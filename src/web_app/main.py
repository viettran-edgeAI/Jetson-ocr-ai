from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "application/pdf",
}
CHAT_CONTENT_TYPE = "application/x-chat-session"
CHAT_SESSION_FILENAME = "Untitled chat"
ASKABLE_STATUSES = {"chat_ready", "ocr_complete", "answered", "llm_failed", "ocr_failed"}
OCR_UPLOAD_ACTION = "ocr_upload"
OCR_UPLOAD_LIMITS = {
    "guest": 10,
    "free": 50,
    "pro": 2000,
}

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
    prompt: str = Field(..., min_length=1, max_length=2000)
    mode: str | None = Field(default=None, max_length=64)


class RenameSessionRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=160)


class BulkDeleteSessionsRequest(BaseModel):
    session_ids: list[str] = Field(..., min_length=1, max_length=200)


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
    return Identity(owner_type="guest", owner_id=guest_id, tier="guest", is_authenticated=False)


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


@app.post("/auth/signup")
async def signup(auth: AuthRequest, response: Response) -> dict[str, Any]:
    email = normalize_email(auth.email)
    validate_password(auth.password)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")

    now = utc_now()
    tier = "owner" if OWNER_EMAIL and email == OWNER_EMAIL else "free"
    existing = store.get_user_by_email(email)
    password_hash = hash_password(auth.password)
    if existing is not None:
        if existing.get("disabled") and not existing.get("password_hash"):
            user = store.activate_placeholder_user(
                user_id=existing["id"],
                password_hash=password_hash,
                tier=tier,
                updated_at=now,
            )
        else:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
    else:
        user = store.create_user(
            user_id=uuid.uuid4().hex,
            email=email,
            password_hash=password_hash,
            tier=tier,
            disabled=False,
            created_at=now,
        )

    issue_auth_cookie(response, user["id"])
    return {"user": serialize_user(user), "rate_limit": rate_limit_status(identity_from_user(user))}


@app.post("/auth/login")
async def login(auth: AuthRequest, response: Response) -> dict[str, Any]:
    email = normalize_email(auth.email)
    user = store.get_user_by_email(email)
    if (
        user is None
        or user.get("disabled")
        or not user.get("password_hash")
        or not verify_password(auth.password, str(user["password_hash"]))
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if OWNER_EMAIL and user["email"] == OWNER_EMAIL and user["tier"] != "owner":
        user = store.activate_placeholder_user(
            user_id=user["id"],
            password_hash=str(user["password_hash"]),
            tier="owner",
            updated_at=utc_now(),
        )
    issue_auth_cookie(response, user["id"])
    return {"user": serialize_user(user), "rate_limit": rate_limit_status(identity_from_user(user))}


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


@app.get("/sessions/recent")
async def recent_sessions(identity: Identity = Depends(current_identity)) -> dict[str, list[dict[str, Any]]]:
    identity = coerce_identity(identity)
    return {
        "sessions": [
            serialize_session_summary(row)
            for row in store.recent_sessions(
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
    prompt = build_prompt(user_prompt, ask.mode, has_ocr=bool(markdown))
    now = utc_now()
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
            message_history_for_llm(session.get("messages", [])),
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
    store.add_message(
        session_id=session_id,
        role="assistant",
        content=answer["answer"],
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
) -> dict[str, Any]:
    request_body: dict[str, Any] = {"ocr_markdown": markdown, "user_request": prompt}
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


def build_prompt(prompt: str, mode: str | None, *, has_ocr: bool = True) -> str:
    cleaned = prompt.strip()
    if mode == "answer":
        if cleaned.lower() == "answer this question":
            if not has_ocr:
                return "Answer this question. If the question is missing, ask for it briefly."
            return "Answer the question contained in the OCR text."
        if not has_ocr:
            return f"Answer this question: {cleaned}"
        return f"Answer this question from the OCR text: {cleaned}"
    if mode == "solve":
        if cleaned.lower() == "solve this problem":
            if not has_ocr:
                return "Solve this problem. If the problem is missing, ask for it briefly."
            return "Solve the problem contained in the OCR text. Show the reasoning steps when useful."
        if not has_ocr:
            return f"Solve this problem: {cleaned}"
        return f"Solve this problem using the OCR text: {cleaned}"
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
        "email": user["email"],
        "tier": tier,
    }


def serialize_identity(identity: Identity) -> dict[str, Any]:
    return {
        "identity": {
            "type": identity.owner_type,
            "id": identity.owner_id,
            "email": identity.email,
            "tier": identity.tier,
            "authenticated": identity.is_authenticated,
        },
        "rate_limit": rate_limit_status(identity),
    }


def coerce_identity(value: Any) -> Identity:
    if isinstance(value, Identity):
        return value
    return Identity(owner_type="user", owner_id="legacy-owner", tier="free", is_authenticated=True)


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
