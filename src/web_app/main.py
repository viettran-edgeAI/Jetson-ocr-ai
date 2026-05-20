from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "application/pdf",
}

for path in (UPLOAD_DIR, OCR_DIR):
    path.mkdir(parents=True, exist_ok=True)

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


@app.get("/sessions/recent")
async def recent_sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": [serialize_session_summary(row) for row in store.recent_sessions()]}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return serialize_session_detail(session)


@app.get("/sessions/{session_id}/original")
async def get_original(session_id: str) -> FileResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
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
async def rename_session(session_id: str, request_body: RenameSessionRequest) -> dict[str, Any]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    filename = sanitize_filename(request_body.filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Session name cannot be empty.")
    store.rename_session(session_id, filename, utc_now())
    updated = store.get_session(session_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Session vanished after rename.")
    return serialize_session_detail(updated)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    session = store.delete_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    for path_value in (session.get("original_path"), session.get("ocr_markdown_path")):
        if path_value:
            Path(path_value).unlink(missing_ok=True)
    return {"status": "deleted"}


@app.post("/sessions/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = sanitize_filename(file.filename or "upload")
    content_type = normalize_content_type(file.content_type or "", filename)
    validate_upload(filename, content_type)

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Upload is empty.")

    session_id = uuid.uuid4().hex
    now = utc_now()
    suffix = Path(filename).suffix.lower()
    original_path = UPLOAD_DIR / f"{session_id}{suffix}"
    original_path.write_bytes(body)

    store.create_session(
        session_id=session_id,
        filename=filename,
        content_type=content_type,
        original_path=original_path,
        created_at=now,
    )
    store.update_session(session_id, utc_now(), status="ocr_running")

    started = time.perf_counter()
    try:
        markdown = await asyncio.to_thread(
            post_ocr_request,
            filename,
            content_type,
            body,
        )
    except ServiceError as exc:
        store.update_session(session_id, utc_now(), status="ocr_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    ocr_path = OCR_DIR / f"{session_id}.md"
    ocr_path.write_text(markdown, encoding="utf-8")
    store.update_session(
        session_id,
        utc_now(),
        status="ocr_complete",
        error=None,
        ocr_markdown_path=str(ocr_path),
        page_count=count_pages(markdown, content_type),
        ocr_elapsed_ms=elapsed_ms,
    )

    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=500, detail="Session vanished after OCR.")
    return serialize_session_detail(session)


@app.post("/sessions/{session_id}/ask")
async def ask_session(session_id: str, ask: AskRequest) -> dict[str, Any]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session["status"] not in {"ocr_complete", "answered", "llm_failed"}:
        raise HTTPException(status_code=409, detail="OCR must complete before asking.")

    markdown = read_session_markdown(session)
    user_prompt = ask.prompt.strip()
    prompt = build_prompt(user_prompt, ask.mode)
    now = utc_now()
    store.add_message(session_id=session_id, role="user", content=user_prompt, created_at=now)
    store.update_session(session_id, now, status="answering")

    started = time.perf_counter()
    try:
        answer = await asyncio.to_thread(post_answer_request, markdown, prompt)
    except ServiceError as exc:
        store.update_session(session_id, utc_now(), status="llm_failed", error=str(exc))
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
    store.update_session(
        session_id,
        utc_now(),
        status="answered",
        error=None,
        answer_elapsed_ms=elapsed_ms,
    )

    updated = store.get_session(session_id)
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


def post_answer_request(markdown: str, prompt: str) -> dict[str, Any]:
    payload = json.dumps({"ocr_markdown": markdown, "user_request": prompt}).encode("utf-8")
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
    return {
        "id": session["id"],
        "filename": session["filename"],
        "content_type": session["content_type"],
        "file_type": file_type_label(session["filename"], session["content_type"]),
        "status": session["status"],
        "error": session.get("error"),
        "page_count": session.get("page_count"),
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "ocr_elapsed_ms": session.get("ocr_elapsed_ms"),
        "answer_elapsed_ms": session.get("answer_elapsed_ms"),
        "thumbnail_url": f"/sessions/{session['id']}/original"
        if str(session["content_type"]).startswith("image/")
        else None,
    }


def read_session_markdown(session: dict[str, Any]) -> str:
    path_value = session.get("ocr_markdown_path")
    if not path_value:
        raise HTTPException(status_code=409, detail="OCR Markdown is not available.")
    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="OCR Markdown file not found.")
    markdown = path.read_text(encoding="utf-8").strip()
    if not markdown:
        raise HTTPException(status_code=409, detail="OCR Markdown is empty.")
    return markdown


def build_prompt(prompt: str, mode: str | None) -> str:
    cleaned = prompt.strip()
    if mode == "answer":
        if cleaned.lower() == "answer this question":
            return "Answer the question contained in the OCR text."
        return f"Answer this question from the OCR text: {cleaned}"
    if mode == "solve":
        if cleaned.lower() == "solve this problem":
            return "Solve the problem contained in the OCR text. Show the reasoning steps when useful."
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


def main() -> None:
    import uvicorn

    uvicorn.run("web_app.main:app", host=WEB_HOST, port=WEB_PORT, reload=False)


if __name__ == "__main__":
    main()
