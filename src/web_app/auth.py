from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from fastapi import HTTPException, Request, Response


AUTH_COOKIE_NAME = "ocr_auth_session"
GUEST_COOKIE_NAME = "ocr_guest_id"
AUTH_SESSION_DAYS = 30
GUEST_SESSION_DAYS = 365


@dataclass(frozen=True)
class Identity:
    owner_type: str
    owner_id: str
    tier: str
    email: str | None = None
    username: str | None = None
    avatar_key: str | None = None
    is_authenticated: bool = False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_expiry(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return "pbkdf2_sha256$260000${}${}".format(
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_value, digest_value = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_value.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_value.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def sign_value(value: str, secret_key: str) -> str:
    signature = hmac.new(secret_key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    encoded = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{value}.{encoded}"


def unsign_value(signed_value: str | None, secret_key: str) -> str | None:
    if not signed_value or "." not in signed_value:
        return None
    value, signature = signed_value.rsplit(".", 1)
    expected = sign_value(value, secret_key).rsplit(".", 1)[1]
    if not hmac.compare_digest(signature, expected):
        return None
    return value


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def new_numeric_code(length: int = 6) -> str:
    upper_bound = 10**length
    return f"{secrets.randbelow(upper_bound):0{length}d}"


def new_guest_id() -> str:
    return secrets.token_urlsafe(24)


def validate_password(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if password.isalpha() or password.isdigit():
        raise HTTPException(status_code=400, detail="Password must include mixed character types.")


def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def normalize_totp_secret(secret: str) -> bytes:
    cleaned = secret.strip().replace(" ", "").upper()
    padding = "=" * ((8 - len(cleaned) % 8) % 8)
    return base64.b32decode((cleaned + padding).encode("ascii"), casefold=True)


def totp_code(secret: str, *, for_time: int | None = None, step_seconds: int = 30) -> str:
    timestamp = epoch_seconds() if for_time is None else int(for_time)
    counter = timestamp // step_seconds
    digest = hmac.new(
        normalize_totp_secret(secret),
        struct.pack(">Q", counter),
        hashlib.sha1,
    ).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return f"{code_int % 1_000_000:06d}"


def verify_totp(secret: str, code: str, *, window: int = 1, at_time: int | None = None) -> bool:
    cleaned = "".join(ch for ch in code if ch.isdigit())
    if len(cleaned) != 6:
        return False
    timestamp = epoch_seconds() if at_time is None else int(at_time)
    for offset in range(-window, window + 1):
        expected = totp_code(secret, for_time=timestamp + offset * 30)
        if hmac.compare_digest(cleaned, expected):
            return True
    return False


def otpauth_uri(*, issuer: str, account_name: str, secret: str) -> str:
    label = f"{issuer}:{account_name}"
    return (
        f"otpauth://totp/{quote(label)}"
        f"?secret={quote(secret)}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30"
    )


def validate_secret_key(secret_key: str) -> None:
    if not secret_key or secret_key == "dev-insecure-change-me":
        return
    if len(secret_key) < 32:
        raise RuntimeError("WEB_APP_SECRET_KEY must be at least 32 characters.")


def set_signed_cookie(
    response: Response,
    *,
    name: str,
    value: str,
    secret_key: str,
    max_age: int,
    secure: bool,
) -> None:
    response.set_cookie(
        key=name,
        value=sign_value(value, secret_key),
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=secure,
    )


def clear_cookie(response: Response, name: str, *, secure: bool) -> None:
    response.delete_cookie(key=name, httponly=True, samesite="lax", secure=secure)


def auth_token_from_request(request: Request, secret_key: str) -> str | None:
    return unsign_value(request.cookies.get(AUTH_COOKIE_NAME), secret_key)


def guest_id_from_request(request: Request, secret_key: str) -> str | None:
    return unsign_value(request.cookies.get(GUEST_COOKIE_NAME), secret_key)


def auth_cookie_max_age() -> int:
    return AUTH_SESSION_DAYS * 24 * 60 * 60


def guest_cookie_max_age() -> int:
    return GUEST_SESSION_DAYS * 24 * 60 * 60


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def epoch_seconds() -> int:
    return int(time.time())
