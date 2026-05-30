from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    username TEXT,
                    password_hash TEXT,
                    avatar_key TEXT NOT NULL DEFAULT 'default',
                    tier TEXT NOT NULL DEFAULT 'free',
                    disabled INTEGER NOT NULL DEFAULT 0,
                    email_verified_at TEXT,
                    two_factor_enabled INTEGER NOT NULL DEFAULT 0,
                    two_factor_secret TEXT,
                    last_login_at TEXT,
                    failed_login_count INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_signups (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    username TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    avatar_key TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    email_code_hash TEXT NOT NULL,
                    email_verified_at TEXT,
                    two_factor_secret TEXT,
                    activate_user_id TEXT,
                    bot_passed_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_challenges (
                    id TEXT PRIMARY KEY,
                    answer_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS login_challenges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS two_factor_recovery_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    code_hash TEXT NOT NULL UNIQUE,
                    used_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    owner_type TEXT NOT NULL DEFAULT 'user',
                    owner_id TEXT NOT NULL DEFAULT 'legacy-owner',
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

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    elapsed_ms INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                    ON sessions(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_token_hash
                    ON auth_sessions(token_hash);
                CREATE INDEX IF NOT EXISTS idx_rate_limit_events_owner_action_created_at
                    ON rate_limit_events(owner_type, owner_id, action, created_at);
                CREATE INDEX IF NOT EXISTS idx_pending_signups_email
                    ON pending_signups(email);
                CREATE INDEX IF NOT EXISTS idx_pending_signups_username
                    ON pending_signups(username);
                CREATE INDEX IF NOT EXISTS idx_bot_challenges_expires_at
                    ON bot_challenges(expires_at);
                CREATE INDEX IF NOT EXISTS idx_login_challenges_user_id
                    ON login_challenges(user_id);
                CREATE INDEX IF NOT EXISTS idx_recovery_codes_user_id
                    ON two_factor_recovery_codes(user_id);
                """
            )
            self._ensure_column(connection, "users", "username", "TEXT")
            self._ensure_column(connection, "users", "avatar_key", "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(connection, "users", "email_verified_at", "TEXT")
            self._ensure_column(connection, "users", "two_factor_enabled", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "users", "two_factor_secret", "TEXT")
            self._ensure_column(connection, "users", "last_login_at", "TEXT")
            self._ensure_column(connection, "users", "failed_login_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "users", "locked_until", "TEXT")
            self._backfill_user_profiles(connection)
            connection.execute("DROP INDEX IF EXISTS idx_users_username_unique")
            self._ensure_column(connection, "sessions", "owner_type", "TEXT NOT NULL DEFAULT 'user'")
            self._ensure_column(connection, "sessions", "owner_id", "TEXT NOT NULL DEFAULT 'legacy-owner'")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_owner_updated_at
                    ON sessions(owner_type, owner_id, updated_at DESC)
                """
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _backfill_user_profiles(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT id, email, username, avatar_key FROM users ORDER BY created_at ASC"
        ).fetchall()
        used = {
            str(row["username"])
            for row in rows
            if row["username"] is not None and str(row["username"]).strip()
        }
        for row in rows:
            updates: dict[str, Any] = {}
            username = str(row["username"] or "").strip()
            if not username:
                username = self._unique_backfill_username(str(row["email"]), used)
                updates["username"] = username
                used.add(username)
            avatar_key = str(row["avatar_key"] or "").strip()
            if not avatar_key:
                updates["avatar_key"] = "default"
            if updates:
                assignments = ", ".join(f"{key} = ?" for key in updates)
                connection.execute(
                    f"UPDATE users SET {assignments} WHERE id = ?",
                    [*updates.values(), row["id"]],
                )

    def _unique_backfill_username(self, email: str, used: set[str]) -> str:
        local_part = email.split("@", 1)[0].lower()
        base = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in local_part)
        base = base.strip("_-")[:24] or "user"
        candidate = base
        suffix = 2
        while candidate in used:
            tail = f"_{suffix}"
            candidate = f"{base[: 32 - len(tail)]}{tail}"
            suffix += 1
        return candidate

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def create_user(
        self,
        *,
        user_id: str,
        email: str,
        username: str,
        password_hash: str | None,
        avatar_key: str,
        tier: str,
        disabled: bool,
        created_at: str,
        email_verified_at: str | None = None,
        two_factor_enabled: bool = False,
        two_factor_secret: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    id, email, username, password_hash, avatar_key, tier, disabled,
                    email_verified_at, two_factor_enabled, two_factor_secret,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email,
                    username,
                    password_hash,
                    avatar_key,
                    tier,
                    int(disabled),
                    email_verified_at,
                    int(two_factor_enabled),
                    two_factor_secret,
                    created_at,
                    created_at,
                ),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("failed to create user")
        return user

    def activate_placeholder_user(
        self,
        *,
        user_id: str,
        password_hash: str,
        username: str,
        avatar_key: str,
        tier: str,
        updated_at: str,
        email_verified_at: str | None = None,
        two_factor_enabled: bool = False,
        two_factor_secret: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, username = ?, avatar_key = ?, tier = ?, disabled = 0,
                    email_verified_at = ?, two_factor_enabled = ?, two_factor_secret = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    password_hash,
                    username,
                    avatar_key,
                    tier,
                    email_verified_at,
                    int(two_factor_enabled),
                    two_factor_secret,
                    updated_at,
                    user_id,
                ),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("failed to activate user")
        return user

    def update_user_auth_state(
        self,
        *,
        user_id: str,
        username: str,
        password_hash: str,
        avatar_key: str,
        tier: str,
        email_verified_at: str,
        updated_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET username = ?, password_hash = ?, avatar_key = ?, tier = ?, disabled = 0,
                    email_verified_at = ?, two_factor_enabled = 0, two_factor_secret = NULL,
                    failed_login_count = 0, locked_until = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    username,
                    password_hash,
                    avatar_key,
                    tier,
                    email_verified_at,
                    updated_at,
                    user_id,
                ),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("failed to update user auth state")
        return user

    def record_successful_login(self, *, user_id: str, logged_in_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET last_login_at = ?, failed_login_count = 0, locked_until = NULL, updated_at = ?
                WHERE id = ?
                """,
                (logged_in_at, logged_in_at, user_id),
            )

    def update_user_tier(self, *, user_id: str, tier: str, updated_at: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                "UPDATE users SET tier = ?, updated_at = ? WHERE id = ?",
                (tier, updated_at, user_id),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("failed to update user tier")
        return user

    def prune_pending_signups(self, now: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM pending_signups WHERE expires_at <= ?", (now,))

    def get_pending_signup(self, pending_id: str, now: str | None = None) -> dict[str, Any] | None:
        where = "id = ?"
        values: list[Any] = [pending_id]
        if now is not None:
            where += " AND expires_at > ?"
            values.append(now)
        with self.connect() as connection:
            row = connection.execute(f"SELECT * FROM pending_signups WHERE {where}", values).fetchone()
        return dict(row) if row else None

    def get_pending_signup_by_email(self, email: str, now: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_signups WHERE email = ? AND expires_at > ?",
                (email, now),
            ).fetchone()
        return dict(row) if row else None

    def get_pending_signup_by_username(self, username: str, now: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_signups WHERE username = ? AND expires_at > ?",
                (username, now),
            ).fetchone()
        return dict(row) if row else None

    def delete_pending_signup(self, pending_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM pending_signups WHERE id = ?", (pending_id,))

    def upsert_pending_signup(
        self,
        *,
        pending_id: str,
        email: str,
        username: str,
        password_hash: str,
        avatar_key: str,
        tier: str,
        email_code_hash: str,
        activate_user_id: str | None,
        bot_passed_at: str,
        expires_at: str,
        created_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("DELETE FROM pending_signups WHERE email = ?", (email,))
            connection.execute(
                """
                INSERT INTO pending_signups (
                    id, email, username, password_hash, avatar_key, tier,
                    email_code_hash, activate_user_id, bot_passed_at,
                    expires_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pending_id,
                    email,
                    username,
                    password_hash,
                    avatar_key,
                    tier,
                    email_code_hash,
                    activate_user_id,
                    bot_passed_at,
                    expires_at,
                    created_at,
                    created_at,
                ),
            )
        pending = self.get_pending_signup(pending_id)
        if pending is None:
            raise RuntimeError("failed to create pending signup")
        return pending

    def mark_pending_email_verified(self, *, pending_id: str, verified_at: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pending_signups
                SET email_verified_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (verified_at, verified_at, pending_id),
            )
        pending = self.get_pending_signup(pending_id)
        if pending is None:
            raise RuntimeError("failed to update pending signup")
        return pending

    def set_pending_two_factor_secret(
        self,
        *,
        pending_id: str,
        two_factor_secret: str,
        updated_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pending_signups
                SET two_factor_secret = ?, updated_at = ?
                WHERE id = ?
                """,
                (two_factor_secret, updated_at, pending_id),
            )
        pending = self.get_pending_signup(pending_id)
        if pending is None:
            raise RuntimeError("failed to update pending signup")
        return pending

    def create_bot_challenge(
        self,
        *,
        challenge_id: str,
        answer_hash: str,
        expires_at: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO bot_challenges (id, answer_hash, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (challenge_id, answer_hash, expires_at, created_at),
            )

    def consume_bot_challenge(self, *, challenge_id: str, answer_hash: str, consumed_at: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id
                FROM bot_challenges
                WHERE id = ? AND answer_hash = ? AND expires_at > ? AND consumed_at IS NULL
                """,
                (challenge_id, answer_hash, consumed_at),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE bot_challenges SET consumed_at = ? WHERE id = ?",
                (consumed_at, challenge_id),
            )
        return True

    def prune_auth_challenges(self, now: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM bot_challenges WHERE expires_at <= ?", (now,))
            connection.execute("DELETE FROM login_challenges WHERE expires_at <= ?", (now,))

    def create_login_challenge(
        self,
        *,
        challenge_id: str,
        user_id: str,
        expires_at: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO login_challenges (id, user_id, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (challenge_id, user_id, expires_at, created_at),
            )

    def get_login_challenge(self, challenge_id: str, now: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM login_challenges
                WHERE id = ? AND expires_at > ? AND consumed_at IS NULL
                """,
                (challenge_id, now),
            ).fetchone()
        return dict(row) if row else None

    def consume_login_challenge(self, *, challenge_id: str, consumed_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE login_challenges SET consumed_at = ? WHERE id = ?",
                (consumed_at, challenge_id),
            )

    def replace_recovery_codes(
        self,
        *,
        user_id: str,
        code_hashes: list[str],
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM two_factor_recovery_codes WHERE user_id = ?", (user_id,))
            connection.executemany(
                """
                INSERT INTO two_factor_recovery_codes (user_id, code_hash, created_at)
                VALUES (?, ?, ?)
                """,
                [(user_id, code_hash, created_at) for code_hash in code_hashes],
            )

    def consume_recovery_code(self, *, user_id: str, code_hash: str, used_at: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id
                FROM two_factor_recovery_codes
                WHERE user_id = ? AND code_hash = ? AND used_at IS NULL
                """,
                (user_id, code_hash),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE two_factor_recovery_codes SET used_at = ? WHERE id = ?",
                (used_at, row["id"]),
            )
        return True

    def create_auth_session(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (user_id, token_hash, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, token_hash, expires_at, created_at),
            )

    def get_user_by_auth_token_hash(self, token_hash: str, now: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.*
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = ?
                  AND auth_sessions.revoked_at IS NULL
                  AND auth_sessions.expires_at > ?
                  AND users.disabled = 0
                """,
                (token_hash, now),
            ).fetchone()
        return dict(row) if row else None

    def revoke_auth_session(self, token_hash: str, revoked_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE token_hash = ?",
                (revoked_at, token_hash),
            )

    def ensure_owner_placeholder(self, *, user_id: str, email: str, created_at: str) -> dict[str, Any]:
        existing = self.get_user_by_email(email)
        if existing is not None:
            if existing["tier"] != "owner":
                with self.connect() as connection:
                    connection.execute(
                        "UPDATE users SET tier = 'owner', updated_at = ? WHERE id = ?",
                        (created_at, existing["id"]),
                    )
                existing = self.get_user_by_id(existing["id"])
            if existing is None:
                raise RuntimeError("owner user disappeared")
            return existing
        with self.connect() as connection:
            used_usernames = {
                str(row["username"])
                for row in connection.execute("SELECT username FROM users WHERE username IS NOT NULL").fetchall()
            }
        return self.create_user(
            user_id=user_id,
            email=email,
            username=self._unique_backfill_username(email, used_usernames),
            password_hash=None,
            avatar_key="default",
            tier="owner",
            disabled=True,
            created_at=created_at,
        )

    def assign_legacy_sessions_to_owner(self, owner_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET owner_type = 'user', owner_id = ?
                WHERE owner_id = 'legacy-owner'
                """,
                (owner_id,),
            )

    def create_session(
        self,
        *,
        session_id: str,
        owner_type: str = "user",
        owner_id: str = "legacy-owner",
        filename: str,
        content_type: str,
        original_path: Path | str,
        created_at: str,
        status: str = "uploading",
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, owner_type, owner_id, filename, content_type, original_path,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    owner_type,
                    owner_id,
                    filename,
                    content_type,
                    str(original_path),
                    status,
                    created_at,
                    created_at,
                ),
            )
        session = self.get_session(session_id, owner_type=owner_type, owner_id=owner_id)
        if session is None:
            raise RuntimeError("failed to create session")
        return session

    def update_session(self, session_id: str, updated_at: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = [f"{key} = ?" for key in fields]
        values = list(fields.values())
        assignments.append("updated_at = ?")
        values.append(updated_at)
        values.append(session_id)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    def update_owned_session(
        self,
        session_id: str,
        owner_type: str,
        owner_id: str,
        updated_at: str,
        **fields: Any,
    ) -> None:
        if not fields:
            return
        assignments = [f"{key} = ?" for key in fields]
        values = list(fields.values())
        assignments.append("updated_at = ?")
        values.extend([updated_at, session_id, owner_type, owner_id])
        with self.connect() as connection:
            connection.execute(
                f"""
                UPDATE sessions
                SET {', '.join(assignments)}
                WHERE id = ? AND owner_type = ? AND owner_id = ?
                """,
                values,
            )

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        created_at: str,
        elapsed_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, elapsed_ms, prompt_tokens,
                    completion_tokens, total_tokens, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    elapsed_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    created_at,
                ),
            )

    def rename_session(self, session_id: str, filename: str, updated_at: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE sessions SET filename = ?, updated_at = ? WHERE id = ?",
                (filename, updated_at, session_id),
            )

    def rename_owned_session(
        self,
        session_id: str,
        owner_type: str,
        owner_id: str,
        filename: str,
        updated_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET filename = ?, updated_at = ?
                WHERE id = ? AND owner_type = ? AND owner_id = ?
                """,
                (filename, updated_at, session_id, owner_type, owner_id),
            )

    def delete_session(
        self,
        session_id: str,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        session = self.get_session(session_id, owner_type=owner_type, owner_id=owner_id)
        if session is None:
            return None
        with self.connect() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            if owner_type and owner_id:
                connection.execute(
                    "DELETE FROM sessions WHERE id = ? AND owner_type = ? AND owner_id = ?",
                    (session_id, owner_type, owner_id),
                )
            else:
                connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return session

    def get_session(
        self,
        session_id: str,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        where = "id = ?"
        values: list[Any] = [session_id]
        if owner_type and owner_id:
            where += " AND owner_type = ? AND owner_id = ?"
            values.extend([owner_type, owner_id])
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT * FROM sessions WHERE {where}",
                values,
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                """
                SELECT role, content, elapsed_ms, prompt_tokens, completion_tokens,
                       total_tokens, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        session = dict(row)
        session["messages"] = [dict(message) for message in messages]
        return session

    def recent_sessions(
        self,
        limit: int = 8,
        owner_type: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where = ""
        values: list[Any] = []
        if owner_type and owner_id:
            where = "WHERE owner_type = ? AND owner_id = ?"
            values.extend([owner_type, owner_id])
        values.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM sessions
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_owned_sessions(
        self,
        *,
        owner_type: str,
        owner_id: str,
        keep_count: int,
    ) -> list[dict[str, Any]]:
        if keep_count < 0:
            keep_count = 0
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM sessions
                WHERE owner_type = ? AND owner_id = ?
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT -1 OFFSET ?
                """,
                (owner_type, owner_id, keep_count),
            ).fetchall()
            pruned = [dict(row) for row in rows]
            for row in rows:
                session_id = str(row["id"])
                connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                connection.execute(
                    "DELETE FROM sessions WHERE id = ? AND owner_type = ? AND owner_id = ?",
                    (session_id, owner_type, owner_id),
                )
        return pruned

    def count_rate_limit_events(
        self,
        *,
        owner_type: str,
        owner_id: str,
        action: str,
        since: str,
    ) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM rate_limit_events
                WHERE owner_type = ? AND owner_id = ? AND action = ? AND created_at > ?
                """,
                (owner_type, owner_id, action, since),
            ).fetchone()
        return int(row["count"]) if row else 0

    def oldest_rate_limit_event_since(
        self,
        *,
        owner_type: str,
        owner_id: str,
        action: str,
        since: str,
    ) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT created_at
                FROM rate_limit_events
                WHERE owner_type = ? AND owner_id = ? AND action = ? AND created_at > ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (owner_type, owner_id, action, since),
            ).fetchone()
        return str(row["created_at"]) if row else None

    def add_rate_limit_event(
        self,
        *,
        owner_type: str,
        owner_id: str,
        action: str,
        created_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO rate_limit_events (owner_type, owner_id, action, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (owner_type, owner_id, action, created_at),
            )

    def prune_rate_limit_events(self, before: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM rate_limit_events WHERE created_at < ?", (before,))
