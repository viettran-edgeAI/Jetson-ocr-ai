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
                    password_hash TEXT,
                    tier TEXT NOT NULL DEFAULT 'free',
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
                """
            )
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

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
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
        password_hash: str | None,
        tier: str,
        disabled: bool,
        created_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO users (id, email, password_hash, tier, disabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, email, password_hash, tier, int(disabled), created_at, created_at),
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
        tier: str,
        updated_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, tier = ?, disabled = 0, updated_at = ?
                WHERE id = ?
                """,
                (password_hash, tier, updated_at, user_id),
            )
        user = self.get_user_by_id(user_id)
        if user is None:
            raise RuntimeError("failed to activate user")
        return user

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
        return self.create_user(
            user_id=user_id,
            email=email,
            password_hash=None,
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
