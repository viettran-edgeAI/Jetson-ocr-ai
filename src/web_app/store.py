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
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
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

                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                    ON sessions(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, created_at ASC);
                """
            )

    def create_session(
        self,
        *,
        session_id: str,
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
                    id, filename, content_type, original_path, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    filename,
                    content_type,
                    str(original_path),
                    status,
                    created_at,
                    created_at,
                ),
            )
        session = self.get_session(session_id)
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

    def delete_session(self, session_id: str) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        with self.connect() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
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

    def recent_sessions(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
