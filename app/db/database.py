"""SQLite database access for Book Keeper."""

import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from app.core.config import settings


class Database:
    """Wraps SQLite connection handling and common query patterns."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def fetch_one(self, query: str, params: tuple = ()) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def execute(self, query: str, params: tuple = ()) -> int | None:
        """Run an INSERT/UPDATE/DELETE, commit, and return lastrowid."""
        with self._connect() as conn:
            cur = conn.execute(query, params)
            conn.commit()
            return cur.lastrowid

    def exists(self, query: str, params: tuple = ()) -> bool:
        with self._connect() as conn:
            return conn.execute(query, params).fetchone() is not None

    @contextmanager
    def transaction(self):
        """Yield a raw connection for multi-statement atomic operations."""
        with self._connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    hashed_password TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    token_hash TEXT PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

@lru_cache
def get_db() -> Database:
    return Database()