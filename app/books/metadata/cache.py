"""SQLite-based metadata cache.

Provides persistent caching for book metadata lookups so that
repeated searches for the same ISBN or title/author combination
do not trigger redundant external API calls.

Cache entries have a configurable TTL, are keyed by normalised
search parameters, and retain source attribution.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.books.metadata.schemas import BookMetadata
from app.core.config import settings

logger = logging.getLogger(__name__)


class MetadataCache:
    """Persistent SQLite cache for book metadata lookup results.

    Cache keys are derived from the normalised search parameters
    (ISBN, title, author).  Entries expire after the TTL configured
    in ``settings.METADATA_CACHE_TTL`` (default 86400 seconds).
    Expired entries are ignored on retrieval and can be purged
    explicitly.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or (settings.DATABASE_PATH.parent / "metadata_cache.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl = settings.METADATA_CACHE_TTL
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open (and return) a new SQLite connection."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the cache table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata_cache (
                    cache_key    TEXT PRIMARY KEY,
                    provider     TEXT NOT NULL,
                    query_type   TEXT NOT NULL,
                    query_value  TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    retrieved_at INTEGER NOT NULL,
                    expires_at   INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_for_cache(value: Optional[str]) -> str:
        """Normalise a single query value for cache-key generation.

        Strips surrounding whitespace, collapses internal whitespace,
        removes hyphens (important for ISBNs), and lowercases so that
        e.g. ``978-0-13-595705-9`` and ``9780135957059`` map to the
        same key.
        """
        if not value:
            return ""
        import re

        cleaned = value.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace("-", "").replace(" ", "")
        return cleaned.lower()

    def _make_cache_key(
        self,
        isbn: Optional[str],
        title: Optional[str],
        author: Optional[str],
    ) -> str:
        """Generate a stable, consistent cache key from search parameters.

        When an ISBN is provided it dominates the key; otherwise the
        key is derived from the normalised title and author.
        """
        if isbn:
            norm_isbn = self._normalize_for_cache(isbn)
            return f"isbn:{norm_isbn}"

        norm_title = self._normalize_for_cache(title or "")
        norm_author = self._normalize_for_cache(author or "")
        return f"title_author:{norm_title}|{norm_author}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        isbn: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Optional[List[BookMetadata]]:
        """Retrieve cached candidates if they exist and have not expired.

        Returns ``None`` on cache miss or when the entry has expired
        (expired entries are silently ignored – the caller should
        re-fetch from providers).

        Args:
            isbn: Normalised ISBN (no hyphens/spaces), optional.
            title: Book title, optional.
            author: Author name, optional.

        Returns:
            List of ``BookMetadata`` candidates, or ``None``.
        """
        cache_key = self._make_cache_key(isbn, title, author)
        now = int(time.time())

        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT response_data, expires_at
                    FROM metadata_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning(f"Cache read error for key {cache_key!r}: {exc}")
            return None

        if row is None:
            logger.debug(f"Cache miss: {cache_key!r}")
            return None

        expires_at = row["expires_at"]
        if now > expires_at:
            logger.debug(f"Cache entry expired: {cache_key!r}")
            return None

        try:
            raw = json.loads(row["response_data"])
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Corrupt cache data for key {cache_key!r}: {exc}")
            return None

        # Reconstruct BookMetadata instances
        candidates: List[BookMetadata] = []
        for item in raw:
            try:
                candidates.append(BookMetadata(**item))
            except Exception as exc:
                logger.warning(
                    f"Skipping unparseable cached candidate for key {cache_key!r}: {exc}"
                )
                continue

        logger.debug(f"Cache hit: {cache_key!r} ({len(candidates)} candidate(s))")
        return candidates if candidates else None

    def set(
        self,
        isbn: Optional[str],
        title: Optional[str],
        author: Optional[str],
        candidates: Sequence[BookMetadata],
        provider: str,
    ) -> None:
        """Store (or overwrite) candidates in the cache.

        Only caches non-empty candidate lists.  Expiry is computed
        as ``now + TTL``.

        Args:
            isbn: Normalised ISBN, optional.
            title: Book title, optional.
            author: Author name, optional.
            candidates: List of ``BookMetadata`` to cache.
            provider: Provider name (e.g. ``"google_books"``) for
                      attribution.
        """
        if not candidates:
            logger.debug("Not caching empty candidate list")
            return

        cache_key = self._make_cache_key(isbn, title, author)
        query_type = "isbn" if isbn else "title_author"
        query_value = (
            self._normalize_for_cache(isbn)
            if isbn
            else f"{self._normalize_for_cache(title)}|{self._normalize_for_cache(author)}"
        )

        now = int(time.time())
        expires_at = now + self._ttl

        # Serialise candidates as JSON-friendly dicts
        serialised = [
            c.model_dump(exclude_none=False) for c in candidates
        ]

        try:
            response_data = json.dumps(serialised, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.warning(f"Failed to serialise candidates for caching: {exc}")
            return

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata_cache
                        (cache_key, provider, query_type, query_value,
                         response_data, retrieved_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        provider,
                        query_type,
                        query_value,
                        response_data,
                        now,
                        expires_at,
                    ),
                )
                conn.commit()
            logger.debug(f"Cached {len(candidates)} candidate(s) under key {cache_key!r}")
        except sqlite3.Error as exc:
            logger.warning(f"Failed to write cache entry for key {cache_key!r}: {exc}")

    def invalidate(self, cache_key: str) -> None:
        """Remove a specific cache entry by its key.

        Args:
            cache_key: The cache key to remove.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM metadata_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()
            logger.debug(f"Invalidated cache key: {cache_key!r}")
        except sqlite3.Error as exc:
            logger.warning(f"Failed to invalidate cache key {cache_key!r}: {exc}")

    def invalidate_for(
        self,
        isbn: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
    ) -> None:
        """Convenience wrapper that computes the key and invalidates.

        Args:
            isbn: Normalised ISBN, optional.
            title: Book title, optional.
            author: Author name, optional.
        """
        cache_key = self._make_cache_key(isbn, title, author)
        self.invalidate(cache_key)

    def clear_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of rows removed.
        """
        now = int(time.time())
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM metadata_cache WHERE expires_at <= ?",
                    (now,),
                )
                conn.commit()
                count = cur.rowcount
            logger.info(f"Cleared {count} expired cache entry/entries")
            return count
        except sqlite3.Error as exc:
            logger.warning(f"Failed to clear expired entries: {exc}")
            return 0