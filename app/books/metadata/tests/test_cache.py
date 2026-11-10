"""Unit tests for the SQLite metadata cache.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 19):

* Store metadata in cache.
* Retrieve existing cached result.
* Expired entries are ignored.
* Cache keys normalize ISBNs.
* Title/author queries produce stable cache keys.
* Different ISBNs do not collide.
* Failed provider responses are not cached indefinitely.
* Cache can be invalidated.
* Cached data retains source attribution.

Section 19 also requires a temporary SQLite database for isolation.  The
``temp_cache_db_path`` / ``temp_cache_db`` / ``metadata_cache`` /
``populated_cache`` fixtures from ``conftest.py`` provide exactly that: every
test gets a private ``metadata_cache.db`` under ``tmp_path`` and never touches
the real application database.

The cache is synchronous and pure SQLite (no network), so these tests are
deterministic and fast.  No HTTP mocking is required.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from app.books.metadata.cache import MetadataCache
from app.books.metadata.schemas import BookMetadata
from app.core.config import settings

# ---------------------------------------------------------------------------
# ISBN / query constants (mirroring the values used by ``conftest.py``)
# ---------------------------------------------------------------------------

ISBN_13 = "9780132350884"  # Clean Code, Prentice Hall, 2008
ISBN_10 = "0132350882"
ISBN_13_HYPHENATED = "978-0-13-235088-4"
ISBN_13_DIGITS_SPACED = "978 0 13 235088 4"

ISBN_13_OTHER = "9780135957059"  # A different book / edition
ISBN_10_OTHER = "0321125215"

TITLE = "Clean Code"
AUTHOR = "Robert C. Martin"

EXPECTED_ISBN_KEY = f"isbn:{ISBN_13}"
EXPECTED_TITLE_AUTHOR_KEY = "title_author:cleancode|robertc.martin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_rows(db_path: Path) -> list[dict[str, Any]]:
    """Return every cache row as a plain dict, in insertion order."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM metadata_cache ORDER BY cache_key"
            )
        ]
    finally:
        conn.close()


def _row_count(db_path: Path) -> int:
    """Return the number of cache rows."""
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM metadata_cache").fetchone()[0])
    finally:
        conn.close()


def _expire_entry(db_path: Path, cache_key: str) -> None:
    """Force an existing cache entry into the past."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE metadata_cache SET expires_at = ? WHERE cache_key = ?",
            (int(time.time()) - 1, cache_key),
        )
        conn.commit()
    finally:
        conn.close()


def _corrupt_entry(db_path: Path, cache_key: str) -> None:
    """Replace a row's payload with data that cannot be parsed."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE metadata_cache SET response_data = ? WHERE cache_key = ?",
            ("{not valid json", cache_key),
        )
        conn.commit()
    finally:
        conn.close()


# ===========================================================================
# 1. Store metadata in cache / retrieve existing cached result
# ===========================================================================


class TestStoreAndRetrieve:
    """``MetadataCache.set`` stores results and ``get`` returns them."""

    def test_set_then_get_returns_cached_candidate(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        cached = metadata_cache.get(isbn=ISBN_13)

        assert cached is not None
        assert len(cached) == 1
        assert isinstance(cached[0], BookMetadata)

    def test_cache_round_trip_preserves_every_field(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        cached = metadata_cache.get(isbn=ISBN_13)

        assert cached is not None
        assert cached[0].model_dump() == sample_book_metadata.model_dump()

    def test_cache_round_trip_preserves_multiple_candidates(
        self,
        metadata_cache: MetadataCache,
        multiple_editions: list[BookMetadata],
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=multiple_editions,
            provider="google_books",
        )

        cached = metadata_cache.get(title=TITLE, author=AUTHOR)

        assert cached is not None
        assert [c.model_dump() for c in cached] == [
            edition.model_dump() for edition in multiple_editions
        ]

    def test_get_returns_none_for_empty_cache(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache.get(isbn=ISBN_13) is None

    def test_populated_cache_fixture_returns_stored_candidate(
        self,
        populated_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        cached = populated_cache.get(isbn=ISBN_13)

        assert cached is not None
        assert cached == [sample_book_metadata]

    def test_cache_persists_across_instances(
        self,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        first = MetadataCache(db_path=temp_cache_db_path)
        first.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        second = MetadataCache(db_path=temp_cache_db_path)

        assert second.get(isbn=ISBN_13) == [sample_book_metadata]

    def test_set_overwrites_an_existing_entry(
        self,
        metadata_cache: MetadataCache,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        original = book_metadata_factory(source="google_books")
        replacement = book_metadata_factory(
            title="Clean Code (2nd Edition)",
            source="open_library",
            source_id="OL999M",
        )

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[original],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[replacement],
            provider="open_library",
        )

        cached = metadata_cache.get(isbn=ISBN_13)

        assert cached is not None
        assert len(cached) == 1
        assert cached[0].title == "Clean Code (2nd Edition)"
        assert cached[0].source == "open_library"

    def test_title_author_query_round_trip(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(title=TITLE, author=AUTHOR) == [
            sample_book_metadata
        ]

    def test_isbn_query_ignores_title_and_author_for_storage(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        """An ISBN query is stored under the ISBN key, not the title key."""
        metadata_cache.set(
            isbn=ISBN_13,
            title=TITLE,
            author=AUTHOR,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(isbn=ISBN_13) == [sample_book_metadata]
        # The title/author form is a different key and must miss.
        assert metadata_cache.get(title=TITLE, author=AUTHOR) is None


# ===========================================================================
# 2. Cache key normalization and stability
# ===========================================================================


class TestCacheKeyNormalization:
    """``_make_cache_key`` / ``_normalize_for_cache`` semantics."""

    @pytest.mark.parametrize(
        "isbn_variant",
        [
            ISBN_13,
            ISBN_13_HYPHENATED,
            ISBN_13_DIGITS_SPACED,
            f"  {ISBN_13_HYPHENATED}  ",
            "978 - 0 - 13 - 235088 - 4",
        ],
    )
    def test_isbn_variants_share_one_key(
        self,
        metadata_cache: MetadataCache,
        isbn_variant: str,
    ) -> None:
        assert (
            metadata_cache._make_cache_key(isbn_variant, None, None)
            == EXPECTED_ISBN_KEY
        )

    @pytest.mark.parametrize(
        "isbn_variant",
        [ISBN_10, "0-13-235088-2", "013 235088 2"],
    )
    def test_isbn10_variants_share_one_key(
        self,
        metadata_cache: MetadataCache,
        isbn_variant: str,
    ) -> None:
        assert (
            metadata_cache._make_cache_key(isbn_variant, None, None)
            == f"isbn:{ISBN_10}"
        )

    def test_isbn_key_is_case_insensitive_for_trailing_x(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        upper = metadata_cache._make_cache_key("0-8044-2957-X", None, None)
        lower = metadata_cache._make_cache_key("080442957x", None, None)

        assert upper == lower == "isbn:080442957x"

    def test_isbn_key_has_stable_prefix(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        key = metadata_cache._make_cache_key(ISBN_13, None, None)

        assert key.startswith("isbn:")
        assert key == EXPECTED_ISBN_KEY

    @pytest.mark.parametrize(
        "title_variant, author_variant",
        [
            (TITLE, AUTHOR),
            ("clean code", "robert c. martin"),
            ("Clean-Code", "Robert C. Martin"),
            ("  Clean   Code  ", "  Robert  C.  Martin  "),
            ("CLEAN CODE", "ROBERT C. MARTIN"),
        ],
    )
    def test_title_author_variants_share_one_key(
        self,
        metadata_cache: MetadataCache,
        title_variant: str,
        author_variant: str,
    ) -> None:
        key = metadata_cache._make_cache_key(None, title_variant, author_variant)

        assert key == EXPECTED_TITLE_AUTHOR_KEY

    def test_title_author_key_is_stable_across_calls(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        keys = {
            metadata_cache._make_cache_key(None, TITLE, AUTHOR)
            for _ in range(5)
        }

        assert keys == {EXPECTED_TITLE_AUTHOR_KEY}

    def test_title_only_key_is_deterministic(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        key = metadata_cache._make_cache_key(None, TITLE, None)

        assert key == "title_author:cleancode|"

    def test_author_only_key_is_deterministic(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        key = metadata_cache._make_cache_key(None, None, AUTHOR)

        assert key == "title_author:|robertc.martin"

    def test_missing_isbn_falls_back_to_title_author_key(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache._make_cache_key(None, TITLE, AUTHOR) == (
            EXPECTED_TITLE_AUTHOR_KEY
        )

    def test_isbn_dominates_title_and_author(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        with_isbn = metadata_cache._make_cache_key(ISBN_13, "Other", "Someone")
        without_extra = metadata_cache._make_cache_key(ISBN_13, None, None)

        assert with_isbn == without_extra == EXPECTED_ISBN_KEY

    def test_different_titles_produce_different_keys(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache._make_cache_key(
            None, "Clean Code", AUTHOR
        ) != metadata_cache._make_cache_key(None, "Clean Architecture", AUTHOR)

    def test_different_authors_produce_different_keys(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache._make_cache_key(
            None, TITLE, AUTHOR
        ) != metadata_cache._make_cache_key(None, TITLE, "Someone Else")

    def test_normalize_for_cache_handles_none_and_empty(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache._normalize_for_cache(None) == ""
        assert metadata_cache._normalize_for_cache("") == ""

    def test_isbn_and_title_queries_use_different_namespaces(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        isbn_key = metadata_cache._make_cache_key(ISBN_13, None, None)
        title_key = metadata_cache._make_cache_key(None, ISBN_13, None)

        assert isbn_key != title_key
        assert isbn_key == f"isbn:{ISBN_13}"
        assert title_key == f"title_author:{ISBN_13}|"


class TestDifferentIsbnsDoNotCollide:
    """Distinct ISBNs must map to distinct keys *and* distinct rows."""

    def test_different_isbn13_keys_differ(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache._make_cache_key(
            ISBN_13, None, None
        ) != metadata_cache._make_cache_key(ISBN_13_OTHER, None, None)

    def test_isbn10_and_isbn13_keys_differ(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        """A 10-digit and a 13-digit ISBN are different keys even for one book."""
        assert metadata_cache._make_cache_key(
            ISBN_10, None, None
        ) != metadata_cache._make_cache_key(ISBN_13, None, None)

    def test_adjacent_isbns_do_not_collide(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        keys = {
            metadata_cache._make_cache_key(isbn, None, None)
            for isbn in (
                "9780132350884",
                "9780132350885",
                "9780132350883",
                "978013235088",
            )
        }

        assert len(keys) == 4

    def test_cached_values_are_stored_per_isbn(
        self,
        metadata_cache: MetadataCache,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        first = book_metadata_factory(
            title="Clean Code",
            isbn_13=ISBN_13,
            isbn_10=ISBN_10,
        )
        second = book_metadata_factory(
            title="Clean Architecture",
            isbn_13=ISBN_13_OTHER,
            isbn_10=ISBN_10_OTHER,
            source="open_library",
            source_id="OL999M",
        )

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[first],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13_OTHER,
            title=None,
            author=None,
            candidates=[second],
            provider="open_library",
        )

        cached_first = metadata_cache.get(isbn=ISBN_13)
        cached_second = metadata_cache.get(isbn=ISBN_13_OTHER)

        assert cached_first is not None
        assert cached_second is not None
        assert cached_first[0].title == "Clean Code"
        assert cached_second[0].title == "Clean Architecture"
        assert cached_first[0].isbn_13 != cached_second[0].isbn_13

    def test_hyphenated_lookup_reads_digits_entry(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(isbn=ISBN_13_HYPHENATED) == [
            sample_book_metadata
        ]


# ===========================================================================
# 3. Expiration (TTL)
# ===========================================================================


class TestExpiration:
    """Expired entries are ignored on read and removable in bulk."""

    def test_entry_within_ttl_is_returned(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(isbn=ISBN_13) == [sample_book_metadata]

    def test_negative_ttl_entry_is_ignored(
        self,
        temp_cache_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_book_metadata: BookMetadata,
    ) -> None:
        """An entry whose TTL already elapsed is treated as a cache miss."""
        monkeypatch.setattr(settings, "METADATA_CACHE_TTL", -1)
        cache = MetadataCache(db_path=temp_cache_db_path)

        cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        # The row exists, but it is expired and must not be served.
        assert _row_count(temp_cache_db_path) == 1
        assert cache.get(isbn=ISBN_13) is None

    def test_entry_expired_in_database_is_ignored(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )
        _expire_entry(temp_cache_db_path, EXPECTED_ISBN_KEY)

        assert metadata_cache.get(isbn=ISBN_13) is None

    def test_ttl_is_applied_to_expiry_timestamp(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        row = _fetch_rows(temp_cache_db_path)[0]

        assert row["expires_at"] - row["retrieved_at"] == (
            settings.METADATA_CACHE_TTL
        )

    def test_custom_ttl_is_respected(
        self,
        temp_cache_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_book_metadata: BookMetadata,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_CACHE_TTL", 120)
        cache = MetadataCache(db_path=temp_cache_db_path)

        cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        row = _fetch_rows(temp_cache_db_path)[0]

        assert row["expires_at"] - row["retrieved_at"] == 120

    def test_clear_expired_removes_only_expired_entries(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        valid = book_metadata_factory(isbn_13=ISBN_13, isbn_10=ISBN_10)
        expired = book_metadata_factory(
            title="Clean Architecture",
            isbn_13=ISBN_13_OTHER,
            isbn_10=ISBN_10_OTHER,
            source="open_library",
            source_id="OL999M",
        )

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[valid],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13_OTHER,
            title=None,
            author=None,
            candidates=[expired],
            provider="open_library",
        )
        _expire_entry(temp_cache_db_path, f"isbn:{ISBN_13_OTHER}")

        removed = metadata_cache.clear_expired()

        assert removed == 1
        assert _row_count(temp_cache_db_path) == 1
        assert metadata_cache.get(isbn=ISBN_13) == [valid]
        assert metadata_cache.get(isbn=ISBN_13_OTHER) is None

    def test_clear_expired_keeps_all_valid_entries(
        self,
        populated_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        assert populated_cache.clear_expired() == 0
        assert _row_count(temp_cache_db_path) == 1
        assert populated_cache.get(isbn=ISBN_13) is not None

    def test_clear_expired_returns_zero_when_cache_is_empty(
        self,
        metadata_cache: MetadataCache,
    ) -> None:
        assert metadata_cache.clear_expired() == 0

    def test_clear_expired_can_remove_every_entry(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        multiple_editions: list[BookMetadata],
    ) -> None:
        """All rows can be expired; ``clear_expired`` reports the total."""
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=multiple_editions,
            provider="google_books",
        )
        _expire_entry(temp_cache_db_path, EXPECTED_TITLE_AUTHOR_KEY)

        assert metadata_cache.clear_expired() == 1
        assert metadata_cache.get(title=TITLE, author=AUTHOR) is None


# ===========================================================================
# 4. Invalidation
# ===========================================================================


class TestInvalidation:
    """``invalidate`` and ``invalidate_for`` remove cached entries."""

    def test_invalidate_by_explicit_key(
        self,
        populated_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        populated_cache.invalidate(EXPECTED_ISBN_KEY)

        assert populated_cache.get(isbn=ISBN_13) is None
        assert _row_count(temp_cache_db_path) == 0

    def test_invalidate_for_isbn_query(
        self,
        populated_cache: MetadataCache,
    ) -> None:
        populated_cache.invalidate_for(isbn=ISBN_13)

        assert populated_cache.get(isbn=ISBN_13) is None

    def test_invalidate_for_normalizes_hyphenated_isbn(
        self,
        populated_cache: MetadataCache,
    ) -> None:
        populated_cache.invalidate_for(isbn=ISBN_13_HYPHENATED)

        assert populated_cache.get(isbn=ISBN_13) is None

    def test_invalidate_for_title_author_query(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        metadata_cache.invalidate_for(
            title="  clean-code ", author="ROBERT C. MARTIN"
        )

        assert metadata_cache.get(title=TITLE, author=AUTHOR) is None

    def test_invalidate_unknown_key_is_a_no_op(
        self,
        populated_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        populated_cache.invalidate("isbn:0000000000000")

        assert _row_count(temp_cache_db_path) == 1
        assert populated_cache.get(isbn=ISBN_13) is not None

    def test_invalidate_only_removes_the_targeted_entry(
        self,
        metadata_cache: MetadataCache,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        first = book_metadata_factory(isbn_13=ISBN_13, isbn_10=ISBN_10)
        second = book_metadata_factory(
            title="Clean Architecture",
            isbn_13=ISBN_13_OTHER,
            isbn_10=ISBN_10_OTHER,
            source="open_library",
            source_id="OL999M",
        )
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[first],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13_OTHER,
            title=None,
            author=None,
            candidates=[second],
            provider="open_library",
        )

        metadata_cache.invalidate_for(isbn=ISBN_13)

        assert metadata_cache.get(isbn=ISBN_13) is None
        assert metadata_cache.get(isbn=ISBN_13_OTHER) == [second]

    def test_entry_can_be_re_populated_after_invalidation(
        self,
        populated_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        populated_cache.invalidate_for(isbn=ISBN_13)
        assert populated_cache.get(isbn=ISBN_13) is None

        populated_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert populated_cache.get(isbn=ISBN_13) == [sample_book_metadata]

    def test_invalidating_isbn_does_not_clear_title_author_entry(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=[sample_book_metadata],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13,
            title=TITLE,
            author=AUTHOR,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        metadata_cache.invalidate_for(isbn=ISBN_13)

        assert metadata_cache.get(isbn=ISBN_13) is None
        assert metadata_cache.get(title=TITLE, author=AUTHOR) is not None


# ===========================================================================
# 5. Failed / incomplete responses are not cached
# ===========================================================================


class TestFailedResponsesAreNotCached:
    """Empty or unusable results must not poison the cache."""

    def test_empty_candidate_list_is_not_stored(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[],
            provider="google_books",
        )

        assert _row_count(temp_cache_db_path) == 0
        assert metadata_cache.get(isbn=ISBN_13) is None

    def test_empty_candidate_list_does_not_overwrite_existing_entry(
        self,
        populated_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        populated_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[],
            provider="open_library",
        )

        assert _row_count(temp_cache_db_path) == 1
        assert populated_cache.get(isbn=ISBN_13) == [sample_book_metadata]

    def test_empty_results_can_be_retried_and_cached_later(
        self,
        metadata_cache: MetadataCache,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[],
            provider="google_books",
        )
        assert metadata_cache.get(isbn=ISBN_13) is None

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(isbn=ISBN_13) == [sample_book_metadata]

    def test_empty_title_author_result_is_not_stored(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title=TITLE,
            author=AUTHOR,
            candidates=[],
            provider="open_library",
        )

        assert _row_count(temp_cache_db_path) == 0
        assert metadata_cache.get(title=TITLE, author=AUTHOR) is None

    def test_corrupt_payload_is_treated_as_a_miss(
        self,
        populated_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        _corrupt_entry(temp_cache_db_path, EXPECTED_ISBN_KEY)

        assert populated_cache.get(isbn=ISBN_13) is None

    def test_unparseable_candidates_are_skipped(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        """A payload with no valid candidates behaves like a cache miss."""
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )
        conn = sqlite3.connect(str(temp_cache_db_path))
        try:
            conn.execute(
                "UPDATE metadata_cache SET response_data = ? WHERE cache_key = ?",
                (json.dumps([{"not": "a candidate"}]), EXPECTED_ISBN_KEY),
            )
            conn.commit()
        finally:
            conn.close()

        assert metadata_cache.get(isbn=ISBN_13) is None


# ===========================================================================
# 6. Source attribution and provenance
# ===========================================================================


class TestSourceAttribution:
    """Cached rows retain provider/source information for later reporting."""

    def test_source_field_survives_the_round_trip(
        self,
        metadata_cache: MetadataCache,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        candidate = book_metadata_factory(
            source="open_library", source_id="OL123456M"
        )
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[candidate],
            provider="open_library",
        )

        cached = metadata_cache.get(isbn=ISBN_13)

        assert cached is not None
        assert cached[0].source == "open_library"
        assert cached[0].source_id == "OL123456M"
        assert cached[0].confidence == candidate.confidence

    def test_provider_column_records_the_requesting_provider(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        row = _fetch_rows(temp_cache_db_path)[0]

        assert row["provider"] == "google_books"
        assert row["query_type"] == "isbn"
        assert row["query_value"] == ISBN_13

    def test_title_author_row_records_normalized_query_metadata(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        metadata_cache.set(
            isbn=None,
            title="  Clean Code ",
            author="Robert C. Martin",
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        row = _fetch_rows(temp_cache_db_path)[0]

        assert row["query_type"] == "title_author"
        assert row["query_value"] == "cleancode|robertc.martin"

    def test_multiple_providers_keep_their_own_attribution(
        self,
        metadata_cache: MetadataCache,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        google = book_metadata_factory(
            isbn_13=ISBN_13, isbn_10=ISBN_10, source="google_books"
        )
        open_library = book_metadata_factory(
            title="Clean Architecture",
            isbn_13=ISBN_13_OTHER,
            isbn_10=ISBN_10_OTHER,
            source="open_library",
            source_id="OL999M",
        )

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[google],
            provider="google_books",
        )
        metadata_cache.set(
            isbn=ISBN_13_OTHER,
            title=None,
            author=None,
            candidates=[open_library],
            provider="open_library",
        )

        cached_google = metadata_cache.get(isbn=ISBN_13)
        cached_open_library = metadata_cache.get(isbn=ISBN_13_OTHER)

        assert cached_google is not None
        assert cached_open_library is not None
        assert cached_google[0].source == "google_books"
        assert cached_open_library[0].source == "open_library"

    def test_retrieved_timestamp_is_recorded(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        before = int(time.time())
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )
        after = int(time.time())

        row = _fetch_rows(temp_cache_db_path)[0]

        assert before <= row["retrieved_at"] <= after
        assert row["expires_at"] > row["retrieved_at"]

    def test_serialized_payload_contains_source_attribution(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
        book_metadata_factory: Callable[..., BookMetadata],
    ) -> None:
        candidate = book_metadata_factory(
            source="open_library", source_id="OL123456M"
        )
        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[candidate],
            provider="open_library",
        )

        row = _fetch_rows(temp_cache_db_path)[0]
        payload = json.loads(row["response_data"])

        assert payload[0]["source"] == "open_library"
        assert payload[0]["source_id"] == "OL123456M"


# ===========================================================================
# 7. Isolation and schema
# ===========================================================================


class TestDatabaseIsolation:
    """The per-test temporary database is independent of the app database."""

    def test_cache_uses_the_temporary_path(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        assert temp_cache_db_path.exists()
        assert metadata_cache._db_path == temp_cache_db_path

    def test_temp_cache_db_alias_is_isolated(
        self,
        temp_cache_db: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        cache = MetadataCache(db_path=temp_cache_db)
        cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert temp_cache_db.exists()
        assert _row_count(temp_cache_db) == 1

    def test_two_temporary_caches_do_not_share_data(
        self,
        metadata_cache: MetadataCache,
        tmp_path: Path,
        sample_book_metadata: BookMetadata,
    ) -> None:
        other_path = tmp_path / "other_metadata_cache.db"
        other_cache = MetadataCache(db_path=other_path)

        metadata_cache.set(
            isbn=ISBN_13,
            title=None,
            author=None,
            candidates=[sample_book_metadata],
            provider="google_books",
        )

        assert metadata_cache.get(isbn=ISBN_13) is not None
        assert other_cache.get(isbn=ISBN_13) is None

    def test_expected_schema_columns_exist(
        self,
        metadata_cache: MetadataCache,
        temp_cache_db_path: Path,
    ) -> None:
        conn = sqlite3.connect(str(temp_cache_db_path))
        try:
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(metadata_cache)")
            }
        finally:
            conn.close()

        assert columns == {
            "cache_key",
            "provider",
            "query_type",
            "query_value",
            "response_data",
            "retrieved_at",
            "expires_at",
        }
