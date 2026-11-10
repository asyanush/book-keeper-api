"""Unit tests for the Open Library metadata provider.

The provider talks to the Open Library REST API through
:class:`httpx.AsyncClient`.  Every request in this module is served by the
``MockHTTPRouter`` fixtures declared in ``conftest.py``; the autouse network
guard there fails the test if a live call ever slips through.

Tests are intentionally synchronous: the project does not depend on
``pytest-asyncio``, so coroutines are driven with :func:`asyncio.run`.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 14):

* ISBN lookup returns a valid edition.
* Search by title returns multiple results.
* Search by author returns relevant results.
* Author IDs are resolved correctly.
* Missing author data is handled.
* Missing edition/work relationships are handled.
* Missing cover ID is handled.
* HTTP 404 returns an empty result.
* HTTP 429 and 5xx errors are handled (including retry behaviour).
* Timeouts do not crash orchestration.
* Malformed JSON is handled.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import httpx
import pytest

from app.books.metadata.providers.base import ProviderResult
from app.books.metadata.providers.open_library import (
    COVER_BASE_URL,
    OpenLibraryProvider,
)
from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISBN_13 = "9780132350884"
ISBN_10 = "0132350882"
EDITION_ID = "OL123456M"
WORK_ID = "OL98765W"
AUTHOR_KEY = "/authors/OL12345A"
AUTHOR_NAME = "Robert C. Martin"
COVER_ID = 1234567
COVER_URL = f"{COVER_BASE_URL}/{COVER_ID}-L.jpg"

#: The provider's real client factory, captured before the autouse test
#: fixture installs its fail-closed / mocked ``_build_client``.
_REAL_BUILD_CLIENT = OpenLibraryProvider.__dict__["_build_client"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """Drive *coro* to completion on a fresh event loop."""
    return asyncio.run(coro)


def candidates_of(result: ProviderResult) -> list[dict[str, Any]]:
    """Return the candidate list from a successful provider result."""
    assert result.success is True, f"Expected success, got error: {result.error!r}"
    assert result.data is not None
    return result.data["candidates"]


def single_candidate(result: ProviderResult) -> dict[str, Any]:
    """Return the only candidate from a successful provider result."""
    candidates = candidates_of(result)
    assert len(candidates) == 1, f"Expected exactly one candidate, got {candidates!r}"
    return candidates[0]


def endpoint_requests(router: Any, fragment: str) -> list[httpx.Request]:
    """Return the recorded requests whose URL contains *fragment*."""
    return [request for request in router.requests if fragment in str(request.url)]


@pytest.fixture
def provider(open_library_provider: OpenLibraryProvider) -> OpenLibraryProvider:
    """A fresh provider instance (its ``_build_client`` is patched by mocks)."""
    return open_library_provider


# ===========================================================================
# 1. Provider configuration
# ===========================================================================


class TestProviderConfiguration:
    """Configuration is read from ``settings`` and no credentials leak."""

    def test_provider_name_is_open_library(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider.name == "open_library"

    def test_provider_reads_base_url_from_settings(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider.base_url == settings.OPEN_LIBRARY_API_URL
        assert provider.base_url == "https://openlibrary.org"

    def test_provider_reads_timeout_and_retries_from_settings(
        self,
        provider: OpenLibraryProvider,
    ) -> None:
        assert provider.timeout == settings.METADATA_TIMEOUT
        assert provider.max_retries == settings.METADATA_MAX_RETRIES

    def test_provider_snapshots_configuration_on_construction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_TIMEOUT", 3)
        monkeypatch.setattr(settings, "METADATA_MAX_RETRIES", 0)
        monkeypatch.setattr(
            settings, "OPEN_LIBRARY_API_URL", "https://openlibrary.test"
        )

        provider = OpenLibraryProvider()

        assert provider.timeout == 3
        assert provider.max_retries == 0
        assert provider.base_url == "https://openlibrary.test"

    def test_build_client_applies_configured_timeout_and_redirects(
        self, provider: OpenLibraryProvider
    ) -> None:
        # The autouse fixture replaces ``_build_client`` with a mock router;
        # ``_REAL_BUILD_CLIENT`` is the original captured at import time.
        client = _REAL_BUILD_CLIENT(provider)
        try:
            assert isinstance(client, httpx.AsyncClient)
            assert client.follow_redirects is True
            assert client.timeout.read == provider.timeout
            assert client.timeout.connect == provider.timeout
        finally:
            run_async(client.aclose())

    def test_open_library_is_served_by_the_shared_provider_router(
        self,
        mock_metadata_providers,
        provider: OpenLibraryProvider,
    ) -> None:
        """The combined provider fixture also routes Open Library requests."""

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert single_candidate(result)["isbn_13"] == ISBN_13
        assert mock_metadata_providers["open_library"].request_count >= 1


# ===========================================================================
# 2. ISBN lookup
# ===========================================================================


class TestSearchByIsbn:
    """``OpenLibraryProvider.search_by_isbn`` behaviour."""

    def test_isbn_lookup_returns_valid_edition(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["title"] == "Clean Code"
        assert candidate["subtitle"] == "A Handbook of Agile Software Craftsmanship"
        assert candidate["authors"] == [AUTHOR_NAME]
        assert candidate["publisher"] == "Prentice Hall"
        assert candidate["published_date"] == "2008"
        assert candidate["description"].startswith("Even bad code can function")
        assert candidate["isbn_10"] == ISBN_10
        assert candidate["isbn_13"] == ISBN_13
        assert candidate["page_count"] == 464
        assert candidate["categories"] == [
            "Computer programming",
            "Software engineering",
            "Agile software development",
        ]
        assert candidate["language"] == "eng"
        assert candidate["cover_url"] == COVER_URL
        assert candidate["source"] == "open_library"
        assert candidate["source_id"] == EDITION_ID
        assert candidate["confidence"] == 0.0
        assert result.raw_response == open_library_isbn_response

    def test_isbn_lookup_uses_isbn_endpoint(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_isbn(ISBN_13))

        # The author-resolution request is issued afterwards, so assert on
        # the first (ISBN) request explicitly.
        request = mock_open_library_http.requests[0]
        assert request.method == "GET"
        assert request.url.path == f"/isbn/{ISBN_13}.json"
        assert str(request.url).startswith(settings.OPEN_LIBRARY_API_URL)

    def test_isbn_lookup_passes_isbn_verbatim(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_isbn(ISBN_10))

        assert mock_open_library_http.requests[0].url.path == (
            f"/isbn/{ISBN_10}.json"
        )

    def test_isbn_lookup_resolves_author_ids_to_names(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_isbn(ISBN_13))

        urls = mock_open_library_http.request_urls
        assert len(urls) == 2
        assert any(f"/isbn/{ISBN_13}.json" in url for url in urls)
        assert any(f"/authors/OL12345A.json" in url for url in urls)

    def test_isbn_lookup_does_not_fetch_work_metadata(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_isbn(ISBN_13))

        assert not any("/works/" in url for url in mock_open_library_http.request_urls)

    def test_isbn_lookup_returns_empty_candidate_for_empty_payload(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(path="/isbn/", json={})

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert result.error is None
        assert result.data == {"candidates": []}

    def test_isbn_lookup_candidate_is_normalizable(
        self,
        mock_open_library_http,
        normalizer,
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(provider.search_by_isbn(ISBN_13))

        metadata = normalizer.normalize_candidate(single_candidate(result))

        assert metadata.title == "Clean Code"
        assert metadata.authors == [AUTHOR_NAME]
        assert metadata.isbn_10 == ISBN_10
        assert metadata.isbn_13 == ISBN_13
        assert metadata.page_count == 464
        assert metadata.source == "open_library"
        assert metadata.source_id == EDITION_ID
        assert metadata.confidence == 0.0

    def test_isbn_lookup_prefers_dedicated_isbn_fields(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["isbn"] = ["9999999999999", "9999999999"]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(run_async(provider.search_by_isbn(ISBN_13)))

        assert candidate["isbn_10"] == ISBN_10
        assert candidate["isbn_13"] == ISBN_13


# ===========================================================================
# 3. Search by title
# ===========================================================================


class TestSearchByTitle:
    """``search_by_title_author`` with title-only / title+author queries."""

    def test_search_by_title_returns_multiple_results(
        self,
        mock_open_library_http,
        open_library_search_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(provider.search_by_title_author("Clean Code"))

        candidates = candidates_of(result)
        assert len(candidates) == 3
        assert [c["title"] for c in candidates] == [
            "Clean Code",
            "Clean Code",
            "Clean Architecture",
        ]
        assert all(c["source"] == "open_library" for c in candidates)
        assert [c["isbn_13"] for c in candidates] == [
            ISBN_13,
            "9780137081073",
            "9780134494166",
        ]
        assert result.raw_response == open_library_search_response

    def test_search_by_title_uses_search_endpoint(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code"))

        request = mock_open_library_http.get_request()
        assert request.url.path == "/search.json"
        assert request.url.params["title"] == "Clean Code"
        assert request.url.params["limit"] == "20"
        assert "author" not in request.url.params

    def test_search_result_fields_are_mapped(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(provider.search_by_title_author("Clean Code"))

        first = candidates_of(result)[0]
        assert first["subtitle"] == "A Handbook of Agile Software Craftsmanship"
        assert first["authors"] == [AUTHOR_NAME]
        assert first["publisher"] == "Prentice Hall"
        assert first["published_date"] == "2008"
        assert first["isbn_10"] == ISBN_10
        assert first["isbn_13"] == ISBN_13
        assert first["cover_url"] == COVER_URL
        assert first["categories"] == [
            "Computer programming",
            "Software engineering",
        ]
        assert first["language"] == "eng"
        assert first["source_id"] == EDITION_ID
        assert first["source"] == "open_library"
        assert first["confidence"] == 0.0
        # Search documents do not carry these fields.
        assert first["description"] is None
        assert first["page_count"] is None

    def test_search_result_source_id_uses_first_edition_key(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        # ``edition_key`` for the first doc is ``["OL123456M", "OL123457M"]``.
        result = run_async(provider.search_by_title_author("Clean Code"))

        assert candidates_of(result)[0]["source_id"] == EDITION_ID

    def test_search_results_build_distinct_cover_urls(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        candidates = candidates_of(
            run_async(provider.search_by_title_author("Clean Code"))
        )

        assert [c["cover_url"] for c in candidates] == [
            COVER_URL,
            f"{COVER_BASE_URL}/2345678-L.jpg",
            f"{COVER_BASE_URL}/3456789-L.jpg",
        ]

    def test_search_results_are_normalizable(
        self,
        mock_open_library_http,
        normalizer,
        provider: OpenLibraryProvider,
    ) -> None:
        candidates = candidates_of(
            run_async(provider.search_by_title_author("Clean Code"))
        )

        metadata = normalizer.normalize_candidate(candidates[0])
        assert metadata.title == "Clean Code"
        assert metadata.authors == [AUTHOR_NAME]
        assert metadata.isbn_13 == ISBN_13
        assert metadata.source == "open_library"

    def test_title_search_does_not_request_author_or_work_endpoints(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        # Author names are inlined in search documents, so exactly one HTTP
        # request should be made regardless of the number of results.
        run_async(provider.search_by_title_author("Clean Code"))

        assert mock_open_library_http.request_count == 1

    def test_search_with_empty_title_and_author_uses_query_fallback(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(provider.search_by_title_author("", AUTHOR_NAME))

        request = mock_open_library_http.get_request()
        assert request.url.path == "/search.json"
        assert request.url.params["q"] == AUTHOR_NAME
        assert "title" not in request.url.params
        assert candidates_of(result)

    def test_search_empty_title_and_no_author_still_queries_api(
        self,
        mock_open_library_http,
        open_library_empty_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", json=open_library_empty_response
        )

        result = run_async(provider.search_by_title_author(""))

        request = mock_open_library_http.get_request()
        assert request.url.params["title"] == ""
        assert result.success is True
        assert candidates_of(result) == []


# ===========================================================================
# 4. Search by author
# ===========================================================================


class TestSearchByAuthor:
    """Author-scoped searches return relevant results."""

    def test_search_by_author_returns_relevant_results(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        result = run_async(
            provider.search_by_title_author("Clean Code", AUTHOR_NAME)
        )

        candidates = candidates_of(result)
        assert candidates
        assert all(candidate["authors"] == [AUTHOR_NAME] for candidate in candidates)
        assert all(candidate["title"].startswith("Clean") for candidate in candidates)

    def test_search_by_author_includes_author_filter(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code", AUTHOR_NAME))

        request = mock_open_library_http.get_request()
        assert request.url.params["title"] == "Clean Code"
        assert request.url.params["author"] == AUTHOR_NAME

    def test_search_without_author_omits_author_filter(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code", author=None))

        assert "author" not in mock_open_library_http.get_request().url.params

    def test_search_with_empty_author_omits_author_filter(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code", author=""))

        assert "author" not in mock_open_library_http.get_request().url.params

    def test_search_by_author_without_title_uses_general_query(
        self,
        mock_open_library_http,
        open_library_search_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", json=open_library_search_response
        )

        result = run_async(provider.search_by_title_author("   ", AUTHOR_NAME))

        request = mock_open_library_http.get_request()
        assert request.url.params["q"] == AUTHOR_NAME
        assert len(candidates_of(result)) == 3


# ===========================================================================
# 5. Author resolution
# ===========================================================================


class TestAuthorResolution:
    """Author keys are resolved into human-readable names."""

    def test_author_key_is_resolved_to_name(
        self,
        mock_open_library_http,
        open_library_author_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["authors"] == ["Robert C. Martin"]
        author_request = mock_open_library_http.requests[-1]
        assert author_request.url.path == f"{AUTHOR_KEY}.json"

    def test_author_name_in_payload_avoids_extra_request(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["authors"] = [{"key": AUTHOR_KEY, "name": "Inline Author"}]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert single_candidate(result)["authors"] == ["Inline Author"]
        assert mock_open_library_http.request_count == 1

    def test_multiple_author_keys_are_resolved(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["authors"] = [
            {"key": "/authors/OL111A"},
            {"key": "/authors/OL222A"},
        ]
        mock_open_library_http.add_response(path="/isbn/", json=payload)
        mock_open_library_http.add_response(
            url="/authors/OL111A.json", json={"key": "/authors/OL111A", "name": "Jane Austen"}
        )
        mock_open_library_http.add_response(
            url="/authors/OL222A.json",
            json={"key": "/authors/OL222A", "name": "Charlotte Bronte"},
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert single_candidate(result)["authors"] == [
            "Jane Austen",
            "Charlotte Bronte",
        ]
        assert mock_open_library_http.request_count == 3

    def test_duplicate_author_names_are_deduplicated(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        # Both keys resolve to the same name via the shared author fallback.
        payload = copy.deepcopy(open_library_isbn_response)
        payload["authors"] = [
            {"key": "/authors/OL12345A"},
            {"key": "/authors/OL12345A"},
        ]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["authors"] == ["Robert C. Martin"]

    def test_personal_name_is_used_when_name_is_missing(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            url="/authors/OL12345A.json",
            json={"key": AUTHOR_KEY, "personal_name": "Robert Cecil Martin"},
        )

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["authors"] == ["Robert Cecil Martin"]

    def test_unresolvable_author_key_is_tolerated(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/authors/", status_code=404, text="Not Found"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["authors"] == []
        assert candidate["title"] == "Clean Code"

    def test_author_resolution_network_error_is_tolerated(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.ConnectError("author lookup failed"), path="/authors/"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["authors"] == []

    def test_resolve_authors_accepts_mixed_reference_formats(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        refs = [
            {"name": "Inline Author"},
            {"key": AUTHOR_KEY},
            "Plain String Author",
            {"key": ""},
        ]

        async def _resolve() -> list[str]:
            async with provider._build_client() as client:
                return await provider._resolve_authors(client, refs)

        names = run_async(_resolve())

        # Inline name and plain string are collected first; resolved keys are
        # appended afterwards, preserving each group's order.
        assert names == ["Inline Author", "Plain String Author", "Robert C. Martin"]
        assert mock_open_library_http.request_count == 1

    def test_authors_are_resolved_concurrently(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        refs = [f"/authors/OL{index}A" for index in range(1, 6)]
        for index in range(1, 6):
            mock_open_library_http.add_response(
                url=f"/authors/OL{index}A.json",
                json={"key": f"/authors/OL{index}A", "name": f"Author {index}"},
            )

        async def _resolve() -> list[str]:
            async with provider._build_client() as client:
                return await provider._resolve_authors(client, refs)

        names = run_async(_resolve())

        assert names == [f"Author {index}" for index in range(1, 6)]
        assert mock_open_library_http.request_count == 5


# ===========================================================================
# 6. Missing data
# ===========================================================================


class TestMissingData:
    """Incomplete edition / work payloads must not crash the provider."""

    def test_missing_author_data_returns_empty_authors(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload.pop("authors", None)
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["authors"] == []
        assert candidate["title"] == "Clean Code"
        # No author endpoint should be queried when no author refs exist.
        assert mock_open_library_http.request_count == 1

    def test_empty_author_list_returns_empty_authors(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["authors"] = []
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["authors"] == []

    def test_missing_edition_work_relationship_uses_edition_id(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload.pop("works", None)
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["source_id"] == EDITION_ID

    def test_missing_edition_and_work_keys_returns_none_source_id(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload.pop("key", None)
        payload.pop("works", None)
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["source_id"] is None
        assert candidate["title"] == "Clean Code"

    def test_missing_work_key_returns_none_work_id(
        self,
        provider: OpenLibraryProvider,
    ) -> None:
        assert provider._extract_work_id({}) is None
        assert provider._extract_work_id({"works": []}) is None
        # A work object without a ``key`` yields an empty string, which is
        # falsy and therefore treated as "missing" by ``source_id`` selection.
        assert not provider._extract_work_id({"works": [{}]})

    def test_missing_cover_id_returns_no_cover_url(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload.pop("covers", None)
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["cover_url"] is None

    def test_empty_covers_list_returns_no_cover_url(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["covers"] = []
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["cover_url"] is None

    def test_search_result_without_cover_returns_no_cover_url(
        self,
        mock_open_library_http,
        open_library_search_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_search_response)
        payload["docs"][0].pop("cover_i", None)
        mock_open_library_http.add_response(path="/search.json", json=payload)

        candidate = candidates_of(
            run_async(provider.search_by_title_author("Clean Code"))
        )[0]

        assert candidate["cover_url"] is None

    def test_missing_all_optional_fields_defaults_cleanly(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", json={"title": "Bare Minimum"}
        )

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["title"] == "Bare Minimum"
        assert candidate["subtitle"] is None
        assert candidate["authors"] == []
        assert candidate["publisher"] is None
        assert candidate["published_date"] is None
        assert candidate["description"] is None
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["page_count"] is None
        assert candidate["categories"] == []
        assert candidate["language"] is None
        assert candidate["cover_url"] is None
        assert candidate["source_id"] is None
        assert candidate["source"] == "open_library"

    def test_plain_string_description_is_preserved(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["description"] = "A plain-text description."
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["description"] == "A plain-text description."

    def test_unsupported_description_type_becomes_none(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["description"] = ["not", "a", "string"]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["description"] is None

    def test_publisher_as_plain_string_is_accepted(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["publishers"] = "Prentice Hall"
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["publisher"] == "Prentice Hall"

    def test_subjects_as_objects_are_flattened(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["subjects"] = [
            {"name": "Fiction"},
            "Computers",
            {"name": ""},
        ]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["categories"] == ["Fiction", "Computers"]

    def test_language_as_plain_string_is_accepted(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload["languages"] = ["fre"]
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["language"] == "fre"

    def test_missing_language_returns_none(
        self,
        mock_open_library_http,
        open_library_isbn_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        payload = copy.deepcopy(open_library_isbn_response)
        payload.pop("languages", None)
        mock_open_library_http.add_response(path="/isbn/", json=payload)

        candidate = single_candidate(
            run_async(provider.search_by_isbn(ISBN_13))
        )

        assert candidate["language"] is None

    def test_search_doc_without_optional_fields_is_safe(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json",
            json={"numFound": 1, "docs": [{"title": "Minimal Doc"}]},
        )

        candidate = single_candidate(
            run_async(provider.search_by_title_author("Minimal Doc"))
        )

        assert candidate["title"] == "Minimal Doc"
        assert candidate["authors"] == []
        assert candidate["publisher"] is None
        assert candidate["published_date"] is None
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["categories"] == []
        assert candidate["language"] is None
        assert candidate["cover_url"] is None
        assert candidate["source_id"] is None

    def test_search_doc_with_non_list_author_names_is_safe(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json",
            json={
                "numFound": 1,
                "docs": [{"title": "Weird Doc", "author_name": "Jane Austen"}],
            },
        )

        candidate = single_candidate(
            run_async(provider.search_by_title_author("Weird Doc"))
        )

        assert candidate["authors"] == []

    def test_search_doc_isbn_list_with_only_isbn_10(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json",
            json={
                "numFound": 1,
                "docs": [{"title": "Only Ten", "isbn": [ISBN_10]}],
            },
        )

        candidate = single_candidate(
            run_async(provider.search_by_title_author("Only Ten"))
        )

        assert candidate["isbn_10"] == ISBN_10
        assert candidate["isbn_13"] is None

    def test_search_doc_isbn_list_with_only_isbn_13(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json",
            json={
                "numFound": 1,
                "docs": [{"title": "Only Thirteen", "isbn": [ISBN_13]}],
            },
        )

        candidate = single_candidate(
            run_async(provider.search_by_title_author("Only Thirteen"))
        )

        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] == ISBN_13


# ===========================================================================
# 7. Empty results and 404
# ===========================================================================


class TestEmptyResults:
    """Empty responses are successful lookups with zero candidates."""

    def test_search_empty_docs_returns_success_with_empty_candidates(
        self,
        mock_open_library_http,
        open_library_empty_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", json=open_library_empty_response
        )

        result = run_async(provider.search_by_title_author("No Such Book"))

        assert result.success is True
        assert result.error is None
        assert result.data == {"candidates": []}

    def test_search_response_without_docs_key_returns_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", json={"numFound": 0, "start": 0}
        )

        result = run_async(provider.search_by_title_author("Nothing"))

        assert result.success is True
        assert candidates_of(result) == []

    def test_empty_search_body_returns_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(path="/search.json", json={})

        result = run_async(provider.search_by_title_author("Nothing"))

        assert result.success is True
        assert candidates_of(result) == []

    def test_isbn_404_returns_success_with_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=404, text="Not Found"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert result.error is None
        assert result.data == {"candidates": []}
        # A 404 is a definitive "not found" – no retries.
        assert mock_open_library_http.request_count == 1

    def test_search_404_returns_success_with_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", status_code=404, text="Not Found"
        )

        result = run_async(provider.search_by_title_author("Missing"))

        assert result.success is True
        assert result.error is None
        assert candidates_of(result) == []
        assert mock_open_library_http.request_count == 1


# ===========================================================================
# 8. HTTP failures, retries and malformed payloads
# ===========================================================================


class TestErrorHandling:
    """429, 5xx, timeouts, bad requests and malformed JSON."""

    def test_rate_limit_exhausts_retries_and_fails(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=429, text="Too Many Requests"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Rate limit exceeded"
        assert result.data is None
        assert mock_open_library_http.request_count == provider.max_retries + 1

    def test_rate_limit_recovers_on_retry(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=429, text="Too Many Requests", times=1
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["isbn_13"] == ISBN_13
        # One 429 followed by one successful retry; the author lookup is a
        # separate request and is therefore excluded here.
        assert len(endpoint_requests(mock_open_library_http, "/isbn/")) == 2

    def test_rate_limit_on_search_exhausts_retries(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", status_code=429, text="Too Many Requests"
        )

        result = run_async(provider.search_by_title_author("Clean Code"))

        assert result.success is False
        assert result.error == "Rate limit exceeded"
        assert mock_open_library_http.request_count == provider.max_retries + 1

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_5xx_errors_are_retried_then_fail(
        self,
        status_code: int,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=status_code, text="server error"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == f"Server error: {status_code}"
        assert result.data is None
        assert mock_open_library_http.request_count == provider.max_retries + 1

    def test_server_error_recovers_on_retry(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=503, text="unavailable", times=1
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["title"] == "Clean Code"
        assert len(endpoint_requests(mock_open_library_http, "/isbn/")) == 2

    def test_timeout_exhausts_retries_and_fails(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.TimeoutException("request timed out"), path="/isbn/"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert "Request failed after retries" in result.error
        assert "timed out" in result.error
        assert mock_open_library_http.request_count == provider.max_retries + 1

    def test_timeout_recovers_on_retry(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.TimeoutException("request timed out"),
            path="/isbn/",
            times=1,
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["isbn_13"] == ISBN_13
        assert len(endpoint_requests(mock_open_library_http, "/isbn/")) == 2

    def test_connection_error_is_retried_then_fails(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.ConnectError("connection refused"), path="/isbn/"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert "Request failed after retries" in (result.error or "")
        assert "connection refused" in (result.error or "")
        assert mock_open_library_http.request_count == provider.max_retries + 1

    def test_bad_request_fails_without_retry(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=400, text="invalid isbn"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Bad request: invalid isbn"
        assert result.data is None
        assert mock_open_library_http.request_count == 1

    def test_isbn_malformed_json_returns_failure(
        self,
        mock_open_library_http,
        open_library_malformed_response: str,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/",
            status_code=200,
            text=open_library_malformed_response,
            headers={"content-type": "text/html"},
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.startswith("Invalid JSON response")
        # A malformed body is not retried.
        assert mock_open_library_http.request_count == 1

    def test_search_malformed_json_returns_failure(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/search.json", status_code=200, text='{"numFound": 1, "docs": ['
        )

        result = run_async(provider.search_by_title_author("Clean Code"))

        assert result.success is False
        assert "Invalid JSON response" in (result.error or "")

    def test_details_malformed_json_returns_failure(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/books/OL00000M", status_code=200, text="}{ not json"
        )

        result = run_async(provider.get_book_details("OL00000M"))

        assert result.success is False
        assert "Invalid JSON response" in (result.error or "")

    def test_zero_retries_configuration_is_respected(
        self,
        mock_open_library_http,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_MAX_RETRIES", 0)
        provider = OpenLibraryProvider()
        assert provider.max_retries == 0
        mock_open_library_http.add_response(
            path="/isbn/", status_code=500, text="boom"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Server error: 500"
        assert mock_open_library_http.request_count == 1


# ===========================================================================
# 9. Edition / work details
# ===========================================================================


class TestGetBookDetails:
    """``get_book_details`` dispatches editions and works to the right URL."""

    def test_edition_id_uses_books_endpoint(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/books/{EDITION_ID}", json=open_library_edition_response
        )

        result = run_async(provider.get_book_details(EDITION_ID))

        request = mock_open_library_http.requests[0]
        assert request.url.path == f"/books/{EDITION_ID}.json"

        candidate = single_candidate(result)
        assert candidate["title"] == "Clean Code"
        assert candidate["source"] == "open_library"
        assert candidate["source_id"] == EDITION_ID

    def test_edition_details_resolve_authors_and_cover(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/books/{EDITION_ID}", json=open_library_edition_response
        )

        candidate = single_candidate(
            run_async(provider.get_book_details(EDITION_ID))
        )

        assert candidate["authors"] == [AUTHOR_NAME]
        assert candidate["cover_url"] == COVER_URL
        assert candidate["isbn_13"] == ISBN_13
        assert mock_open_library_http.request_count == 2

    def test_work_id_uses_works_endpoint(
        self,
        mock_open_library_http,
        open_library_work_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/works/{WORK_ID}", json=open_library_work_response
        )

        result = run_async(provider.get_book_details(WORK_ID))

        request = mock_open_library_http.get_request()
        assert request.url.path == f"/works/{WORK_ID}.json"
        assert result.success is True

    def test_work_details_are_parsed_without_edition_fields(
        self,
        mock_open_library_http,
        open_library_work_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/works/{WORK_ID}", json=open_library_work_response
        )

        candidate = single_candidate(run_async(provider.get_book_details(WORK_ID)))

        assert candidate["title"] == "Clean Code"
        # Work payloads omit edition-only fields.
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["page_count"] is None
        assert candidate["publisher"] is None
        # ``first_publish_date`` is not an edition ``publish_date``.
        assert candidate["published_date"] is None
        assert candidate["cover_url"] == COVER_URL
        assert candidate["source_id"] == WORK_ID

    def test_work_details_do_not_request_author_endpoint(
        self,
        mock_open_library_http,
        open_library_work_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        # Work author refs are nested under an ``author`` key that the
        # provider does not resolve, so no extra HTTP calls are made.
        mock_open_library_http.add_response(
            path=f"/works/{WORK_ID}", json=open_library_work_response
        )

        result = run_async(provider.get_book_details(WORK_ID))

        assert single_candidate(result)["authors"] == []
        assert mock_open_library_http.request_count == 1

    def test_full_edition_path_is_supported(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/books/{EDITION_ID}", json=open_library_edition_response
        )

        result = run_async(provider.get_book_details(f"/books/{EDITION_ID}"))

        assert mock_open_library_http.requests[0].url.path == (
            f"/books/{EDITION_ID}.json"
        )
        assert single_candidate(result)["source_id"] == EDITION_ID

    def test_full_work_path_is_supported(
        self,
        mock_open_library_http,
        open_library_work_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/works/{WORK_ID}", json=open_library_work_response
        )

        result = run_async(provider.get_book_details(f"/works/{WORK_ID}"))

        assert mock_open_library_http.get_request().url.path == (
            f"/works/{WORK_ID}.json"
        )
        assert single_candidate(result)["source_id"] == WORK_ID

    def test_unrecognised_id_defaults_to_edition_endpoint(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/books/OL12345X", json=open_library_edition_response
        )

        result = run_async(provider.get_book_details("OL12345X"))

        assert mock_open_library_http.requests[0].url.path == (
            "/books/OL12345X.json"
        )
        assert result.success is True

    def test_full_path_without_edition_or_work_marker_is_passed_through(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        # "books/OL12345X" contains neither an 'M' nor a 'W' marker, so the
        # provider falls back to using the supplied path verbatim.
        mock_open_library_http.add_response(
            path="/books/OL12345X", json=open_library_edition_response
        )

        result = run_async(provider.get_book_details("/books/OL12345X"))

        assert mock_open_library_http.requests[0].url.path == (
            "/books/OL12345X.json"
        )
        assert result.success is True

    def test_details_404_returns_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path="/books/OL00000M", status_code=404, text="Not Found"
        )

        result = run_async(provider.get_book_details("OL00000M"))

        assert result.success is True
        assert result.error is None
        assert candidates_of(result) == []

    def test_details_empty_payload_returns_empty_candidates(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(path="/books/OL00000M", json={})

        result = run_async(provider.get_book_details("OL00000M"))

        assert result.success is True
        assert candidates_of(result) == []

    def test_details_rate_limit_recovers_on_retry(
        self,
        mock_open_library_http,
        open_library_edition_response: dict[str, Any],
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/books/{EDITION_ID}", status_code=429, text="slow down", times=1
        )
        mock_open_library_http.add_response(
            path=f"/books/{EDITION_ID}", json=open_library_edition_response
        )

        result = run_async(provider.get_book_details(EDITION_ID))

        assert result.success is True
        assert single_candidate(result)["source_id"] == EDITION_ID
        assert (
            len(endpoint_requests(mock_open_library_http, f"/books/{EDITION_ID}"))
            == 2
        )

    def test_details_5xx_exhausts_retries(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_response(
            path=f"/works/{WORK_ID}", status_code=502, text="bad gateway"
        )

        result = run_async(provider.get_book_details(WORK_ID))

        assert result.success is False
        assert result.error == "Server error: 502"
        assert mock_open_library_http.request_count == provider.max_retries + 1

    def test_details_timeout_fails_after_retries(
        self,
        mock_open_library_http,
        provider: OpenLibraryProvider,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.TimeoutException("request timed out"), path=f"/books/{EDITION_ID}"
        )

        result = run_async(provider.get_book_details(EDITION_ID))

        assert result.success is False
        assert "Request failed after retries" in (result.error or "")
        assert mock_open_library_http.request_count == provider.max_retries + 1


# ===========================================================================
# 10. Cover URL construction
# ===========================================================================


class TestCoverUrlConstruction:
    """``_build_cover_url`` builds canonical Open Library cover URLs."""

    def test_default_size_is_large(self, provider: OpenLibraryProvider) -> None:
        assert provider._build_cover_url(COVER_ID) == COVER_URL
        assert provider._build_cover_url(COVER_ID) == (
            "https://covers.openlibrary.org/b/id/1234567-L.jpg"
        )

    @pytest.mark.parametrize("size", ["S", "M", "L"])
    def test_requested_size_is_used(
        self, size: str, provider: OpenLibraryProvider
    ) -> None:
        assert provider._build_cover_url(COVER_ID, size=size) == (
            f"{COVER_BASE_URL}/{COVER_ID}-{size}.jpg"
        )

    @pytest.mark.parametrize("cover_id", [None, 0])
    def test_missing_cover_id_returns_none(
        self, cover_id: Any, provider: OpenLibraryProvider
    ) -> None:
        assert provider._build_cover_url(cover_id) is None

    def test_cover_url_is_https(self, provider: OpenLibraryProvider) -> None:
        url = provider._build_cover_url(COVER_ID)
        assert url is not None
        assert url.startswith("https://")
        assert COVER_BASE_URL in url


# ===========================================================================
# 11. Field extraction helpers
# ===========================================================================


class TestFieldExtractionHelpers:
    """Small pure helpers used by the parsing pipeline."""

    def test_extract_edition_id_from_key(self, provider: OpenLibraryProvider) -> None:
        assert provider._extract_edition_id({"key": "/books/OL123456M"}) == EDITION_ID
        assert provider._extract_edition_id({"key": "/books/OL123456M/"}) == EDITION_ID

    @pytest.mark.parametrize("data", [{}, {"key": ""}])
    def test_extract_edition_id_missing_key_returns_none(
        self, data: dict[str, Any], provider: OpenLibraryProvider
    ) -> None:
        assert provider._extract_edition_id(data) is None

    def test_extract_work_id_from_works_array(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider._extract_work_id({"works": [{"key": f"/works/{WORK_ID}"}]}) == (
            WORK_ID
        )

    @pytest.mark.parametrize(
        "data",
        [{}, {"works": []}, {"works": [{}]}, {"works": ["not-a-dict"]}],
    )
    def test_extract_work_id_missing_returns_none(
        self, data: dict[str, Any], provider: OpenLibraryProvider
    ) -> None:
        # Empty/malformed ``works`` entries never produce a usable work ID.
        assert not provider._extract_work_id(data)

    def test_extract_isbns_uses_dedicated_fields(
        self, provider: OpenLibraryProvider
    ) -> None:
        data = {"isbn_10": [ISBN_10], "isbn_13": [ISBN_13]}
        assert provider._extract_isbns(data) == (ISBN_10, ISBN_13)

    def test_extract_isbns_uses_first_of_each_list(
        self, provider: OpenLibraryProvider
    ) -> None:
        data = {
            "isbn_10": [ISBN_10, "0000000000"],
            "isbn_13": [ISBN_13, "9780000000002"],
        }
        assert provider._extract_isbns(data) == (ISBN_10, ISBN_13)

    def test_extract_isbns_falls_back_to_generic_list(
        self, provider: OpenLibraryProvider
    ) -> None:
        data = {"isbn": [ISBN_13, ISBN_10, "9780000000002"]}
        assert provider._extract_isbns(data) == (ISBN_10, ISBN_13)

    def test_extract_isbns_handles_plain_strings(
        self, provider: OpenLibraryProvider
    ) -> None:
        data = {"isbn_10": ISBN_10, "isbn_13": ISBN_13}
        assert provider._extract_isbns(data) == (ISBN_10, ISBN_13)

    def test_extract_isbns_missing_returns_none_pair(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider._extract_isbns({}) == (None, None)
        assert provider._extract_isbns({"isbn": []}) == (None, None)

    def test_parse_language_from_key_object(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider._parse_language([{"key": "/languages/eng"}]) == "eng"

    def test_parse_language_from_plain_string(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider._parse_language(["eng"]) == "eng"

    def test_parse_language_empty_returns_none(
        self, provider: OpenLibraryProvider
    ) -> None:
        assert provider._parse_language([]) is None
        assert provider._parse_language(None) is None
        # Malformed entries (neither dict nor string) are ignored.
        assert provider._parse_language([42]) is None

    @pytest.mark.parametrize(
        "subjects, expected",
        [
            ([], []),
            (None, []),
            ("not-a-list", []),
            (["Fiction"], ["Fiction"]),
            ([{"name": "Fiction"}], ["Fiction"]),
            ([{"name": "Fiction"}, "Computers", {"name": ""}], ["Fiction", "Computers"]),
            ([{"no_name": True}, 42], []),
        ],
    )
    def test_extract_subjects(
        self,
        subjects: Any,
        expected: list[str],
        provider: OpenLibraryProvider,
    ) -> None:
        assert provider._extract_subjects(subjects) == expected


# ===========================================================================
# 12. Orchestration resilience
# ===========================================================================


class TestOrchestrationResilience:
    """Provider failures are contained by the metadata orchestrator."""

    @pytest.fixture
    def open_library_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force the retriever to query only Open Library."""
        monkeypatch.setattr(settings, "METADATA_STRATEGY", "OPEN_LIBRARY_ONLY")

    def test_timeout_does_not_crash_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        mock_open_library_http.add_exception(
            httpx.TimeoutException("request timed out"), path="/isbn/"
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates == []
        assert response.selected_candidate is None
        assert response.query.isbn == ISBN_13

    def test_rate_limit_failure_does_not_crash_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=429, text="Too Many Requests"
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates == []

    def test_404_returns_empty_candidates_via_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        mock_open_library_http.add_response(
            path="/isbn/", status_code=404, text="Not Found"
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates == []

    def test_successful_isbn_lookup_reaches_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates
        assert all(
            candidate.source == "open_library" for candidate in response.candidates
        )
        assert any(
            candidate.isbn_13 == ISBN_13 for candidate in response.candidates
        )
        # The exact ISBN match should be decisive for a single candidate.
        assert response.selected_candidate is not None
        assert response.selected_candidate.isbn_13 == ISBN_13

    def test_title_search_flows_through_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        retriever = metadata_retriever_factory()

        response = run_async(
            retriever.retrieve(title="Clean Code", author=AUTHOR_NAME)
        )

        assert response.status == "success"
        assert response.candidates
        assert all(c.source == "open_library" for c in response.candidates)
        assert all(
            candidate.confidence >= 0.0 and candidate.confidence <= 1.0
            for candidate in response.candidates
        )

    def test_unresolvable_authors_still_reach_orchestration(
        self,
        open_library_only,
        metadata_retriever_factory,
        mock_open_library_http,
    ) -> None:
        mock_open_library_http.add_response(
            path="/authors/", status_code=404, text="Not Found"
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates
        assert response.candidates[0].authors == []
