"""Unit tests for the Google Books metadata provider.

The provider talks to the Google Books REST API through
:class:`httpx.AsyncClient`.  Every request in this module is served by the
``MockHTTPRouter`` fixtures declared in ``conftest.py``; the autouse network
guard there fails the test if a live call ever slips through.

Tests are intentionally synchronous: the project does not depend on
``pytest-asyncio``, so coroutines are driven with :func:`asyncio.run`.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Optional

import httpx
import pytest

from app.books.metadata.providers.base import ProviderResult
from app.books.metadata.providers.google_books import GoogleBooksProvider
from app.core.config import settings

# ``VALID_ISBN_10`` / ``VALID_ISBN_13`` are provided by the conftest fixtures
# ``valid_isbn_10`` / ``valid_isbn``; these literals are used inside
# parametrisations where fixtures cannot be injected.
ISBN_13 = "9780132350884"
ISBN_10 = "0132350882"
VOLUME_ID = "zyTCAlFPjgYC"


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


@pytest.fixture
def provider() -> GoogleBooksProvider:
    """A fresh provider instance (its ``_build_client`` is patched by mocks)."""
    return GoogleBooksProvider()


# ===========================================================================
# 1. Search by ISBN
# ===========================================================================


class TestSearchByIsbn:
    """``GoogleBooksProvider.search_by_isbn`` behaviour."""

    def test_search_by_valid_isbn_13_returns_matching_book(
        self,
        mock_google_books_http,
        google_books_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["title"] == "Clean Code"
        assert candidate["subtitle"] == "A Handbook of Agile Software Craftsmanship"
        assert candidate["authors"] == ["Robert C. Martin"]
        assert candidate["publisher"] == "Prentice Hall"
        assert candidate["published_date"] == "2008-08-01"
        assert candidate["isbn_10"] == ISBN_10
        assert candidate["isbn_13"] == ISBN_13
        assert candidate["page_count"] == 464
        assert candidate["language"] == "en"
        assert candidate["source"] == "google_books"
        assert candidate["source_id"] == VOLUME_ID
        assert candidate["confidence"] == 0.0
        assert result.raw_response == google_books_response

    def test_search_by_valid_isbn_10_returns_matching_book(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        result = run_async(provider.search_by_isbn(ISBN_10))

        candidate = single_candidate(result)
        assert candidate["isbn_10"] == ISBN_10
        assert candidate["title"] == "Clean Code"

    def test_search_by_isbn_sends_expected_query(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        run_async(provider.search_by_isbn(ISBN_13))

        request = mock_google_books_http.get_request()
        assert request.method == "GET"
        assert request.url.path == "/books/v1/volumes"
        assert request.url.params["q"] == f"isbn:{ISBN_13}"
        assert "key" not in request.url.params

    def test_search_by_isbn_includes_api_key_when_configured(
        self,
        mock_google_books_http,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "GOOGLE_BOOKS_API_KEY", "test-api-key")
        provider = GoogleBooksProvider()
        assert provider.api_key == "test-api-key"

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        request = mock_google_books_http.get_request()
        assert request.url.params["key"] == "test-api-key"

    def test_search_by_isbn_returns_all_matching_volumes(
        self,
        mock_google_books_http,
        google_books_multi_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_multi_response
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        candidates = candidates_of(result)
        assert len(candidates) == 3
        assert [c["source_id"] for c in candidates] == [
            VOLUME_ID,
            "paperback-clean-code",
            "ebook-clean-code",
        ]


# ===========================================================================
# 2. Search by title / author
# ===========================================================================


class TestSearchByTitleAndAuthor:
    """``GoogleBooksProvider.search_by_title_author`` behaviour."""

    def test_search_by_title_returns_matching_candidates(
        self,
        mock_google_books_http,
        google_books_title_search_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_title_search_response
        )

        result = run_async(provider.search_by_title_author("Clean Code"))

        candidates = candidates_of(result)
        assert len(candidates) == 3
        assert {c["title"] for c in candidates} == {"Clean Code"}
        assert all(c["source"] == "google_books" for c in candidates)
        assert [c["isbn_13"] for c in candidates] == [
            ISBN_13,
            "9780137081073",
            "9780135957059",
        ]

    def test_search_by_title_sends_intitle_query(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code"))

        request = mock_google_books_http.get_request()
        assert request.url.params["q"] == 'intitle:"Clean Code"'

    def test_search_by_author_returns_matching_candidates(
        self,
        mock_google_books_http,
        google_books_title_search_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_title_search_response
        )

        result = run_async(
            provider.search_by_title_author("Clean Code", "Robert C. Martin")
        )

        candidates = candidates_of(result)
        assert candidates
        assert all(
            "Robert C. Martin" in candidate["authors"] for candidate in candidates
        )

    def test_search_by_title_and_author_includes_both_filters(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code", "Robert C. Martin"))

        request = mock_google_books_http.get_request()
        assert (
            request.url.params["q"]
            == 'intitle:"Clean Code" inauthor:"Robert C. Martin"'
        )

    def test_search_by_title_without_author_omits_inauthor(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Clean Code", author=None))

        request = mock_google_books_http.get_request()
        assert "inauthor" not in request.url.params["q"]

    def test_search_by_title_with_unicode_preserves_query(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        run_async(provider.search_by_title_author("Café Society", "José Saramago"))

        request = mock_google_books_http.get_request()
        assert (
            request.url.params["q"]
            == 'intitle:"Café Society" inauthor:"José Saramago"'
        )


# ===========================================================================
# 3. Empty results
# ===========================================================================


class TestEmptyResults:
    """Empty provider responses are successful, not errors."""

    def test_isbn_search_empty_results_returns_success_with_empty_candidates(
        self,
        mock_google_books_http,
        google_books_empty_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_empty_response
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert result.error is None
        assert result.data == {"candidates": []}

    def test_title_search_empty_results_returns_success_with_empty_candidates(
        self,
        mock_google_books_http,
        google_books_empty_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_empty_response
        )

        result = run_async(provider.search_by_title_author("No Such Book"))

        assert result.success is True
        assert result.error is None
        assert result.data == {"candidates": []}

    def test_author_search_empty_results_returns_success_with_empty_candidates(
        self,
        mock_google_books_http,
        google_books_empty_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_empty_response
        )

        result = run_async(
            provider.search_by_title_author("No Such Book", "No Such Author")
        )

        assert result.success is True
        assert candidates_of(result) == []

    def test_search_404_returns_success_with_empty_candidates(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=404, text="Not Found"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert result.error is None
        assert candidates_of(result) == []

    def test_search_response_without_items_key_returns_empty_list(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        # The real API omits ``items`` entirely for zero results.
        mock_google_books_http.add_response(
            url="/volumes", json={"kind": "books#volumes", "totalItems": 0}
        )

        result = run_async(provider.search_by_title_author("Nothing Here"))

        assert result.success is True
        assert candidates_of(result) == []


# ===========================================================================
# 4. Missing / partial fields
# ===========================================================================


class TestMissingFields:
    """Incomplete volumes must still produce usable candidates."""

    def test_missing_authors_returns_empty_authors_list(
        self,
        mock_google_books_http,
        google_books_missing_fields_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_missing_fields_response
        )

        result = run_async(provider.search_by_isbn("9780000000002"))

        candidate = single_candidate(result)
        assert candidate["authors"] == []
        assert candidate["title"] == "An Anonymous Work"

    def test_missing_isbn_returns_metadata_without_isbn(
        self,
        mock_google_books_http,
        google_books_missing_fields_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_missing_fields_response
        )

        result = run_async(provider.search_by_isbn("9780000000002"))

        candidate = single_candidate(result)
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["title"] == "An Anonymous Work"
        assert candidate["source_id"] == "no-isbn-no-authors"

    def test_missing_isbn_normalizes_to_none(
        self,
        mock_google_books_http,
        google_books_missing_fields_response: dict[str, Any],
        normalizer,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_missing_fields_response
        )

        result = run_async(provider.search_by_isbn("9780000000002"))
        metadata = normalizer.normalize_candidate(single_candidate(result))

        assert metadata.title == "An Anonymous Work"
        assert metadata.authors == []
        assert metadata.isbn_10 is None
        assert metadata.isbn_13 is None
        assert metadata.cover_url is None
        assert metadata.source == "google_books"

    def test_missing_volume_id_does_not_crash(
        self,
        mock_google_books_http,
        google_books_volume: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        volume = copy.deepcopy(google_books_volume)
        volume.pop("id", None)
        payload = {"kind": "books#volumes", "totalItems": 1, "items": [volume]}
        mock_google_books_http.add_response(url="/volumes", json=payload)

        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["source_id"] is None
        assert candidate["title"] == "Clean Code"
        assert candidate["source"] == "google_books"

    def test_missing_volume_info_does_not_crash(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        payload = {
            "kind": "books#volumes",
            "totalItems": 1,
            "items": [{"kind": "books#volume", "id": "volume-without-info"}],
        }
        mock_google_books_http.add_response(url="/volumes", json=payload)

        result = run_async(provider.search_by_isbn(ISBN_13))

        candidate = single_candidate(result)
        assert candidate["title"] is None
        assert candidate["authors"] == []
        assert candidate["categories"] == []
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["cover_url"] is None
        assert candidate["source_id"] == "volume-without-info"

    def test_missing_optional_fields_default_cleanly(
        self,
        mock_google_books_http,
        google_books_missing_fields_response: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_missing_fields_response
        )

        candidate = single_candidate(
            run_async(provider.search_by_isbn("9780000000002"))
        )

        assert candidate["subtitle"] is None
        assert candidate["publisher"] is None
        assert candidate["published_date"] is None
        assert candidate["description"] is None
        assert candidate["page_count"] is None
        assert candidate["categories"] == []
        assert candidate["language"] is None
        assert candidate["cover_url"] is None


# ===========================================================================
# 5. Error handling and retries
# ===========================================================================


class TestErrorHandling:
    """HTTP failures, timeouts and malformed payloads."""

    def test_rate_limit_exhausts_retries_and_fails(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=429, text="rate limited"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Rate limit exceeded"
        assert result.data is None
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_rate_limit_recovers_on_retry(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=429, text="rate limited", times=1
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["title"] == "Clean Code"
        # First attempt hit 429, second attempt used the fallback 200.
        assert mock_google_books_http.request_count == 2

    def test_server_error_exhausts_retries_and_fails(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=500, text="boom"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Server error: 500"
        assert result.data is None
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_server_error_recovers_on_retry(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=500, text="boom", times=1
        )

        result = run_async(provider.search_by_title_author("Clean Code"))

        assert result.success is True
        assert candidates_of(result)
        assert mock_google_books_http.request_count == 2

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_5xx_family_is_retried(
        self,
        status_code: int,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=status_code, text="server error"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == f"Server error: {status_code}"
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_timeout_exhausts_retries_and_fails(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_exception(
            httpx.TimeoutException("request timed out"), url="/volumes"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert "Request failed after retries" in result.error
        assert "timed out" in result.error
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_timeout_recovers_on_retry(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_exception(
            httpx.TimeoutException("request timed out"),
            url="/volumes",
            times=1,
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is True
        assert single_candidate(result)["isbn_13"] == ISBN_13
        assert mock_google_books_http.request_count == 2

    def test_connection_error_is_retried_then_fails(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_exception(
            httpx.ConnectError("connection refused"), url="/volumes"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert "Request failed after retries" in (result.error or "")
        assert "connection refused" in (result.error or "")
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_invalid_json_returns_failure(
        self,
        mock_google_books_http,
        google_books_malformed_response: str,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=200, text=google_books_malformed_response
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.data is None
        assert result.error is not None
        assert result.error.startswith("Invalid JSON response")
        # A malformed body is not retried.
        assert mock_google_books_http.request_count == 1

    def test_invalid_json_on_title_search_returns_failure(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes",
            status_code=200,
            text="<html><body>not json</body></html>",
            headers={"content-type": "text/html"},
        )

        result = run_async(provider.search_by_title_author("Clean Code"))

        assert result.success is False
        assert "Invalid JSON response" in (result.error or "")

    def test_bad_request_returns_failure_without_retry(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes", status_code=400, text="invalid query"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Bad request: invalid query"
        assert mock_google_books_http.request_count == 1

    def test_zero_retries_configuration_is_respected(
        self,
        mock_google_books_http,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_MAX_RETRIES", 0)
        provider = GoogleBooksProvider()
        assert provider.max_retries == 0
        mock_google_books_http.add_response(
            url="/volumes", status_code=500, text="boom"
        )

        result = run_async(provider.search_by_isbn(ISBN_13))

        assert result.success is False
        assert result.error == "Server error: 500"
        assert mock_google_books_http.request_count == 1


# ===========================================================================
# 6. Volume lookup by provider id
# ===========================================================================


class TestGetBookDetails:
    """``GoogleBooksProvider.get_book_details`` behaviour."""

    def test_get_book_details_returns_candidate(
        self,
        mock_google_books_http,
        google_books_volume: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", json=google_books_volume
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        candidate = single_candidate(result)
        assert candidate["title"] == "Clean Code"
        assert candidate["source_id"] == VOLUME_ID
        assert candidate["source"] == "google_books"
        assert result.raw_response == google_books_volume

        request = mock_google_books_http.get_request()
        assert request.url.path.endswith(f"/volumes/{VOLUME_ID}")

    def test_get_book_details_404_returns_empty_candidates(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes/missing-volume", status_code=404, text="Not Found"
        )

        result = run_async(provider.get_book_details("missing-volume"))

        assert result.success is True
        assert result.error is None
        assert candidates_of(result) == []

    def test_get_book_details_invalid_json_returns_failure(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url="/volumes/broken-volume", status_code=200, text="}{not json"
        )

        result = run_async(provider.get_book_details("broken-volume"))

        assert result.success is False
        assert "Invalid JSON response" in (result.error or "")

    def test_get_book_details_rate_limit_is_retried(
        self,
        mock_google_books_http,
        google_books_volume: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", status_code=429, text="slow down", times=1
        )
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", json=google_books_volume
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is True
        assert single_candidate(result)["source_id"] == VOLUME_ID
        assert mock_google_books_http.request_count == 2

    def test_get_book_details_timeout_fails_after_retries(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_exception(
            httpx.TimeoutException("request timed out"),
            url=f"/volumes/{VOLUME_ID}",
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is False
        assert "Request failed after retries" in (result.error or "")
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_get_book_details_connection_error_fails_after_retries(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_exception(
            httpx.ConnectError("connection refused"),
            url=f"/volumes/{VOLUME_ID}",
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is False
        assert "Request failed after retries" in (result.error or "")
        assert "connection refused" in (result.error or "")
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_get_book_details_rate_limit_exhausts_retries(
        self,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", status_code=429, text="slow down"
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is False
        assert result.error == "Rate limit exceeded"
        assert mock_google_books_http.request_count == provider.max_retries + 1

    @pytest.mark.parametrize("status_code", [500, 502, 503])
    def test_get_book_details_server_error_exhausts_retries(
        self,
        status_code: int,
        mock_google_books_http,
        provider: GoogleBooksProvider,
    ) -> None:
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", status_code=status_code, text="server error"
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is False
        assert result.error == f"Server error: {status_code}"
        assert mock_google_books_http.request_count == provider.max_retries + 1

    def test_get_book_details_includes_api_key_when_configured(
        self,
        mock_google_books_http,
        google_books_volume: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "GOOGLE_BOOKS_API_KEY", "test-api-key")
        provider = GoogleBooksProvider()
        mock_google_books_http.add_response(
            url=f"/volumes/{VOLUME_ID}", json=google_books_volume
        )

        result = run_async(provider.get_book_details(VOLUME_ID))

        assert result.success is True
        request = mock_google_books_http.get_request()
        assert request.url.params["key"] == "test-api-key"


# ===========================================================================
# 7. Candidate extraction / normalisation
# ===========================================================================


class TestCandidateExtraction:
    """``_extract_volume_info`` and interaction with the normalizer."""

    def test_provider_name_is_google_books(self, provider: GoogleBooksProvider) -> None:
        assert provider.name == "google_books"

    def test_extract_full_volume_matches_expected_candidate(
        self,
        google_books_volume: dict[str, Any],
        google_books_candidate: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        extracted = provider._extract_volume_info(google_books_volume)

        assert extracted == google_books_candidate
        assert extracted["description"].startswith("Even bad code can function")
        assert extracted["categories"] == ["Computers", "Software Engineering"]
        assert extracted["confidence"] == 0.0

    def test_extract_prefers_largest_available_cover_image(
        self,
        google_books_volume: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        volume = copy.deepcopy(google_books_volume)
        volume["volumeInfo"]["imageLinks"] = {
            "smallThumbnail": "https://example.test/small-thumb.jpg",
            "thumbnail": "https://example.test/thumb.jpg",
            "small": "https://example.test/small.jpg",
            "medium": "https://example.test/medium.jpg",
            "large": "https://example.test/large.jpg",
            "extraLarge": "https://example.test/extra-large.jpg",
        }

        candidate = provider._extract_volume_info(volume)

        assert candidate["cover_url"] == "https://example.test/extra-large.jpg"

    def test_extract_falls_back_to_thumbnail(
        self,
        google_books_volume: dict[str, Any],
        provider: GoogleBooksProvider,
    ) -> None:
        volume = copy.deepcopy(google_books_volume)
        volume["volumeInfo"]["imageLinks"] = {
            "smallThumbnail": "https://example.test/small-thumb.jpg",
            "thumbnail": "https://example.test/thumb.jpg",
        }

        candidate = provider._extract_volume_info(volume)

        assert candidate["cover_url"] == "https://example.test/thumb.jpg"

    def test_extract_handles_empty_volume(self, provider: GoogleBooksProvider) -> None:
        candidate = provider._extract_volume_info({})

        assert candidate["title"] is None
        assert candidate["authors"] == []
        assert candidate["categories"] == []
        assert candidate["isbn_10"] is None
        assert candidate["isbn_13"] is None
        assert candidate["cover_url"] is None
        assert candidate["source_id"] is None
        assert candidate["source"] == "google_books"

    def test_normalizer_accepts_extracted_candidate(
        self,
        google_books_candidate: dict[str, Any],
        normalizer,
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)

        assert metadata.title == "Clean Code"
        assert metadata.subtitle == "A Handbook of Agile Software Craftsmanship"
        assert metadata.authors == ["Robert C. Martin"]
        assert metadata.publisher == "Prentice Hall"
        assert metadata.published_date == "2008-08-01"
        assert metadata.isbn_10 == ISBN_10
        assert metadata.isbn_13 == ISBN_13
        assert metadata.page_count == 464
        assert metadata.categories == ["Computers", "Software Engineering"]
        assert metadata.language == "en"
        assert metadata.source == "google_books"
        assert metadata.source_id == VOLUME_ID
        assert metadata.confidence == 0.0

    def test_search_results_are_normalizable_end_to_end(
        self,
        mock_google_books_http,
        normalizer,
        provider: GoogleBooksProvider,
    ) -> None:
        result = run_async(provider.search_by_isbn(ISBN_13))

        metadata = normalizer.normalize_candidate(single_candidate(result))

        assert metadata.isbn_13 == ISBN_13
        assert metadata.source_id == VOLUME_ID
        assert metadata.confidence == 0.0
