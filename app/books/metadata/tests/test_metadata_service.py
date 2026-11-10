"""Integration tests for the metadata retrieval orchestrator.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 17) and exercises the *entire* provider orchestration
pipeline exposed by :class:`~app.books.metadata.service.BookMetadataRetriever`
and :func:`~app.books.metadata.service.retrieve_book_metadata`:

==========================================  =================================
Scenario                                    Expected behaviour
==========================================  =================================
Google Books finds exact ISBN               Returns matching candidate
Google Books empty, Open Library succeeds   Returns Open Library result
Both providers find same book               Deduplicates candidates
Providers return complementary fields       Merges compatible metadata
Providers return conflicting ISBNs          Preserves conflict
Both providers return no results            Returns empty candidates
Google Books timeout                        Open Library is still queried
Both APIs fail                              Handles provider failure
Only title supplied                         Performs title search
Only author supplied                        Performs author search
Invalid ISBN                                Rejects request
Scraping disabled                           Does not invoke scraper
APIs insufficient, scraping enabled         Invokes scraper fallback
Multiple ambiguous candidates               Does not automatically select one
==========================================  =================================

All external providers are mocked: the ``conftest.py`` fixtures install
fail-closed HTTP transports on the provider classes, and the autouse
network guard fails the test if a live request ever slips through.  Some
tests additionally inject raw provider candidates directly through the
``ProviderResult`` boundary so that orchestration behaviour can be
verified without re-testing provider parsing.

Tests are synchronous: the project does not depend on ``pytest-asyncio``,
so coroutines are driven with :func:`asyncio.run`, matching the style of
the other Phase 3 test modules.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Optional

import httpx
import pytest
from pydantic import ValidationError

from app.books.exceptions import MetadataServiceUnavailable
from app.books.metadata.schemas import BookMetadata
from app.books.metadata.service import retrieve_book_metadata
from app.core.config import settings

# ---------------------------------------------------------------------------
# ISBN constants (mirroring ``conftest.py``)
# ---------------------------------------------------------------------------

ISBN_13 = "9780132350884"  # Clean Code (Prentice Hall, 2008)
ISBN_10 = "0132350882"

PAPERBACK_ISBN_13 = "9780137081073"
PAPERBACK_ISBN_10 = "0137081071"

CLEAN_CODE_TITLE = "Clean Code"
CLEAN_CODE_AUTHOR = "Robert C. Martin"

#: Invalid ISBNs that are *not* reduced to ``None`` by input normalisation.
INVALID_ISBNS = [
    "123",  # too short
    "1234567890",  # invalid ISBN-10 checksum
    "9780135957058",  # invalid ISBN-13 checksum
    "0321125214",  # invalid ISBN-10 checksum
    "97801359570590",  # too long
    "abcdefghij",  # non-digit ISBN-10
    "97801X5957059",  # 'X' in the wrong position
    "X123456789",  # 'X' in the wrong position
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro: Any) -> Any:
    """Drive *coro* to completion on a fresh event loop."""
    return asyncio.run(coro)


def _dumps(editions: list[BookMetadata]) -> list[dict[str, Any]]:
    """Serialise ``BookMetadata`` objects into raw provider candidate dicts."""
    return [copy.deepcopy(edition.model_dump(exclude_none=False)) for edition in editions]


def _request_urls(mock_metadata_providers: dict[str, Any]) -> list[str]:
    """Return every URL seen by the shared provider mock router."""
    return list(mock_metadata_providers["google_books"].request_urls)


def _requests_to(mock_metadata_providers: dict[str, Any], fragment: str) -> list[Any]:
    """Return recorded requests whose URL contains *fragment*."""
    router = mock_metadata_providers["google_books"]
    return [request for request in router.requests if fragment in str(request.url)]


# ===========================================================================
# 1. Provider orchestration and merging
# ===========================================================================


class TestProviderOrchestration:
    """The full Google Books + Open Library orchestration pipeline."""

    def test_google_books_finds_exact_isbn(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """An exact ISBN hit from Google Books is returned and selected."""
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.query.isbn == ISBN_13
        assert len(response.candidates) == 1

        candidate = response.candidates[0]
        assert candidate.title == CLEAN_CODE_TITLE
        assert candidate.isbn_13 == ISBN_13
        assert candidate.source == "google_books"
        assert candidate.confidence == pytest.approx(1.0)

        assert response.selected_candidate is not None
        assert response.selected_candidate.isbn_13 == ISBN_13

    def test_google_empty_open_library_succeeds(
        self,
        mock_metadata_providers: dict[str, Any],
        google_books_empty_response: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """An empty Google Books result falls through to Open Library."""
        mock_metadata_providers["google_books"].add_response(
            url="/volumes",
            json=google_books_empty_response,
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert len(response.candidates) == 1
        candidate = response.candidates[0]
        assert candidate.source == "open_library"
        assert candidate.title == CLEAN_CODE_TITLE
        assert candidate.isbn_13 == ISBN_13
        assert candidate.confidence == pytest.approx(1.0)
        assert response.selected_candidate is not None

    def test_both_providers_find_same_book_and_deduplicate(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """Two providers returning the same edition yield one candidate."""
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        # Both providers were in fact queried...
        urls = _request_urls(mock_metadata_providers)
        assert any("googleapis" in url for url in urls)
        assert any("openlibrary" in url for url in urls)

        # ...but the duplicate record was collapsed into a single candidate.
        assert len(response.candidates) == 1
        assert response.candidates[0].isbn_13 == ISBN_13
        assert response.candidates[0].confidence == pytest.approx(1.0)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known implementation gap: CandidateMatcher.detect_duplicates "
            "discards the lower-ranked duplicate instead of merging "
            "complementary fields. Roadmap Section 8 step 4 requires merging "
            "compatible metadata when providers describe the same edition."
        ),
    )
    def test_complementary_provider_fields_are_merged(
        self,
        complementary_provider_responses: dict[str, dict[str, Any]],
        provider_result_factory: Any,
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Complementary fields from both providers are merged into one record."""
        google = complementary_provider_responses["google_books"]
        open_library = complementary_provider_responses["open_library"]

        retriever = metadata_retriever_factory()

        async def fake_google(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(google)]}
            )

        async def fake_open_library(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(open_library)]}
            )

        monkeypatch.setattr(retriever._google, "search_by_isbn", fake_google)
        monkeypatch.setattr(
            retriever._open_library, "search_by_isbn", fake_open_library
        )

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert len(response.candidates) == 1
        merged = response.candidates[0]

        # Google supplies page count; Open Library supplies description,
        # categories, language, and cover.  A correct merge keeps all of them.
        assert merged.page_count == google["page_count"]
        assert merged.description == open_library["description"]
        assert merged.language == open_library["language"]
        assert merged.cover_url == open_library["cover_url"]
        assert set(merged.categories) == set(open_library["categories"])

    def test_conflicting_isbns_are_preserved(
        self,
        conflicting_isbn_provider_responses: dict[str, dict[str, Any]],
        provider_result_factory: Any,
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two editions with different ISBNs remain separate candidates."""
        google = conflicting_isbn_provider_responses["google_books"]
        open_library = conflicting_isbn_provider_responses["open_library"]

        retriever = metadata_retriever_factory()

        async def fake_google(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(google)]}
            )

        async def fake_open_library(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(open_library)]}
            )

        monkeypatch.setattr(retriever._google, "search_by_isbn", fake_google)
        monkeypatch.setattr(
            retriever._open_library, "search_by_isbn", fake_open_library
        )

        response = run_async(
            retriever.retrieve(
                isbn=ISBN_13,
                title=CLEAN_CODE_TITLE,
                author=CLEAN_CODE_AUTHOR,
            )
        )

        # The conflicting ISBN records must not be collapsed together.
        assert len(response.candidates) == 2
        assert {c.isbn_13 for c in response.candidates} == {
            ISBN_13,
            PAPERBACK_ISBN_13,
        }

        by_source = {c.source: c for c in response.candidates}
        assert by_source["google_books"].confidence == pytest.approx(1.0)
        # A conflicting ISBN significantly reduces confidence, but the
        # candidate is preserved rather than silently overwritten.
        assert by_source["open_library"].confidence < by_source["google_books"].confidence
        assert by_source["open_library"].confidence <= 0.5

    def test_conflicting_metadata_same_isbn_is_not_silently_overwritten(
        self,
        conflicting_provider_responses: dict[str, dict[str, Any]],
        provider_result_factory: Any,
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same-edition records are deduplicated without blending conflicting values."""
        google = conflicting_provider_responses["google_books"]
        open_library = conflicting_provider_responses["open_library"]

        retriever = metadata_retriever_factory()

        async def fake_google(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(google)]}
            )

        async def fake_open_library(_isbn: str) -> Any:
            return provider_result_factory(
                data={"candidates": [copy.deepcopy(open_library)]}
            )

        monkeypatch.setattr(retriever._google, "search_by_isbn", fake_google)
        monkeypatch.setattr(
            retriever._open_library, "search_by_isbn", fake_open_library
        )

        response = run_async(
            retriever.retrieve(
                isbn=ISBN_13,
                title=CLEAN_CODE_TITLE,
                author=CLEAN_CODE_AUTHOR,
            )
        )

        # The two providers describe the same ISBN/edition, so the service
        # collapses them to one record rather than presenting the same book
        # twice.
        assert len(response.candidates) == 1
        kept = response.candidates[0]

        # The surviving record keeps its own values; conflicting values from
        # the other provider are neither blended in nor used to overwrite it.
        assert kept.source == "google_books"
        assert kept.publisher == google["publisher"]
        assert kept.page_count == google["page_count"]
        assert kept.published_date == google["published_date"]

    def test_both_providers_return_no_results(
        self,
        mock_metadata_providers: dict[str, Any],
        google_books_empty_response: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """No results from either provider produces an empty success response."""
        mock_metadata_providers["google_books"].add_response(
            url="/volumes",
            json=google_books_empty_response,
        )
        mock_metadata_providers["open_library"].add_response(
            path="/isbn/",
            json={"candidates": []},
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates == []
        assert response.selected_candidate is None

    def test_google_books_timeout_still_queries_open_library(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """A Google Books timeout never prevents Open Library from answering."""
        mock_metadata_providers["google_books"].add_exception(
            httpx.ReadTimeout("Google Books timed out"),
            url="/volumes",
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert len(response.candidates) == 1
        assert response.candidates[0].source == "open_library"

        urls = _request_urls(mock_metadata_providers)
        assert any("openlibrary" in url for url in urls)

        # The provider retries timeouts (initial attempt + METADATA_MAX_RETRIES).
        google_requests = _requests_to(mock_metadata_providers, "googleapis")
        assert len(google_requests) == settings.METADATA_MAX_RETRIES + 1

    def test_open_library_failure_does_not_block_google_result(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raising Open Library provider never suppresses Google results."""
        retriever = metadata_retriever_factory()

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("Open Library exploded")

        monkeypatch.setattr(retriever._open_library, "search_by_isbn", explode)

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert len(response.candidates) == 1
        assert response.candidates[0].source == "google_books"
        assert response.candidates[0].isbn_13 == ISBN_13

    def test_both_providers_http_errors_degrade_gracefully(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """HTTP-level provider failures do not crash the orchestration."""
        mock_metadata_providers["google_books"].add_response(
            url="/volumes", status_code=500, text="Google Books unavailable"
        )
        mock_metadata_providers["open_library"].add_response(
            path="/isbn/", status_code=500, text="Open Library unavailable"
        )
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.status == "success"
        assert response.candidates == []
        assert response.selected_candidate is None

    def test_both_providers_unreachable_raises_service_unavailable(
        self,
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When every provider raises, the service raises MetadataServiceUnavailable."""
        retriever = metadata_retriever_factory()

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("provider down")

        monkeypatch.setattr(retriever._google, "search_by_isbn", explode)
        monkeypatch.setattr(retriever._open_library, "search_by_isbn", explode)

        with pytest.raises(MetadataServiceUnavailable):
            run_async(retriever.retrieve(isbn=ISBN_13))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known implementation gap: _query_google/_query_open_library "
            "report a provider as reachable whenever its search method "
            "returns normally, even when it returns ProviderResult(success="
            "False). Roadmap Section 11 requires HTTP 503 when all required "
            "providers fail and no cached result exists."
        ),
    )
    def test_all_providers_http_failure_raises_service_unavailable(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """All providers returning HTTP errors must surface as a service outage."""
        mock_metadata_providers["google_books"].add_response(
            url="/volumes", status_code=500, text="Google Books unavailable"
        )
        mock_metadata_providers["open_library"].add_response(
            path="/isbn/", status_code=500, text="Open Library unavailable"
        )
        retriever = metadata_retriever_factory()

        with pytest.raises(MetadataServiceUnavailable):
            run_async(retriever.retrieve(isbn=ISBN_13))

    def test_retrieve_book_metadata_returns_plain_dict(
        self,
        mock_metadata_providers: dict[str, Any],
    ) -> None:
        """The module-level convenience wrapper returns a serialisable dict."""
        result = run_async(retrieve_book_metadata(isbn=ISBN_13))

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["query"]["isbn"] == ISBN_13
        assert len(result["candidates"]) == 1
        assert result["selected_candidate"]["isbn_13"] == ISBN_13


# ===========================================================================
# 2. Search modes: ISBN, title, author
# ===========================================================================


class TestSearchModes:
    """Which provider methods are called for each query shape."""

    def test_only_title_supplied_performs_title_search(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(title=CLEAN_CODE_TITLE))

        assert response.query.title == CLEAN_CODE_TITLE
        assert response.query.isbn is None
        assert response.candidates

        google_requests = _requests_to(mock_metadata_providers, "volumes")
        assert google_requests
        google_query = google_requests[-1].url.params.get("q", "")
        assert 'intitle:"Clean Code"' in google_query
        assert "isbn:" not in google_query

        open_library_requests = _requests_to(
            mock_metadata_providers, "search.json"
        )
        assert open_library_requests
        assert (
            open_library_requests[-1].url.params.get("title")
            == CLEAN_CODE_TITLE
        )

    def test_only_author_supplied_performs_author_search(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(author=CLEAN_CODE_AUTHOR))

        assert response.query.author == CLEAN_CODE_AUTHOR
        assert response.query.title is None
        assert response.query.isbn is None
        assert response.candidates

        google_requests = _requests_to(mock_metadata_providers, "volumes")
        assert google_requests
        google_query = google_requests[-1].url.params.get("q", "")
        assert f'inauthor:"{CLEAN_CODE_AUTHOR}"' in google_query
        assert "isbn:" not in google_query

        open_library_requests = _requests_to(
            mock_metadata_providers, "search.json"
        )
        assert open_library_requests
        # With no title, Open Library receives the author as a general query.
        assert (
            open_library_requests[-1].url.params.get("q")
            == CLEAN_CODE_AUTHOR
        )

    @pytest.mark.parametrize("bad_isbn", INVALID_ISBNS)
    def test_invalid_isbn_rejects_request(
        self,
        bad_isbn: str,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """An invalid ISBN checksum fails validation before any provider call."""
        retriever = metadata_retriever_factory()

        with pytest.raises(ValidationError):
            run_async(retriever.retrieve(isbn=bad_isbn))

        # The request is rejected before the pipeline touches the providers.
        assert mock_metadata_providers["google_books"].request_count == 0


# ===========================================================================
# 3. Scraping fallback
# ===========================================================================


class TestScrapingFallback:
    """Scraping is a disabled-by-default, conditional fallback."""

    def test_scraping_disabled_does_not_invoke_scraper(
        self,
        disable_scraping: bool,
        mock_metadata_providers: dict[str, Any],
        google_books_empty_response: dict[str, Any],
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with insufficient API results, a disabled scraper is never called."""
        assert disable_scraping is False
        mock_metadata_providers["google_books"].add_response(
            url="/volumes",
            json=google_books_empty_response,
        )
        mock_metadata_providers["open_library"].add_response(
            path="/isbn/",
            json={"candidates": []},
        )
        retriever = metadata_retriever_factory()

        calls: list[tuple[str, Any]] = []

        def fake_search_by_isbn(isbn: str) -> list[BookMetadata]:
            calls.append(("isbn", isbn))
            return []

        def fake_search_by_title_author(
            title: str, author: Optional[str] = None
        ) -> list[BookMetadata]:
            calls.append(("title_author", (title, author)))
            return []

        monkeypatch.setattr(
            retriever._scraper, "search_by_isbn", fake_search_by_isbn
        )
        monkeypatch.setattr(
            retriever._scraper,
            "search_by_title_author",
            fake_search_by_title_author,
        )

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.candidates == []
        assert calls == []

    def test_scraping_enabled_invokes_fallback_when_apis_insufficient(
        self,
        enable_scraping: bool,
        mock_metadata_providers: dict[str, Any],
        google_books_empty_response: dict[str, Any],
        metadata_retriever_factory: Any,
        book_metadata_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When APIs return nothing and scraping is enabled, the scraper runs."""
        assert enable_scraping is True
        mock_metadata_providers["google_books"].add_response(
            url="/volumes",
            json=google_books_empty_response,
        )
        mock_metadata_providers["open_library"].add_response(
            path="/isbn/",
            json={"candidates": []},
        )
        retriever = metadata_retriever_factory()

        scraped = book_metadata_factory(
            source="goodreads",
            source_id="goodreads-clean-code",
        )
        calls: list[str] = []

        def fake_search_by_isbn(isbn: str) -> list[BookMetadata]:
            calls.append(isbn)
            return [scraped]

        monkeypatch.setattr(
            retriever._scraper, "search_by_isbn", fake_search_by_isbn
        )

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert calls == [ISBN_13]
        assert len(response.candidates) == 1
        assert response.candidates[0].source == "goodreads"
        assert response.candidates[0].confidence == pytest.approx(1.0)
        assert response.selected_candidate is not None

    def test_scraping_enabled_skipped_when_api_results_sufficient(
        self,
        enable_scraping: bool,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A confident API result suppresses the scraping fallback."""
        assert enable_scraping is True
        retriever = metadata_retriever_factory()

        calls: list[str] = []

        def fake_search_by_isbn(isbn: str) -> list[BookMetadata]:
            calls.append(isbn)
            return []

        monkeypatch.setattr(
            retriever._scraper, "search_by_isbn", fake_search_by_isbn
        )

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.candidates
        assert response.candidates[0].confidence >= 0.7
        assert calls == []


# ===========================================================================
# 4. Caching behaviour
# ===========================================================================


class TestCaching:
    """The orchestrator consults and populates the SQLite metadata cache."""

    def test_populated_cache_short_circuits_providers(
        self,
        populated_cache: Any,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """A cache hit returns stored candidates without calling providers."""
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert len(response.candidates) == 1
        assert response.candidates[0].title == CLEAN_CODE_TITLE
        assert response.candidates[0].isbn_13 == ISBN_13
        assert mock_metadata_providers["google_books"].request_count == 0

    def test_successful_result_is_cached_and_reused(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """The second identical query is served entirely from cache."""
        retriever = metadata_retriever_factory()

        first = run_async(retriever.retrieve(isbn=ISBN_13))
        request_count_after_first = mock_metadata_providers[
            "google_books"
        ].request_count
        assert request_count_after_first > 0
        assert first.candidates

        second = run_async(retriever.retrieve(isbn=ISBN_13))

        assert (
            mock_metadata_providers["google_books"].request_count
            == request_count_after_first
        )
        assert [c.title for c in second.candidates] == [
            c.title for c in first.candidates
        ]
        assert [c.isbn_13 for c in second.candidates] == [
            c.isbn_13 for c in first.candidates
        ]

    def test_hyphenated_isbn_hits_the_same_cache_entry(
        self,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """Cache keys normalise separators so formatted ISBNs share an entry."""
        retriever = metadata_retriever_factory()

        run_async(retriever.retrieve(isbn=ISBN_13))
        request_count_after_first = mock_metadata_providers[
            "google_books"
        ].request_count

        response = run_async(retriever.retrieve(isbn="978-0132350884"))

        assert (
            mock_metadata_providers["google_books"].request_count
            == request_count_after_first
        )
        assert response.query.isbn == ISBN_13
        assert response.candidates
        assert response.candidates[0].isbn_13 == ISBN_13

    def test_empty_results_are_not_cached(
        self,
        mock_metadata_providers: dict[str, Any],
        google_books_empty_response: dict[str, Any],
        google_books_response: dict[str, Any],
        open_library_isbn_response: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """Failed/empty lookups are re-queried rather than cached indefinitely."""
        router = mock_metadata_providers["google_books"]
        router.add_response(url="/volumes", json=google_books_empty_response)
        router.add_response(path="/isbn/", json={"candidates": []})

        retriever = metadata_retriever_factory()

        first = run_async(retriever.retrieve(isbn=ISBN_13))
        assert first.candidates == []
        request_count_after_first = router.request_count

        # Providers start returning data; higher priority overrides the empties.
        router.add_response(
            url="/volumes", json=google_books_response, priority=5
        )
        router.add_response(
            path="/isbn/", json=open_library_isbn_response, priority=5
        )

        second = run_async(retriever.retrieve(isbn=ISBN_13))

        assert second.candidates
        assert router.request_count > request_count_after_first


# ===========================================================================
# 5. METADATA_STRATEGY configurations
# ===========================================================================


class TestMetadataStrategies:
    """Provider selection honours ``settings.METADATA_STRATEGY``."""

    def test_google_only_strategy_skips_open_library(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_STRATEGY", "GOOGLE_ONLY")
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.candidates
        assert response.candidates[0].source == "google_books"
        urls = _request_urls(mock_metadata_providers)
        assert any("googleapis" in url for url in urls)
        assert not any("openlibrary" in url for url in urls)

    def test_open_library_only_strategy_skips_google_books(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_STRATEGY", "OPEN_LIBRARY_ONLY")
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.candidates
        assert response.candidates[0].source == "open_library"
        assert response.candidates[0].authors == [CLEAN_CODE_AUTHOR]
        urls = _request_urls(mock_metadata_providers)
        assert any("openlibrary" in url for url in urls)
        assert not any("googleapis" in url for url in urls)

    def test_hybrid_strategy_queries_both_providers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        monkeypatch.setattr(settings, "METADATA_STRATEGY", "HYBRID")
        retriever = metadata_retriever_factory()

        response = run_async(retriever.retrieve(isbn=ISBN_13))

        assert response.candidates
        urls = _request_urls(mock_metadata_providers)
        assert any("googleapis" in url for url in urls)
        assert any("openlibrary" in url for url in urls)

    def test_unknown_strategy_raises_service_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_metadata_providers: dict[str, Any],
        metadata_retriever_factory: Any,
    ) -> None:
        """An unrecognised strategy selects no providers and reports an outage."""
        monkeypatch.setattr(settings, "METADATA_STRATEGY", "MAGIC")
        retriever = metadata_retriever_factory()

        with pytest.raises(MetadataServiceUnavailable):
            run_async(retriever.retrieve(isbn=ISBN_13))

        assert mock_metadata_providers["google_books"].request_count == 0


# ===========================================================================
# 6. Ambiguity and candidate selection
# ===========================================================================


class TestCandidateSelection:
    """Automatic selection only happens when one candidate is clearly best."""

    def test_multiple_ambiguous_candidates_are_not_selected(
        self,
        multiple_editions: list[BookMetadata],
        provider_result_factory: Any,
        metadata_retriever_factory: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Several equally plausible editions leave ``selected_candidate`` null."""
        retriever = metadata_retriever_factory()
        raw_editions = _dumps(multiple_editions)

        async def fake_google(
            _title: str, _author: Optional[str] = None
        ) -> Any:
            return provider_result_factory(
                data={"candidates": copy.deepcopy(raw_editions)}
            )

        async def fake_open_library(
            _title: str, _author: Optional[str] = None
        ) -> Any:
            return provider_result_factory(data={"candidates": []})

        monkeypatch.setattr(
            retriever._google, "search_by_title_author", fake_google
        )
        monkeypatch.setattr(
            retriever._open_library,
            "search_by_title_author",
            fake_open_library,
        )

        response = run_async(
            retriever.retrieve(
                title=CLEAN_CODE_TITLE,
                author=CLEAN_CODE_AUTHOR,
            )
        )

        assert len(response.candidates) >= 3
        assert response.selected_candidate is None

        top, runner_up = response.candidates[0], response.candidates[1]
        assert top.confidence - runner_up.confidence < 0.15
