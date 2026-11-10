"""Endpoint tests for ``POST /v1/books/metadata/search``.

Covers Part 2, Section 20 of
``specs/roadmap/phase-03-metadata-retrieval.md``.

Design notes
------------
* **No live external calls.**  Every test replaces
  ``app.books.router.retrieve_book_metadata`` with an async fake, so the
  Google Books / Open Library / scraping pipeline is never executed.  The
  individual provider interactions are exercised by the unit and integration
  suites under ``app/books/metadata/tests/``.
* **Authentication uses the shared fixtures** (``auth_headers``,
  ``authenticated_client``, ``unauthenticated_client``) defined in
  ``app/books/tests/conftest.py``.
* **Contract first.**  Beyond status codes, the tests assert the response
  matches :class:`MetadataSearchResponse`, that confidence values stay within
  ``0..1``, and that credentials, provider internals, raw HTML and
  unprocessed provider payloads never reach clients.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from starlette import status
from starlette.testclient import TestClient

from app.books.exceptions import MetadataServiceUnavailable
from app.books.metadata.schemas import MetadataSearchResponse
from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_URL = "/v1/books/metadata/search"

VALID_ISBN_13 = "9780132350884"  # Clean Code (Prentice Hall, 2008)
VALID_ISBN_10 = "0132350882"

#: Exact public keys the response model is allowed to expose.
EXPECTED_TOP_LEVEL_KEYS = {
    "status",
    "query",
    "candidates",
    "selected_candidate",
}

EXPECTED_CANDIDATE_KEYS = {
    "title",
    "subtitle",
    "authors",
    "publisher",
    "published_date",
    "isbn_13",
    "isbn_10",
    "page_count",
    "categories",
    "language",
    "cover_url",
    "source",
    "confidence",
}

#: Substrings that must never leak from provider internals to a client.
INTERNAL_LEAK_MARKERS = (
    "traceback",
    "httpx",
    "requests.exceptions",
    "connectionpool",
    "timeoutexception",
    "429",
    "rate limit",
    "volumeinfo",
    "books#volumes",
    "openlibrary.org",
    "googleapis.com",
    "stack trace",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(**overrides: Any) -> dict[str, Any]:
    """Return a normalized metadata candidate dict with sensible defaults."""
    candidate: dict[str, Any] = {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "published_date": "2008",
        "isbn_13": VALID_ISBN_13,
        "isbn_10": VALID_ISBN_10,
        "page_count": 464,
        "categories": ["Computers"],
        "language": "en",
        "cover_url": None,
        "source": "google_books",
        "confidence": 0.92,
    }
    candidate.update(overrides)
    return candidate


def make_result(
    *,
    query: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    selected_candidate: dict[str, Any] | None = None,
    status_value: str = "success",
) -> dict[str, Any]:
    """Build a ``MetadataSearchResponse``-compatible result dict."""
    return {
        "status": status_value,
        "query": query or {},
        "candidates": list(candidates or []),
        "selected_candidate": selected_candidate,
    }


def install_fake_retriever(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: dict[str, Any] | None = None,
    error: BaseException | None = None,
    captured: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Patch the router's ``retrieve_book_metadata`` with an async fake.

    Parameters
    ----------
    result:
        Dict returned by the fake.  Deep-copied on every call so tests cannot
        mutate shared fixture state.
    error:
        Exception raised by the fake instead of returning ``result``.
    captured:
        Optional dict populated with the keyword arguments the endpoint
        forwarded to the service (``isbn``, ``title``, ``author``).
    """

    async def fake_retrieve_book_metadata(
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> dict[str, Any]:
        if captured is not None:
            captured.update({"isbn": isbn, "title": title, "author": author})
        if error is not None:
            raise error
        assert result is not None, "Test forgot to provide a result for the fake retriever"
        return copy.deepcopy(result)

    monkeypatch.setattr(
        "app.books.router.retrieve_book_metadata",
        fake_retrieve_book_metadata,
    )
    return fake_retrieve_book_metadata


# ---------------------------------------------------------------------------
# 1. Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    """The endpoint must be protected by the existing JWT dependency."""

    def test_unauthenticated_request_returns_401(
        self,
        unauthenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(query={"isbn": VALID_ISBN_13}),
            captured=captured,
        )

        resp = unauthenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_401_UNAUTHORIZED
        # The bearer challenge is advertised so clients know how to retry.
        assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")
        body = resp.json()
        assert "message" in body
        # Authentication is enforced *before* any provider work happens.
        assert captured == {}

    def test_invalid_token_returns_401(
        self,
        unauthenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(query={"title": "Clean Code"}),
            captured=captured,
        )

        resp = unauthenticated_client.post(
            METADATA_URL,
            json={"title": "Clean Code"},
            headers={"Authorization": "Bearer this-is-not-a-valid-jwt"},
        )

        assert resp.status_code == status.HTTP_401_UNAUTHORIZED
        assert captured == {}


# ---------------------------------------------------------------------------
# 2. Search by each supported parameter
# ---------------------------------------------------------------------------


class TestSearchByParameter:
    """Successful searches keyed on ISBN, title, or author all return 200."""

    def test_search_by_isbn_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"isbn": VALID_ISBN_13},
                candidates=[make_candidate()],
                selected_candidate=make_candidate(),
            ),
            captured=captured,
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_200_OK
        assert captured == {
            "isbn": VALID_ISBN_13,
            "title": None,
            "author": None,
        }
        body = resp.json()
        assert body["status"] == "success"
        assert body["query"]["isbn"] == VALID_ISBN_13
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["isbn_13"] == VALID_ISBN_13

    def test_search_by_title_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"title": "Clean Code"},
                candidates=[make_candidate()],
                selected_candidate=make_candidate(),
            ),
            captured=captured,
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"title": "Clean Code"}
        )

        assert resp.status_code == status.HTTP_200_OK
        assert captured == {
            "isbn": None,
            "title": "Clean Code",
            "author": None,
        }
        body = resp.json()
        assert body["query"]["title"] == "Clean Code"
        assert body["candidates"][0]["title"] == "Clean Code"

    def test_search_by_author_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"author": "Robert C. Martin"},
                candidates=[make_candidate()],
            ),
            captured=captured,
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"author": "Robert C. Martin"}
        )

        assert resp.status_code == status.HTTP_200_OK
        assert captured == {
            "isbn": None,
            "title": None,
            "author": "Robert C. Martin",
        }
        body = resp.json()
        assert body["query"]["author"] == "Robert C. Martin"
        assert body["candidates"][0]["authors"] == ["Robert C. Martin"]

    def test_search_with_all_parameters_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={
                    "isbn": VALID_ISBN_13,
                    "title": "Clean Code",
                    "author": "Robert C. Martin",
                },
                candidates=[make_candidate()],
            ),
            captured=captured,
        )

        resp = authenticated_client.post(
            METADATA_URL,
            json={
                "isbn": VALID_ISBN_13,
                "title": "Clean Code",
                "author": "Robert C. Martin",
            },
        )

        assert resp.status_code == status.HTTP_200_OK
        assert captured == {
            "isbn": VALID_ISBN_13,
            "title": "Clean Code",
            "author": "Robert C. Martin",
        }


# ---------------------------------------------------------------------------
# 3. Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Malformed or incomplete searches are rejected before provider work."""

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"isbn": None, "title": None, "author": None},
        ],
        ids=["empty-body", "explicit-nulls"],
    )
    def test_missing_all_search_parameters_returns_400(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        payload: dict[str, Any],
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(),
            captured=captured,
        )

        resp = authenticated_client.post(METADATA_URL, json=payload)

        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        body = resp.json()
        assert "message" in body
        assert "at least one" in body["message"].lower()
        # No provider lookup should be attempted for an empty query.
        assert captured == {}

    @pytest.mark.parametrize(
        "invalid_isbn",
        [
            "not-an-isbn",       # contains letters
            "12345",             # wrong length
            "9780132350885",     # invalid ISBN-13 checksum
            "0132350883",        # invalid ISBN-10 checksum
        ],
        ids=[
            "non-numeric",
            "wrong-length",
            "bad-isbn13-checksum",
            "bad-isbn10-checksum",
        ],
    )
    def test_invalid_isbn_returns_422(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        invalid_isbn: str,
    ) -> None:
        captured: dict[str, Any] = {}
        install_fake_retriever(
            monkeypatch,
            result=make_result(query={"isbn": invalid_isbn}),
            captured=captured,
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": invalid_isbn}
        )

        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        body = resp.json()
        assert "message" in body
        # Request validation must short-circuit before the service runs.
        assert captured == {}


# ---------------------------------------------------------------------------
# 4. Result states
# ---------------------------------------------------------------------------


class TestResultStates:
    """No hits, outages, partial provider failures, and ambiguity."""

    def test_no_matching_book_returns_200_with_empty_candidates(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"title": "A Book That Does Not Exist"},
                candidates=[],
                selected_candidate=None,
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"title": "A Book That Does Not Exist"}
        )

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["status"] == "success"
        assert body["candidates"] == []
        assert body["selected_candidate"] is None

    def test_all_providers_unavailable_without_cache_returns_503(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_fake_retriever(
            monkeypatch,
            error=MetadataServiceUnavailable(),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        body = resp.json()
        # Only the public, provider-agnostic message is exposed.
        assert body["message"] == MetadataServiceUnavailable.message
        lowered = resp.text.lower()
        for marker in INTERNAL_LEAK_MARKERS:
            assert marker not in lowered, f"internal detail leaked: {marker!r}"

    def test_google_books_unavailable_open_library_succeeds_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        open_library_candidate = make_candidate(
            source="open_library",
            confidence=0.88,
        )
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"isbn": VALID_ISBN_13},
                candidates=[open_library_candidate],
                selected_candidate=open_library_candidate,
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        # One provider failing must not turn a usable result into an error.
        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["source"] == "open_library"
        assert body["selected_candidate"]["source"] == "open_library"

    def test_valid_request_with_multiple_candidates_returns_200(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = make_candidate(confidence=0.92, source="google_books")
        second = make_candidate(
            title="Clean Architecture",
            isbn_13="9780134494166",
            isbn_10="0134494168",
            confidence=0.61,
            source="open_library",
        )
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"title": "Clean Code", "author": "Robert C. Martin"},
                candidates=[first, second],
                selected_candidate=None,
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL,
            json={"title": "Clean Code", "author": "Robert C. Martin"},
        )

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert len(body["candidates"]) == 2
        assert {c["source"] for c in body["candidates"]} == {
            "google_books",
            "open_library",
        }
        # Ambiguous results are not auto-selected.
        assert body["selected_candidate"] is None


# ---------------------------------------------------------------------------
# 5. Response contract & data-safety guarantees
# ---------------------------------------------------------------------------


class TestResponseContract:
    """The payload must honour the schema and never leak sensitive internals."""

    def test_response_follows_pydantic_schema(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        candidate = make_candidate()
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"isbn": VALID_ISBN_13},
                candidates=[candidate],
                selected_candidate=candidate,
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()

        # Round-tripping through the response model proves schema compliance.
        parsed = MetadataSearchResponse.model_validate(body)
        assert parsed.status == "success"
        assert parsed.query.isbn == VALID_ISBN_13
        assert len(parsed.candidates) == 1
        assert parsed.selected_candidate is not None
        assert parsed.selected_candidate.title == "Clean Code"

        # Only documented keys are present.
        assert set(body) == EXPECTED_TOP_LEVEL_KEYS
        assert set(body["query"]) == {"isbn", "title", "author"}
        for item in body["candidates"]:
            assert set(item) == EXPECTED_CANDIDATE_KEYS

    def test_candidate_confidence_is_within_zero_and_one(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        candidates = [
            make_candidate(confidence=0.0),
            make_candidate(isbn_13=VALID_ISBN_13, confidence=0.57),
            make_candidate(isbn_13=VALID_ISBN_13, confidence=1.0),
        ]
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"title": "Clean Code"},
                candidates=candidates,
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"title": "Clean Code"}
        )

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert len(body["candidates"]) == 3
        for candidate in body["candidates"]:
            confidence = candidate["confidence"]
            assert 0.0 <= confidence <= 1.0
        parsed = MetadataSearchResponse.model_validate(body)
        for candidate in parsed.candidates:
            assert 0.0 <= candidate.confidence <= 1.0

    def test_api_credentials_are_not_exposed(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret_api_key = "GOOGLE-SECRET-KEY-do-not-leak-1234567890"
        monkeypatch.setattr(settings, "GOOGLE_BOOKS_API_KEY", secret_api_key)

        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"isbn": VALID_ISBN_13},
                candidates=[make_candidate()],
                selected_candidate=make_candidate(),
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_200_OK
        assert secret_api_key not in resp.text
        lowered = resp.text.lower()
        assert "api_key" not in lowered
        assert "apikey" not in lowered
        assert secret_api_key.lower() not in lowered

    def test_provider_internal_errors_are_not_exposed(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_fake_retriever(
            monkeypatch,
            error=MetadataServiceUnavailable(),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        body = resp.json()
        # The error envelope contains only a single human-readable message.
        assert set(body) == {"message"}
        assert body["message"] == MetadataServiceUnavailable.message
        lowered = resp.text.lower()
        for marker in INTERNAL_LEAK_MARKERS:
            assert marker not in lowered, f"internal detail leaked: {marker!r}"

    def test_raw_html_and_unprocessed_provider_payloads_are_not_returned(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Smuggle provider-native keys and markup into the service result.  The
        # response model must strip them before serialization.
        dirty_candidate = make_candidate(
            volumeInfo={"title": "Clean Code", "raw": "<b>unprocessed</b>"},
            raw_html="<html><body>google books payload</body></html>",
            searchInfo={"textSnippet": "unprocessed snippet"},
            accessInfo={"viewability": "PARTIAL"},
        )
        dirty_result = make_result(
            query={"isbn": VALID_ISBN_13},
            candidates=[dirty_candidate],
            selected_candidate=dirty_candidate,
        )
        dirty_result["raw_payload"] = "<html>unprocessed top-level payload</html>"
        dirty_result["provider_response"] = {"kind": "books#volumes", "items": []}

        install_fake_retriever(monkeypatch, result=dirty_result)

        resp = authenticated_client.post(
            METADATA_URL, json={"isbn": VALID_ISBN_13}
        )

        assert resp.status_code == status.HTTP_200_OK
        assert resp.headers["content-type"].startswith("application/json")

        text = resp.text
        lowered = text.lower()
        assert "<html" not in lowered
        assert "unprocessed" not in lowered
        assert "volumeinfo" not in lowered
        assert "searchinfo" not in lowered
        assert "accessinfo" not in lowered
        assert "provider_response" not in text
        assert "raw_payload" not in text

        body = resp.json()
        assert set(body) == EXPECTED_TOP_LEVEL_KEYS
        for item in body["candidates"]:
            assert set(item) == EXPECTED_CANDIDATE_KEYS
        assert set(body["selected_candidate"]) == EXPECTED_CANDIDATE_KEYS

    def test_search_result_does_not_contain_exception_details(
        self,
        authenticated_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A successful result should be pure metadata, with no diagnostic
        # scaffolding such as exception class names or tracebacks.
        install_fake_retriever(
            monkeypatch,
            result=make_result(
                query={"author": "Robert C. Martin"},
                candidates=[make_candidate()],
            ),
        )

        resp = authenticated_client.post(
            METADATA_URL, json={"author": "Robert C. Martin"}
        )

        assert resp.status_code == status.HTTP_200_OK
        lowered = resp.text.lower()
        assert "exception" not in lowered
        assert "traceback" not in lowered
        assert "error" not in lowered
