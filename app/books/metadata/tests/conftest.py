"""Reusable pytest fixtures for the Phase 3 metadata retrieval test suite.

This module is discovered automatically by pytest for every test module under
``app/books/metadata/tests/``.  The fixtures cover the requirements in
``specs/roadmap/phase-03-metadata-retrieval.md`` (Part 2, Section 21) and are
intentionally dependency-free beyond the packages already used by the
application (``pytest``, ``httpx``, ``requests``, ``starlette``).

Design notes
------------
* **No live network calls.**  An autouse fixture installs fail-closed guards
  around ``httpx``/``requests``.  Tests must explicitly opt in to mocked
  responses with ``mock_http`` / ``mock_scraper_http``.
* **Isolation.**  An autouse fixture points ``settings.DATABASE_PATH`` at the
  per-test ``tmp_path`` so the SQLite cache created by the metadata service
  never touches the real application database.
* **Speed.**  Provider retry back-offs (``asyncio.sleep``) are collapsed to
  zero delays for tests, while retry *behaviour* is still exercised.
* **Provider payloads are copied.**  Fixtures return deep copies so a test
  that mutates a payload cannot leak state into another test.

The fixtures fall into these groups:

Provider payloads
    ``google_books_response``, ``open_library_isbn_response``,
    ``open_library_search_response``, ``multiple_editions``, ...
Error/edge-case data
    ``empty_provider_response``, ``malformed_provider_response``,
    ``invalid_isbns``, ...
Authentication
    ``valid_authenticated_user``, ``authenticated_client``, ``auth_headers``
Caching
    ``temp_cache_db_path``, ``metadata_cache``, ``populated_cache``
HTTP mocking
    ``mock_http``, ``mock_google_books_http``, ``mock_open_library_http``,
    ``mock_metadata_providers``, ``mock_scraper_http``
Scraping samples
    ``goodreads_html``, ``amazon_html``, ``goodreads_search_html``, ...
Unit helpers
    ``normalizer``, ``matcher``, ``book_metadata_factory``,
    ``provider_result_factory``, ``metadata_retriever_factory``
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import httpx
import pytest
import requests
from starlette.testclient import TestClient

from app.books.metadata.cache import MetadataCache
from app.books.metadata.matcher import CandidateMatcher
from app.books.metadata.normalizer import MetadataNormalizer
from app.books.metadata.providers.base import ProviderResult
from app.books.metadata.providers.google_books import GoogleBooksProvider
from app.books.metadata.providers.open_library import OpenLibraryProvider
from app.books.metadata.providers.scraper import MetadataScraper
from app.books.metadata.schemas import BookMetadata
from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical metadata search endpoint.
METADATA_ENDPOINT = "/v1/books/metadata/search"

#: A valid ISBN-13 / ISBN-10 pair used throughout the fixtures.
VALID_ISBN_13 = "9780132350884"  # Clean Code (Prentice Hall, 2008)
VALID_ISBN_10 = "0132350882"

#: Additional valid ISBNs from the roadmap examples and edition fixtures.
VALID_ISBN_13_ROADMAP = "9780135957059"
VALID_ISBN_10_ROADMAP = "0321125215"
VALID_ISBN_10_ROADMAP_HYPHENATED = "0-321-12521-5"
VALID_ISBN_13_HYPHENATED = "978-0-13-595705-9"

#: ISBNs for the alternative editions used by the matcher tests.
PAPERBACK_ISBN_13 = "9780137081073"
PAPERBACK_ISBN_10 = "0137081071"
EBOOK_ISBN_13 = "9780135957059"
EBOOK_ISBN_10 = "0135957052"
ARCHITECTURE_ISBN_13 = "9780134494166"
ARCHITECTURE_ISBN_10 = "0134494168"

# ---------------------------------------------------------------------------
# Sample provider payloads
# ---------------------------------------------------------------------------

GOOGLE_BOOKS_VOLUME: dict[str, Any] = {
    "kind": "books#volume",
    "id": "zyTCAlFPjgYC",
    "etag": "book-keepers-test-etag",
    "selfLink": "https://www.googleapis.com/books/v1/volumes/zyTCAlFPjgYC",
    "volumeInfo": {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "publishedDate": "2008-08-01",
        "description": (
            "Even bad code can function. But if code isn't clean, it can bring "
            "a development organization to its knees."
        ),
        "industryIdentifiers": [
            {"type": "ISBN_10", "identifier": VALID_ISBN_10},
            {"type": "ISBN_13", "identifier": VALID_ISBN_13},
        ],
        "readingModes": {"text": True, "image": False},
        "pageCount": 464,
        "printType": "BOOK",
        "categories": ["Computers", "Software Engineering"],
        "averageRating": 4.5,
        "ratingsCount": 1234,
        "maturityRating": "NOT_MATURE",
        "allowAnonLogging": True,
        "contentVersion": "1.2.3.4.preview.0",
        "panelizations": {"image": False, "text": True},
        "imageLinks": {
            "smallThumbnail": (
                "https://books.google.com/books/content?id=zyTCAlFPjgYC"
                "&printsec=frontcover&img=1&zoom=5"
            ),
            "thumbnail": (
                "https://books.google.com/books/content?id=zyTCAlFPjgYC"
                "&printsec=frontcover&img=1&zoom=1"
            ),
            "medium": (
                "https://books.google.com/books/content?id=zyTCAlFPjgYC"
                "&printsec=frontcover&img=1&zoom=2"
            ),
        },
        "language": "en",
        "previewLink": "https://books.google.com/books?id=zyTCAlFPjgYC&printsec=frontcover",
        "infoLink": "https://books.google.com/books?id=zyTCAlFPjgYC",
        "canonicalVolumeLink": "https://books.google.com/books/about/Clean_Code.html?id=zyTCAlFPjgYC",
    },
    "saleInfo": {
        "country": "US",
        "saleability": "FOR_SALE",
        "isEbook": True,
        "listPrice": {"amount": 39.99, "currencyCode": "USD"},
        "retailPrice": {"amount": 29.99, "currencyCode": "USD"},
        "buyLink": "https://play.google.com/store/books/details?id=zyTCAlFPjgYC",
    },
    "accessInfo": {
        "country": "US",
        "viewability": "PARTIAL",
        "embeddable": True,
        "publicDomain": False,
        "textToSpeechPermission": "ALLOWED",
        "epub": {"isAvailable": True},
        "pdf": {"isAvailable": True},
        "webReaderLink": "https://play.google.com/books/reader?id=zyTCAlFPjgYC",
        "accessViewStatus": "SAMPLE",
        "quoteSharingAllowed": False,
    },
    "searchInfo": {
        "textSnippet": "Even bad code can function. But if code isn&#39;t clean..."
    },
}

GOOGLE_BOOKS_RESPONSE: dict[str, Any] = {
    "kind": "books#volumes",
    "totalItems": 1,
    "items": [GOOGLE_BOOKS_VOLUME],
}

GOOGLE_BOOKS_PAPERBACK_VOLUME: dict[str, Any] = {
    "kind": "books#volume",
    "id": "paperback-clean-code",
    "etag": "book-keepers-test-etag-paperback",
    "volumeInfo": {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "publishedDate": "2009-07-01",
        "description": "A paperback printing of the same handbook.",
        "industryIdentifiers": [
            {"type": "ISBN_10", "identifier": PAPERBACK_ISBN_10},
            {"type": "ISBN_13", "identifier": PAPERBACK_ISBN_13},
        ],
        "pageCount": 464,
        "printType": "BOOK",
        "categories": ["Computers"],
        "language": "en",
        "imageLinks": {
            "thumbnail": (
                "https://books.google.com/books/content?id=paperback-clean-code"
                "&printsec=frontcover&img=1&zoom=1"
            )
        },
        "infoLink": "https://books.google.com/books?id=paperback-clean-code",
    },
    "saleInfo": {"country": "US", "saleability": "FOR_SALE", "isEbook": False},
    "accessInfo": {"country": "US", "viewability": "PARTIAL"},
}

GOOGLE_BOOKS_EBOOK_VOLUME: dict[str, Any] = {
    "kind": "books#volume",
    "id": "ebook-clean-code",
    "etag": "book-keepers-test-etag-ebook",
    "volumeInfo": {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Pearson",
        "publishedDate": "2012-03-01",
        "description": "The ebook edition of Clean Code.",
        "industryIdentifiers": [
            {"type": "ISBN_10", "identifier": EBOOK_ISBN_10},
            {"type": "ISBN_13", "identifier": EBOOK_ISBN_13},
        ],
        "pageCount": 480,
        "printType": "BOOK",
        "categories": ["Computers", "Programming"],
        "language": "en",
        "imageLinks": {
            "thumbnail": (
                "https://books.google.com/books/content?id=ebook-clean-code"
                "&printsec=frontcover&img=1&zoom=1"
            )
        },
    },
    "saleInfo": {"country": "US", "saleability": "FOR_SALE", "isEbook": True},
    "accessInfo": {"country": "US", "viewability": "PARTIAL"},
}

GOOGLE_BOOKS_MULTI_RESPONSE: dict[str, Any] = {
    "kind": "books#volumes",
    "totalItems": 3,
    "items": [
        GOOGLE_BOOKS_VOLUME,
        GOOGLE_BOOKS_PAPERBACK_VOLUME,
        GOOGLE_BOOKS_EBOOK_VOLUME,
    ],
}

GOOGLE_BOOKS_EMPTY_RESPONSE: dict[str, Any] = {
    "kind": "books#volumes",
    "totalItems": 0,
    # Deliberately no ``items`` key – matches the real API for zero results.
}

GOOGLE_BOOKS_MISSING_FIELDS_RESPONSE: dict[str, Any] = {
    "kind": "books#volumes",
    "totalItems": 1,
    "items": [
        {
            "kind": "books#volume",
            "id": "no-isbn-no-authors",
            "volumeInfo": {
                "title": "An Anonymous Work",
                # No authors, no identifiers, no imageLinks, no page count.
            },
        }
    ],
}

OPEN_LIBRARY_ISBN_RESPONSE: dict[str, Any] = {
    "key": "/books/OL123456M",
    "title": "Clean Code",
    "subtitle": "A Handbook of Agile Software Craftsmanship",
    "authors": [
        {"key": "/authors/OL12345A"},
    ],
    "publishers": ["Prentice Hall"],
    "publish_date": "2008",
    "publish_places": ["Upper Saddle River, NJ"],
    "number_of_pages": 464,
    "isbn_10": [VALID_ISBN_10],
    "isbn_13": [VALID_ISBN_13],
    "covers": [1234567],
    "subjects": [
        "Computer programming",
        "Software engineering",
        "Agile software development",
    ],
    "subject_places": ["United States"],
    "languages": [{"key": "/languages/eng"}],
    "works": [{"key": "/works/OL98765W"}],
    "description": {
        "type": "/type/text",
        "value": (
            "Even bad code can function. But if code isn't clean, it can bring "
            "a development organization to its knees."
        ),
    },
    "url": "https://openlibrary.org/books/OL123456M/Clean_Code",
}

OPEN_LIBRARY_AUTHOR_RESPONSE: dict[str, Any] = {
    "key": "/authors/OL12345A",
    "name": "Robert C. Martin",
    "personal_name": "Robert Cecil Martin",
    "birth_date": "1952",
    "type": {"key": "/type/author"},
}

OPEN_LIBRARY_WORK_RESPONSE: dict[str, Any] = {
    "key": "/works/OL98765W",
    "title": "Clean Code",
    "subtitle": "A Handbook of Agile Software Craftsmanship",
    "authors": [{"author": {"key": "/authors/OL12345A"}}],
    "first_publish_date": "2008",
    "subjects": ["Computer programming", "Software engineering"],
    "covers": [1234567],
    "description": {
        "type": "/type/text",
        "value": "A handbook of agile software craftsmanship.",
    },
}

OPEN_LIBRARY_SEARCH_RESPONSE: dict[str, Any] = {
    "numFound": 3,
    "start": 0,
    "docs": [
        {
            "key": "/works/OL98765W",
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "author_name": ["Robert C. Martin"],
            "isbn": [VALID_ISBN_13, VALID_ISBN_10],
            "cover_i": 1234567,
            "publisher": ["Prentice Hall"],
            "first_publish_year": 2008,
            "subject": ["Computer programming", "Software engineering"],
            "language": ["eng"],
            "edition_key": ["OL123456M", "OL123457M"],
        },
        {
            "key": "/works/OL98766W",
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "author_name": ["Robert C. Martin"],
            "isbn": [PAPERBACK_ISBN_13, PAPERBACK_ISBN_10],
            "cover_i": 2345678,
            "publisher": ["Prentice Hall PTR"],
            "first_publish_year": 2009,
            "subject": ["Software engineering"],
            "language": ["eng"],
            "edition_key": ["OL223456M"],
        },
        {
            "key": "/works/OL98767W",
            "title": "Clean Architecture",
            "subtitle": "A Craftsman's Guide to Software Structure and Design",
            "author_name": ["Robert C. Martin"],
            "isbn": [ARCHITECTURE_ISBN_13, ARCHITECTURE_ISBN_10],
            "cover_i": 3456789,
            "publisher": ["Prentice Hall"],
            "first_publish_year": 2017,
            "subject": ["Software architecture", "Computer programming"],
            "language": ["eng"],
            "edition_key": ["OL323456M"],
        },
    ],
}

OPEN_LIBRARY_EMPTY_RESPONSE: dict[str, Any] = {
    "numFound": 0,
    "start": 0,
    "docs": [],
}

# ---------------------------------------------------------------------------
# Multiple editions (normalised BookMetadata objects)
# ---------------------------------------------------------------------------


def _make_edition(
    *,
    title: str,
    subtitle: Optional[str],
    publisher: str,
    published_date: str,
    isbn_10: Optional[str],
    isbn_13: Optional[str],
    page_count: int,
    source: str,
    source_id: str,
    categories: Sequence[str],
    language: str = "en",
    description: Optional[str] = None,
    cover_url: Optional[str] = None,
) -> BookMetadata:
    """Small internal factory used by the editions fixture."""
    return BookMetadata(
        title=title,
        subtitle=subtitle,
        authors=["Robert C. Martin"],
        publisher=publisher,
        published_date=published_date,
        description=description,
        isbn_10=isbn_10,
        isbn_13=isbn_13,
        page_count=page_count,
        categories=list(categories),
        language=language,
        cover_url=cover_url,
        source=source,
        source_id=source_id,
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Static scraper HTML samples
# ---------------------------------------------------------------------------

GOODREADS_BOOK_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Clean Code by Robert C. Martin | Goodreads</title>
    <meta property="og:title" content="Clean Code" />
    <meta
      property="og:description"
      content="Even bad code can function. But if code isn't clean, it can bring a development organization to its knees."
    />
    <meta property="books:isbn" content="9780132350884" />
    <meta property="og:book:author" content="Robert C. Martin" />
    <meta property="og:book:publisher" content="Prentice Hall" />
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Book",
        "name": "Clean Code",
        "author": [
          {"@type": "Person", "name": "Robert C. Martin"}
        ],
        "isbn": "9780132350884",
        "publisher": {"@type": "Organization", "name": "Prentice Hall"},
        "datePublished": "2008-08-01",
        "description": "Even bad code can function. But if code isn't clean, it can bring a development organization to its knees.",
        "numberOfPages": 464,
        "url": "https://www.goodreads.com/book/show/3735293-clean-code"
      }
    </script>
  </head>
  <body>
    <h1 data-testid="bookTitle">Clean Code</h1>
    <span data-testid="name">Robert C. Martin</span>
  </body>
</html>
"""

GOODREADS_SEARCH_HTML = """<!DOCTYPE html>
<html lang="en">
  <head><title>Search results for 'Clean Code' | Goodreads</title></head>
  <body>
    <table class="tableList">
      <tr>
        <td>
          <a class="bookTitle" href="/book/show/3735293-clean-code">
            Clean Code
          </a>
        </td>
      </tr>
      <tr>
        <td>
          <a class="bookTitle" href="/book/show/7890123-clean-architecture">
            Clean Architecture
          </a>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

AMAZON_BOOK_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Clean Code: A Handbook of Agile Software Craftsmanship</title>
    <meta name="title" content="Clean Code: A Handbook of Agile Software Craftsmanship" />
    <meta name="description" content="Even bad code can function. But if code isn't clean, it can bring a development organization to its knees." />
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Clean Code: A Handbook of Agile Software Craftsmanship",
        "author": {"@type": "Person", "name": "Robert C. Martin"},
        "isbn": "9780132350884",
        "publisher": {"@type": "Organization", "name": "Prentice Hall"},
        "datePublished": "2008",
        "description": "Even bad code can function. But if code isn't clean, it can bring a development organization to its knees.",
        "numberOfPages": 464,
        "url": "https://www.amazon.com/dp/0132350884"
      }
    </script>
  </head>
  <body>
    <span id="productTitle">Clean Code: A Handbook of Agile Software Craftsmanship</span>
  </body>
</html>
"""

AMAZON_SEARCH_HTML = """<!DOCTYPE html>
<html lang="en">
  <head><title>Amazon.com : Clean Code</title></head>
  <body>
    <div data-component-type="s-search-result">
      <a class="a-link-normal s-no-outline"
         href="/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350884">
        Clean Code
      </a>
    </div>
    <div data-component-type="s-search-result">
      <a class="a-link-normal s-no-outline"
         href="/Clean-Architecture-Craftsmans-Software-Structure/dp/0134494164">
        Clean Architecture
      </a>
    </div>
  </body>
</html>
"""

MALFORMED_JSON_LD_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Broken structured data page</title>
    <meta property="og:title" content="Fallback Title From Meta Tags" />
    <script type="application/ld+json">
      { this is not valid JSON at all, and it is missing every quote }
    </script>
  </head>
  <body>
    <h1>Fallback Title From Body</h1>
  </body>
</html>
"""

EMPTY_HTML = "<!DOCTYPE html><html><head></head><body></body></html>"

AMAZON_CAPTCHA_HTML = """<!DOCTYPE html>
<html lang="en">
  <head><title>Amazon.com</title></head>
  <body>
    <h1>Enter the characters you see below</h1>
    <p>Sorry, we just need to make sure you're not a robot.</p>
    <form action="/errors/validateCaptcha" method="get">
      <input type="text" name="field-keywords" />
      <input type="submit" value="Continue shopping" />
    </form>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Internal HTTP mocking helpers
# ---------------------------------------------------------------------------


class MockHTTPResponse:
    """A small description of a response served by :class:`MockHTTPRouter`."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        json_data: Any = None,
        text: Optional[str] = None,
        content: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.json_data = json_data
        self.text = text
        self.content = content
        self.headers = dict(headers or {})

    def to_httpx(self, request: httpx.Request) -> httpx.Response:
        """Render this description as an ``httpx.Response``."""
        headers = dict(self.headers)
        if self.content is not None:
            body = self.content
        elif self.json_data is not None:
            body = json.dumps(self.json_data).encode("utf-8")
            headers.setdefault("content-type", "application/json")
        elif self.text is not None:
            body = self.text.encode("utf-8")
        else:
            body = b""
        return httpx.Response(
            status_code=self.status_code,
            headers=headers,
            content=body,
            request=request,
        )


class _HTTPRoute:
    """A URL matcher plus the response (or exception) to serve."""

    def __init__(
        self,
        matcher: Callable[[httpx.Request], bool],
        *,
        response: Optional[MockHTTPResponse] = None,
        callback: Optional[Callable[[httpx.Request], Any]] = None,
        exception: Optional[BaseException] = None,
        times: Optional[int] = None,
        priority: int = 0,
    ) -> None:
        self.matcher = matcher
        self.response = response
        self.callback = callback
        self.exception = exception
        self.times = times
        self.remaining = times
        self.priority = priority

    def matches(self, request: httpx.Request) -> bool:
        if self.remaining is not None and self.remaining <= 0:
            return False
        if not self.matcher(request):
            return False
        if self.remaining is not None:
            self.remaining -= 1
        return True


class MockHTTPRouter:
    """Tiny, dependency-free stand-in for ``pytest-httpx``/``respx``.

    The router builds an ``httpx.MockTransport`` which can be installed on a
    provider instance or class.  Every request is recorded in
    :attr:`requests` for assertions.  Unmatched requests raise
    ``AssertionError``, guaranteeing that no live HTTP call can silently slip
    through.

    Example
    -------
    >>> def test_search(mock_http, monkeypatch):
    ...     mock_http.add_response(url="/volumes", json={"items": []})
    ...     mock_http.install_on_class(GoogleBooksProvider, monkeypatch)
    ...     provider = GoogleBooksProvider()
    ...     # ... await provider.search_by_isbn(...)
    """

    def __init__(self) -> None:
        self._routes: list[_HTTPRoute] = []
        self.requests: list[httpx.Request] = []

    # -- route registration -------------------------------------------------

    def add_response(
        self,
        url: Optional[str | re.Pattern[str]] = None,
        *,
        path: Optional[str] = None,
        method: str = "GET",
        json: Any = None,
        text: Optional[str] = None,
        content: Optional[bytes] = None,
        status_code: int = 200,
        headers: Optional[Mapping[str, str]] = None,
        callback: Optional[Callable[[httpx.Request], Any]] = None,
        times: Optional[int] = None,
        priority: int = 0,
    ) -> "MockHTTPRouter":
        """Register a JSON/text/bytes response for a matching request.

        ``priority`` controls precedence when several routes match the same
        request.  Higher values win; equal values are tried in registration
        order.  The provider mock fixtures register fallback responses with
        ``priority=-1`` so a test can override them by registering its own
        response (the default priority) without clearing the router.
        """
        response = MockHTTPResponse(
            status_code,
            json_data=json,
            text=text,
            content=content,
            headers=headers,
        )
        self._routes.append(
            _HTTPRoute(
                self._make_matcher(url=url, path=path, method=method),
                response=response,
                callback=callback,
                times=times,
                priority=priority,
            )
        )
        return self

    def add_exception(
        self,
        exception: BaseException,
        url: Optional[str | re.Pattern[str]] = None,
        *,
        path: Optional[str] = None,
        method: str = "GET",
        times: Optional[int] = None,
        priority: int = 0,
    ) -> "MockHTTPRouter":
        """Register an exception (e.g. ``httpx.TimeoutException``) to raise."""
        self._routes.append(
            _HTTPRoute(
                self._make_matcher(url=url, path=path, method=method),
                exception=exception,
                times=times,
                priority=priority,
            )
        )
        return self

    # ``add_error`` is a more discoverable alias for connection/timeout tests.
    add_error = add_exception

    def add_callback(
        self,
        callback: Callable[[httpx.Request], Any],
        url: Optional[str | re.Pattern[str]] = None,
        *,
        path: Optional[str] = None,
        method: str = "GET",
        times: Optional[int] = None,
        priority: int = 0,
    ) -> "MockHTTPRouter":
        """Register a callback that produces the response dynamically."""
        self._routes.append(
            _HTTPRoute(
                self._make_matcher(url=url, path=path, method=method),
                callback=callback,
                times=times,
                priority=priority,
            )
        )
        return self

    # -- matching -----------------------------------------------------------

    @staticmethod
    def _make_matcher(
        *,
        url: Optional[str | re.Pattern[str]],
        path: Optional[str],
        method: str,
    ) -> Callable[[httpx.Request], bool]:
        def matcher(request: httpx.Request) -> bool:
            if method and request.method.upper() != method.upper():
                return False
            full_url = str(request.url)
            if url is not None:
                if hasattr(url, "search"):
                    return bool(url.search(full_url))  # type: ignore[union-attr]
                return str(url) in full_url
            if path is not None:
                return path in request.url.path
            return True

        return matcher

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # Higher-priority routes are tried first; equal priorities keep
        # registration order (stable sort).
        routes = sorted(
            enumerate(self._routes),
            key=lambda pair: (-pair[1].priority, pair[0]),
        )
        for _, route in routes:
            if not route.matches(request):
                continue
            if route.exception is not None:
                raise route.exception
            if route.callback is not None:
                result = route.callback(request)
                if isinstance(result, httpx.Response):
                    return result
                if isinstance(result, MockHTTPResponse):
                    return result.to_httpx(request)
                if isinstance(result, Mapping):
                    return MockHTTPResponse(json_data=dict(result)).to_httpx(request)
                raise TypeError(
                    "MockHTTPRouter callback must return httpx.Response, "
                    "MockHTTPResponse, or a mapping, got "
                    f"{type(result).__name__}"
                )
            assert route.response is not None
            return route.response.to_httpx(request)

        raise AssertionError(
            "Unexpected live HTTP request blocked by MockHTTPRouter: "
            f"{request.method} {request.url}. "
            "Register a response with mock_http.add_response(...)."
        )

    # -- installation -------------------------------------------------------

    def client(self) -> httpx.AsyncClient:
        """Return a new async client backed by this router's transport."""
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler),
            follow_redirects=True,
        )

    def install(
        self,
        provider: Any,
        monkeypatch: Optional[pytest.MonkeyPatch] = None,
    ) -> Any:
        """Patch ``provider._build_client`` to use this router."""

        def _build_client() -> httpx.AsyncClient:
            return self.client()

        if monkeypatch is not None:
            monkeypatch.setattr(provider, "_build_client", _build_client)
        else:
            provider._build_client = _build_client  # type: ignore[method-assign]
        return provider

    def install_on_class(
        self,
        provider_cls: type,
        monkeypatch: pytest.MonkeyPatch,
    ) -> "MockHTTPRouter":
        """Patch a provider *class* so future instances use this router."""

        def _build_client(_self: Any) -> httpx.AsyncClient:
            return self.client()

        monkeypatch.setattr(provider_cls, "_build_client", _build_client)
        return self

    # -- assertions ---------------------------------------------------------

    @property
    def request_urls(self) -> list[str]:
        """Return the string URL of every request seen by the router."""
        return [str(req.url) for req in self.requests]

    @property
    def request_count(self) -> int:
        """Return the number of requests seen by the router."""
        return len(self.requests)

    def get_requests(self) -> list[httpx.Request]:
        """Return all recorded requests (pytest-httpx-style helper)."""
        return list(self.requests)

    def get_request(self) -> httpx.Request:
        """Return the most recent recorded request."""
        assert self.requests, "No HTTP requests were recorded"
        return self.requests[-1]

    def clear(self) -> None:
        """Forget recorded requests and registered routes."""
        self._routes.clear()
        self.requests.clear()


def make_requests_response(
    *,
    status_code: int = 200,
    json: Any = None,
    text: Optional[str] = None,
    content: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    url: str = "https://example.test/",
) -> requests.Response:
    """Build a real ``requests.Response`` without touching the network."""
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response.headers.update(headers or {})
    if content is not None:
        response._content = content
    elif json is not None:
        response._content = json_module_dumps(json).encode("utf-8")
        response.headers.setdefault("content-type", "application/json")
    else:
        response._content = (text or "").encode("utf-8")
    response.encoding = "utf-8"
    return response


def json_module_dumps(value: Any) -> str:
    """Indirection so ``json`` parameters can shadow the module name."""
    return json.dumps(value)


class _RequestsRoute:
    def __init__(
        self,
        pattern: Optional[str | re.Pattern[str]],
        *,
        response: Optional[requests.Response] = None,
        exception: Optional[BaseException] = None,
        times: Optional[int] = None,
    ) -> None:
        self.pattern = pattern
        self.response = response
        self.exception = exception
        self.times = times
        self.remaining = times

    def matches(self, url: str) -> bool:
        if self.remaining is not None and self.remaining <= 0:
            return False
        if self.pattern is None:
            matched = True
        elif hasattr(self.pattern, "search"):
            matched = bool(self.pattern.search(url))  # type: ignore[union-attr]
        else:
            matched = str(self.pattern) in url
        if matched and self.remaining is not None:
            self.remaining -= 1
        return matched


class MockRequestsRouter:
    """Minimal ``requests.get`` mock used by scraper tests.

    Unlike :class:`MockHTTPRouter`, this patches the synchronous ``requests``
    library used by :mod:`app.books.metadata.providers.scraper`.  It also
    neutralises ``robots.txt`` checks and per-domain rate-limit sleeps so
    scraper tests stay fast and deterministic.
    """

    def __init__(self) -> None:
        self._routes: list[_RequestsRoute] = []
        self.requests: list[dict[str, Any]] = []
        self.robots_allowed = True

    def add_response(
        self,
        url: Optional[str | re.Pattern[str]] = None,
        *,
        status_code: int = 200,
        json: Any = None,
        text: Optional[str] = None,
        content: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        times: Optional[int] = None,
    ) -> "MockRequestsRouter":
        response = make_requests_response(
            status_code=status_code,
            json=json,
            text=text,
            content=content,
            headers=headers,
            url=url if isinstance(url, str) else "https://example.test/",
        )
        self._routes.append(
            _RequestsRoute(url, response=response, times=times)
        )
        return self

    def add_exception(
        self,
        exception: BaseException,
        url: Optional[str | re.Pattern[str]] = None,
        *,
        times: Optional[int] = None,
    ) -> "MockRequestsRouter":
        self._routes.append(
            _RequestsRoute(url, exception=exception, times=times)
        )
        return self

    add_error = add_exception

    def handler(self, url: str, **kwargs: Any) -> requests.Response:
        self.requests.append({"url": url, "kwargs": kwargs})
        for route in self._routes:
            if not route.matches(url):
                continue
            if route.exception is not None:
                raise route.exception
            assert route.response is not None
            return route.response
        raise AssertionError(
            "Unexpected live HTTP request blocked by MockRequestsRouter: "
            f"GET {url}. Register a response with "
            "mock_scraper_http.add_response(...)."
        )

    def install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        allow_robots: bool = True,
    ) -> "MockRequestsRouter":
        """Patch ``requests.get``, robots checks, and rate-limit sleeps."""
        self.robots_allowed = allow_robots
        monkeypatch.setattr(
            "app.books.metadata.providers.scraper.requests.get",
            self.handler,
        )
        monkeypatch.setattr(
            MetadataScraper,
            "_check_robots",
            lambda _self, _url: self.robots_allowed,
        )
        monkeypatch.setattr(
            "app.books.metadata.providers.scraper.time.sleep",
            lambda *_args, **_kwargs: None,
        )
        return self

    @property
    def requested_urls(self) -> list[str]:
        return [entry["url"] for entry in self.requests]

    @property
    def request_count(self) -> int:
        return len(self.requests)


# ---------------------------------------------------------------------------
# Autouse fixtures: isolation, speed, and no-live-network guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_metadata_environment(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Keep the metadata SQLite cache out of the real application database.

    The metadata service constructs ``MetadataCache()`` without an explicit
    path, which normally resolves to ``settings.DATABASE_PATH.parent /
    "metadata_cache.db"``.  Pointing ``DATABASE_PATH`` at ``tmp_path`` gives
    every test a private cache.
    """
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "test.db")
    monkeypatch.setattr(settings, "GOOGLE_BOOKS_API_KEY", None)
    monkeypatch.setattr(settings, "SCRAPING_ENABLED", False)
    monkeypatch.setattr(settings, "METADATA_STRATEGY", "HYBRID")
    return tmp_path


@pytest.fixture(autouse=True)
def _fast_metadata_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse provider retry back-offs so retry tests don't sleep."""

    async def _instant_sleep(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


@pytest.fixture(autouse=True)
def _block_live_metadata_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test forgets to mock an external HTTP call.

    The provider-specific mock fixtures (``mock_google_books_http`` etc.)
    override these guards for the test that requests them.
    """

    def _blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            "Live HTTP request blocked in metadata tests: "
            f"{request.method} {request.url}. "
            "Use mock_http / mock_google_books_http / mock_open_library_http."
        )

    def _blocked_async_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(_blocked_handler),
            follow_redirects=True,
        )

    monkeypatch.setattr(
        GoogleBooksProvider,
        "_build_client",
        lambda _self: _blocked_async_client(),
    )
    monkeypatch.setattr(
        OpenLibraryProvider,
        "_build_client",
        lambda _self: _blocked_async_client(),
    )

    def _blocked_requests_get(url: str, **_kwargs: Any) -> requests.Response:
        raise AssertionError(
            "Live HTTP request blocked in metadata tests: "
            f"GET {url}. Use mock_scraper_http or disable scraping."
        )

    monkeypatch.setattr(
        "app.books.metadata.providers.scraper.requests.get",
        _blocked_requests_get,
    )

    def _blocked_robots(_self: Any, url: str) -> bool:
        raise AssertionError(
            "Live robots.txt check blocked in metadata tests: "
            f"{url}. Use mock_scraper_http."
        )

    monkeypatch.setattr(MetadataScraper, "_check_robots", _blocked_robots)
    monkeypatch.setattr(
        "app.books.metadata.providers.scraper.time.sleep",
        lambda *_args, **_kwargs: None,
    )


# ---------------------------------------------------------------------------
# Google Books payload fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def google_books_response() -> dict[str, Any]:
    """A complete, valid Google Books ``/volumes`` response (one volume)."""
    return copy.deepcopy(GOOGLE_BOOKS_RESPONSE)


# Common alias used by provider and endpoint tests.
@pytest.fixture
def google_books_api_response(google_books_response: dict[str, Any]) -> dict[str, Any]:
    """Alias for :func:`google_books_response`."""
    return google_books_response


@pytest.fixture
def google_books_volume(google_books_response: dict[str, Any]) -> dict[str, Any]:
    """The single ``volume`` object inside :func:`google_books_response`."""
    return copy.deepcopy(google_books_response["items"][0])


@pytest.fixture
def google_books_volume_info(
    google_books_volume: dict[str, Any],
) -> dict[str, Any]:
    """The ``volumeInfo`` object from :func:`google_books_volume`."""
    return copy.deepcopy(google_books_volume["volumeInfo"])


@pytest.fixture
def google_books_candidate(google_books_volume: dict[str, Any]) -> dict[str, Any]:
    """The provider's raw candidate dict for the standard Google Books volume.

    This is the shape passed to ``MetadataNormalizer.normalize_candidate``.
    """
    provider = GoogleBooksProvider()
    return copy.deepcopy(provider._extract_volume_info(google_books_volume))


@pytest.fixture
def google_books_multi_response() -> dict[str, Any]:
    """A Google Books response containing three editions of the same book."""
    return copy.deepcopy(GOOGLE_BOOKS_MULTI_RESPONSE)


@pytest.fixture
def google_books_title_search_response(
    google_books_multi_response: dict[str, Any],
) -> dict[str, Any]:
    """Alias for a title/author search returning multiple candidates."""
    return google_books_multi_response


@pytest.fixture
def google_books_empty_response() -> dict[str, Any]:
    """A valid Google Books response with zero results."""
    return copy.deepcopy(GOOGLE_BOOKS_EMPTY_RESPONSE)


@pytest.fixture
def google_books_missing_fields_response() -> dict[str, Any]:
    """Google Books payload with no authors, ISBNs, or cover image."""
    return copy.deepcopy(GOOGLE_BOOKS_MISSING_FIELDS_RESPONSE)


@pytest.fixture
def google_books_malformed_response() -> str:
    """An invalid JSON body as returned by a broken Google Books endpoint."""
    return '{"kind": "books#volumes", "items": [{"volumeInfo": {"title": '


# ---------------------------------------------------------------------------
# Open Library payload fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def open_library_isbn_response() -> dict[str, Any]:
    """A complete, valid Open Library ``/isbn/{isbn}.json`` edition payload."""
    return copy.deepcopy(OPEN_LIBRARY_ISBN_RESPONSE)


@pytest.fixture
def open_library_response(
    open_library_isbn_response: dict[str, Any],
) -> dict[str, Any]:
    """Alias for the Open Library ISBN-lookup payload."""
    return open_library_isbn_response


@pytest.fixture
def open_library_edition_response(
    open_library_isbn_response: dict[str, Any],
) -> dict[str, Any]:
    """Alias for the Open Library edition payload."""
    return open_library_isbn_response


@pytest.fixture
def open_library_candidate() -> dict[str, Any]:
    """A parsed Open Library candidate in ``BookMetadata``-compatible shape.

    Unlike :func:`open_library_isbn_response` (the raw API payload, whose
    authors are ``{"key": ...}`` objects), this fixture contains resolved
    author names and is ready for the normalizer / matcher.
    """
    return {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "published_date": "2008",
        "description": (
            "Even bad code can function. But if code isn't clean, it can "
            "bring a development organization to its knees."
        ),
        "isbn_10": VALID_ISBN_10,
        "isbn_13": VALID_ISBN_13,
        "page_count": 464,
        "categories": [
            "Computer programming",
            "Software engineering",
            "Agile software development",
        ],
        "language": "eng",
        "cover_url": "https://covers.openlibrary.org/b/id/1234567-L.jpg",
        "source": "open_library",
        "source_id": "OL123456M",
        "confidence": 0.0,
    }


@pytest.fixture
def provider_candidates(
    google_books_candidate: dict[str, Any],
    open_library_candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Provider-ready candidate dicts keyed by provider name."""
    return {
        "google_books": google_books_candidate,
        "open_library": open_library_candidate,
    }


@pytest.fixture
def open_library_search_response() -> dict[str, Any]:
    """An Open Library ``/search.json`` response with several matching docs."""
    return copy.deepcopy(OPEN_LIBRARY_SEARCH_RESPONSE)


@pytest.fixture
def open_library_author_response() -> dict[str, Any]:
    """An Open Library ``/authors/{key}.json`` payload used for name lookup."""
    return copy.deepcopy(OPEN_LIBRARY_AUTHOR_RESPONSE)


@pytest.fixture
def open_library_work_response() -> dict[str, Any]:
    """An Open Library ``/works/{key}.json`` payload."""
    return copy.deepcopy(OPEN_LIBRARY_WORK_RESPONSE)


@pytest.fixture
def open_library_empty_response() -> dict[str, Any]:
    """A valid Open Library search response with zero results."""
    return copy.deepcopy(OPEN_LIBRARY_EMPTY_RESPONSE)


@pytest.fixture
def open_library_malformed_response() -> str:
    """An invalid JSON body as returned by a broken Open Library endpoint."""
    return "<html><body>429 Too Many Requests</body></html>"


# ---------------------------------------------------------------------------
# Multi-edition and conflicting-provider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multiple_editions() -> list[BookMetadata]:
    """Three different editions of *Clean Code* by Robert C. Martin.

    The editions share a title and author but differ in ISBN, publisher,
    publication date, page count, and source.  They are ideal for matcher and
    deduplication tests: none of them should be treated as the same edition
    merely because the title/author match.
    """
    return [
        _make_edition(
            title="Clean Code",
            subtitle="A Handbook of Agile Software Craftsmanship",
            publisher="Prentice Hall",
            published_date="2008-08-01",
            isbn_10=VALID_ISBN_10,
            isbn_13=VALID_ISBN_13,
            page_count=464,
            source="google_books",
            source_id="gb-clean-code-hardcover",
            categories=["Computers", "Software Engineering"],
            description=(
                "Even bad code can function. But if code isn't clean, it can "
                "bring a development organization to its knees."
            ),
            cover_url=(
                "https://books.google.com/books/content?id=zyTCAlFPjgYC"
                "&printsec=frontcover&img=1&zoom=1"
            ),
        ),
        _make_edition(
            title="Clean Code",
            subtitle="A Handbook of Agile Software Craftsmanship",
            publisher="Prentice Hall PTR",
            published_date="2009-07-01",
            isbn_10=PAPERBACK_ISBN_10,
            isbn_13=PAPERBACK_ISBN_13,
            page_count=464,
            source="open_library",
            source_id="OL223456M",
            categories=["Software engineering"],
            description="The paperback printing of the same handbook.",
            cover_url="https://covers.openlibrary.org/b/id/2345678-L.jpg",
        ),
        _make_edition(
            title="Clean Code",
            subtitle="A Handbook of Agile Software Craftsmanship",
            publisher="Pearson",
            published_date="2012-03-01",
            isbn_10=EBOOK_ISBN_10,
            isbn_13=EBOOK_ISBN_13,
            page_count=480,
            source="google_books",
            source_id="gb-clean-code-ebook",
            categories=["Computers", "Programming"],
            description="The ebook edition of Clean Code.",
            cover_url=(
                "https://books.google.com/books/content?id=ebook-clean-code"
                "&printsec=frontcover&img=1&zoom=1"
            ),
        ),
    ]


@pytest.fixture
def multiple_editions_raw(
    multiple_editions: list[BookMetadata],
) -> list[dict[str, Any]]:
    """Raw dict form of :func:`multiple_editions` for normalizer tests."""
    return [edition.model_dump(exclude_none=False) for edition in multiple_editions]


@pytest.fixture
def multiple_editions_same_book(
    multiple_editions: list[BookMetadata],
) -> list[BookMetadata]:
    """Alias for :func:`multiple_editions`."""
    return multiple_editions


@pytest.fixture
def conflicting_provider_responses() -> dict[str, dict[str, Any]]:
    """Same book/ISBN from two providers with deliberately conflicting fields.

    Google Books says *Prentice Hall / 464 pages / 2008*; Open Library says
    *Prentice Hall PTR / 480 pages / 2009*.  The ISBN and title/author agree,
    so the service must preserve the conflict rather than silently
    overwriting one provider with the other.
    """
    return {
        "google_books": {
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall",
            "published_date": "2008-08-01",
            "description": "Google Books description of the book.",
            "isbn_10": VALID_ISBN_10,
            "isbn_13": VALID_ISBN_13,
            "page_count": 464,
            "categories": ["Computers", "Software Engineering"],
            "language": "en",
            "cover_url": (
                "https://books.google.com/books/content?id=zyTCAlFPjgYC"
                "&printsec=frontcover&img=1&zoom=1"
            ),
            "source": "google_books",
            "source_id": "zyTCAlFPjgYC",
            "confidence": 0.0,
        },
        "open_library": {
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall PTR",
            "published_date": "2009",
            "description": "Open Library description of the book.",
            "isbn_10": VALID_ISBN_10,
            "isbn_13": VALID_ISBN_13,
            "page_count": 480,
            "categories": ["Software engineering"],
            "language": "eng",
            "cover_url": "https://covers.openlibrary.org/b/id/1234567-L.jpg",
            "source": "open_library",
            "source_id": "OL123456M",
            "confidence": 0.0,
        },
    }


@pytest.fixture
def conflicting_isbn_provider_responses() -> dict[str, dict[str, Any]]:
    """Same title/author from two providers, but different ISBNs.

    This models two different editions being returned for one query and is
    used to verify that an ISBN conflict reduces confidence and preserves the
    disagreement.
    """
    return {
        "google_books": {
            "title": "Clean Code",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall",
            "published_date": "2008-08-01",
            "isbn_10": VALID_ISBN_10,
            "isbn_13": VALID_ISBN_13,
            "page_count": 464,
            "categories": ["Computers"],
            "language": "en",
            "cover_url": None,
            "source": "google_books",
            "source_id": "zyTCAlFPjgYC",
            "confidence": 0.0,
        },
        "open_library": {
            "title": "Clean Code",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall PTR",
            "published_date": "2009",
            "isbn_10": PAPERBACK_ISBN_10,
            "isbn_13": PAPERBACK_ISBN_13,
            "page_count": 464,
            "categories": ["Software engineering"],
            "language": "eng",
            "cover_url": None,
            "source": "open_library",
            "source_id": "OL223456M",
            "confidence": 0.0,
        },
    }


@pytest.fixture
def complementary_provider_responses() -> dict[str, dict[str, Any]]:
    """Two providers whose records describe the same edition with complementary data.

    Google Books supplies the title, authors, publisher, ISBNs, and date;
    Open Library supplies the description, categories, language, and cover.
    """
    return {
        "google_books": {
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall",
            "published_date": "2008-08-01",
            "description": None,
            "isbn_10": VALID_ISBN_10,
            "isbn_13": VALID_ISBN_13,
            "page_count": 464,
            "categories": [],
            "language": None,
            "cover_url": None,
            "source": "google_books",
            "source_id": "zyTCAlFPjgYC",
            "confidence": 0.0,
        },
        "open_library": {
            "title": "Clean Code",
            "subtitle": "A Handbook of Agile Software Craftsmanship",
            "authors": ["Robert C. Martin"],
            "publisher": "Prentice Hall",
            "published_date": "2008",
            "description": "Even bad code can function...",
            "isbn_10": VALID_ISBN_10,
            "isbn_13": VALID_ISBN_13,
            "page_count": None,
            "categories": ["Computer programming", "Software engineering"],
            "language": "eng",
            "cover_url": "https://covers.openlibrary.org/b/id/1234567-L.jpg",
            "source": "open_library",
            "source_id": "OL123456M",
            "confidence": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Empty / malformed / invalid input fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_provider_response() -> dict[str, Any]:
    """A generic "no results" payload understood by all provider shapes."""
    return {
        "items": [],
        "totalItems": 0,
        "docs": [],
        "numFound": 0,
        "candidates": [],
    }


@pytest.fixture
def empty_provider_responses() -> dict[str, Any]:
    """Per-provider empty payloads, keyed by provider name."""
    return {
        "google_books": copy.deepcopy(GOOGLE_BOOKS_EMPTY_RESPONSE),
        "open_library": copy.deepcopy(OPEN_LIBRARY_EMPTY_RESPONSE),
        "generic": {
            "items": [],
            "totalItems": 0,
            "docs": [],
            "numFound": 0,
            "candidates": [],
        },
    }


@pytest.fixture
def malformed_provider_response() -> str:
    """Invalid JSON returned with HTTP 200 by a misbehaving provider."""
    return '{"items": [{"volumeInfo": {"title": "Truncated response"'


@pytest.fixture
def malformed_provider_responses() -> dict[str, Any]:
    """A collection of malformed payload shapes for provider edge-case tests.

    Keys:
        ``invalid_json``: truncated JSON (the common malformed case).
        ``html_error_page``: an HTML error page returned with HTTP 200.
        ``missing_fields``: valid JSON with an empty item.
        ``missing_volume_info``: a Google Books item without ``volumeInfo``.
        ``wrong_types``: fields with the wrong JSON types.
    """
    return {
        "invalid_json": '{"items": [{"volumeInfo": {"title": "Truncated response"',
        "html_error_page": "<html><body>503 Service Unavailable</body></html>",
        "missing_fields": {"items": [{}]},
        "missing_volume_info": {"items": [{"id": "no-volume-info"}]},
        "wrong_types": {
            "items": "this-should-be-a-list",
            "totalItems": "not-a-number",
        },
    }


#: Valid ISBN inputs (formatted and unformatted) from the roadmap examples.
VALID_ISBN_CASES: list[dict[str, str]] = [
    {"isbn": "9780135957059", "format": "isbn13"},
    {"isbn": "978-0-13-595705-9", "format": "isbn13_hyphenated"},
    {"isbn": "0-321-12521-5", "format": "isbn10_hyphenated"},
    {"isbn": "0321125215", "format": "isbn10"},
    {"isbn": VALID_ISBN_13, "format": "isbn13"},
    {"isbn": VALID_ISBN_10, "format": "isbn10"},
    {"isbn": "978 0 13 595705 9", "format": "isbn13_spaced"},
    {"isbn": "0 321 12521 5", "format": "isbn10_spaced"},
]

#: Invalid ISBN inputs covering length, characters, and checksum failures.
INVALID_ISBN_CASES: list[dict[str, str]] = [
    {"isbn": "", "reason": "empty"},
    {"isbn": "123", "reason": "too_short"},
    {"isbn": "1234567890", "reason": "invalid_isbn10_checksum"},
    {"isbn": "9780135957058", "reason": "invalid_isbn13_checksum"},
    {"isbn": "0321125214", "reason": "invalid_isbn10_checksum"},
    {"isbn": "978-0-13-595705-8", "reason": "invalid_isbn13_checksum_hyphenated"},
    {"isbn": "97801359570590", "reason": "too_long"},
    {"isbn": "12345678901234", "reason": "too_long"},
    {"isbn": "abcdefghij", "reason": "non_digit_isbn10"},
    {"isbn": "97801X5957059", "reason": "x_in_wrong_position"},
    {"isbn": "X123456789", "reason": "isbn10_x_in_wrong_position"},
    {"isbn": "-", "reason": "only_separator"},
]


@pytest.fixture
def valid_isbns() -> list[str]:
    """Valid ISBN strings, including hyphenated and spaced variants."""
    return [case["isbn"] for case in VALID_ISBN_CASES]


@pytest.fixture
def valid_isbn_cases() -> list[dict[str, str]]:
    """Valid ISBNs with a descriptive ``format`` label for parametrisation."""
    return copy.deepcopy(VALID_ISBN_CASES)


@pytest.fixture
def invalid_isbns() -> list[str]:
    """Invalid ISBN strings suitable for ``@pytest.mark.parametrize``."""
    return [case["isbn"] for case in INVALID_ISBN_CASES]


@pytest.fixture
def invalid_isbn_cases() -> list[dict[str, str]]:
    """Invalid ISBNs with a descriptive ``reason`` label."""
    return copy.deepcopy(INVALID_ISBN_CASES)


@pytest.fixture
def valid_isbn() -> str:
    """A single valid ISBN-13."""
    return VALID_ISBN_13


@pytest.fixture
def valid_isbn_10() -> str:
    """A single valid ISBN-10."""
    return VALID_ISBN_10


@pytest.fixture
def valid_isbn_10_with_x() -> str:
    """A valid ISBN-10 whose checksum character is ``X``.

    Kept separate from :func:`valid_isbns` because not every implementation
    path may support the roman-numeral check digit yet.
    """
    return "097522980X"


@pytest.fixture
def invalid_isbn() -> str:
    """A single invalid ISBN-13 (wrong checksum)."""
    return "9780135957058"


# ---------------------------------------------------------------------------
# Authentication fixtures
# ---------------------------------------------------------------------------
# ``auth_headers`` already exists in ``app/books/tests/conftest.py``, but
# pytest scopes conftest fixtures to their own directory subtree.  Metadata
# tests live in a sibling directory, so the equivalent helpers are recreated
# here on top of the shared root ``client`` fixture.


@pytest.fixture
def valid_user_credentials() -> dict[str, str]:
    """Unique credentials for a throw-away user account."""
    return {
        "username": f"metadata-user-{uuid.uuid4().hex[:10]}",
        "password": "secret123",
    }


@pytest.fixture
def authenticated_user(
    client: TestClient,
    valid_user_credentials: dict[str, str],
) -> dict[str, Any]:
    """Register and log in a user, returning credentials + JWT metadata.

    Returns a dict containing ``username``, ``password``, ``id``,
    ``access_token``, ``refresh_token``, ``token_type``, ``user`` (the
    registration response body), and ``headers`` (ready for
    ``Authorization``).
    """
    register_response = client.post(
        "/v1/auth/register",
        json=valid_user_credentials,
    )
    assert register_response.status_code == 201, (
        f"User registration failed: {register_response.text}"
    )
    registered_user = register_response.json()

    login_response = client.post(
        "/v1/auth/login",
        json=valid_user_credentials,
    )
    assert login_response.status_code == 200, (
        f"Login failed: {login_response.text}"
    )
    tokens = login_response.json()

    access_token = tokens["access_token"]
    return {
        **valid_user_credentials,
        "id": registered_user["id"],
        "user": registered_user,
        "access_token": access_token,
        "refresh_token": tokens["refresh_token"],
        "token_type": tokens.get("token_type", "bearer"),
        "headers": {"Authorization": f"Bearer {access_token}"},
    }


@pytest.fixture
def valid_authenticated_user(
    authenticated_user: dict[str, Any],
) -> dict[str, Any]:
    """Alias for :func:`authenticated_user`."""
    return authenticated_user


@pytest.fixture
def auth_headers(authenticated_user: dict[str, Any]) -> dict[str, str]:
    """``{"Authorization": "Bearer <token>"}`` for the authenticated user."""
    return dict(authenticated_user["headers"])


@pytest.fixture
def authenticated_client(
    client: TestClient,
    auth_headers: dict[str, str],
) -> TestClient:
    """A ``TestClient`` whose default headers include the JWT bearer token."""
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def unauthenticated_client(client: TestClient) -> TestClient:
    """A ``TestClient`` with no ``Authorization`` header.

    Note: do not request this fixture together with ``authenticated_client``
    in the same test, since both wrap the same per-test ``client`` instance.
    """
    client.headers.pop("Authorization", None)
    return client


@pytest.fixture
def metadata_search_payload() -> dict[str, str]:
    """A representative authenticated metadata-search request body."""
    return {
        "isbn": VALID_ISBN_13,
        "title": "Clean Code",
        "author": "Robert C. Martin",
    }


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_cache_db_path(tmp_path: Any) -> Any:
    """Path to an isolated, per-test SQLite metadata cache database."""
    return tmp_path / "metadata_cache.db"


@pytest.fixture
def temp_cache_db(temp_cache_db_path: Any) -> Any:
    """Alias for :func:`temp_cache_db_path`."""
    return temp_cache_db_path


@pytest.fixture
def metadata_cache(temp_cache_db_path: Any) -> MetadataCache:
    """A ``MetadataCache`` backed by the per-test temporary SQLite file."""
    return MetadataCache(db_path=temp_cache_db_path)


@pytest.fixture
def populated_cache(
    metadata_cache: MetadataCache,
    sample_book_metadata: BookMetadata,
) -> MetadataCache:
    """A cache pre-populated with one candidate for the Clean Code ISBN."""
    metadata_cache.set(
        isbn=VALID_ISBN_13,
        title=None,
        author=None,
        candidates=[sample_book_metadata],
        provider="google_books",
    )
    return metadata_cache


# ---------------------------------------------------------------------------
# HTTP mocking fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http() -> MockHTTPRouter:
    """An empty async HTTP mock router for provider tests.

    Register responses with ``mock_http.add_response(...)`` and install the
    router with ``mock_http.install(provider, monkeypatch)`` or
    ``mock_http.install_on_class(ProviderClass, monkeypatch)``.
    """
    return MockHTTPRouter()


@pytest.fixture
def httpx_mock(mock_http: MockHTTPRouter) -> MockHTTPRouter:
    """Alias for :func:`mock_http` (pytest-httpx-style usage)."""
    return mock_http


@pytest.fixture
def mock_httpx(mock_http: MockHTTPRouter) -> MockHTTPRouter:
    """Alias for :func:`mock_http`."""
    return mock_http


@pytest.fixture
def mock_google_books_http(
    mock_http: MockHTTPRouter,
    monkeypatch: pytest.MonkeyPatch,
    google_books_response: dict[str, Any],
) -> MockHTTPRouter:
    """Mock Google Books with the standard valid search response installed.

    The standard response is registered as a low-priority fallback.  A test
    can override it (for example to return a 429 followed by a 200) by
    registering its own route on the returned router::

        mock_google_books_http.add_response(
            url="/volumes", status_code=429, text="rate limited", times=1
        )
        mock_google_books_http.add_response(
            url="/volumes", json=google_books_response
        )
    """
    mock_http.add_response(
        url="/volumes", json=google_books_response, priority=-1
    )
    mock_http.install_on_class(GoogleBooksProvider, monkeypatch)
    return mock_http


@pytest.fixture
def mock_open_library_http(
    mock_http: MockHTTPRouter,
    monkeypatch: pytest.MonkeyPatch,
    open_library_isbn_response: dict[str, Any],
    open_library_search_response: dict[str, Any],
    open_library_author_response: dict[str, Any],
) -> MockHTTPRouter:
    """Mock Open Library with ISBN, search, and author endpoints installed.

    Defaults are low-priority fallbacks; register a route on the returned
    router to override a specific endpoint for the current test.
    """
    mock_http.add_response(
        path="/isbn/", json=open_library_isbn_response, priority=-1
    )
    mock_http.add_response(
        path="/search.json", json=open_library_search_response, priority=-1
    )
    mock_http.add_response(
        path="/authors/", json=open_library_author_response, priority=-1
    )
    mock_http.install_on_class(OpenLibraryProvider, monkeypatch)
    return mock_http


@pytest.fixture
def mock_metadata_providers(
    mock_google_books_http: MockHTTPRouter,
    mock_open_library_http: MockHTTPRouter,
) -> dict[str, MockHTTPRouter]:
    """Mock both API providers on a single shared router.

    Returns a mapping ``{"google_books": router, "open_library": router}``;
    both values are the same router because the two fixtures share the
    function-scoped ``mock_http`` instance.
    """
    return {
        "google_books": mock_google_books_http,
        "open_library": mock_open_library_http,
    }


@pytest.fixture
def mock_scraper_http(monkeypatch: pytest.MonkeyPatch) -> MockRequestsRouter:
    """Mock ``requests.get`` for scraper tests and allow ``robots.txt``.

    Register responses with ``mock_scraper_http.add_response(...)``.  Set
    ``mock_scraper_http.robots_allowed = False`` to simulate a disallowed
    page.
    """
    router = MockRequestsRouter()
    router.install(monkeypatch, allow_robots=True)
    return router


@pytest.fixture
def mock_requests(mock_scraper_http: MockRequestsRouter) -> MockRequestsRouter:
    """Alias for :func:`mock_scraper_http`."""
    return mock_scraper_http


@pytest.fixture
def requests_response_factory() -> Callable[..., requests.Response]:
    """Return :func:`make_requests_response` for ad-hoc scraper responses."""
    return make_requests_response


# ---------------------------------------------------------------------------
# Static Goodreads / Amazon HTML fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def goodreads_html() -> str:
    """A static Goodreads book page with valid JSON-LD and meta tags."""
    return GOODREADS_BOOK_HTML


@pytest.fixture
def goodreads_book_html(goodreads_html: str) -> str:
    """Alias for :func:`goodreads_html`."""
    return goodreads_html


@pytest.fixture
def goodreads_search_html() -> str:
    """A static Goodreads search-results page with an ``a.bookTitle`` link."""
    return GOODREADS_SEARCH_HTML


@pytest.fixture
def amazon_html() -> str:
    """A static Amazon product page with valid JSON-LD and meta tags."""
    return AMAZON_BOOK_HTML


@pytest.fixture
def amazon_book_html(amazon_html: str) -> str:
    """Alias for :func:`amazon_html`."""
    return amazon_html


@pytest.fixture
def amazon_search_html() -> str:
    """A static Amazon search-results page with a product link."""
    return AMAZON_SEARCH_HTML


@pytest.fixture
def malformed_json_ld_html() -> str:
    """HTML whose JSON-LD block is invalid but whose meta tags are usable."""
    return MALFORMED_JSON_LD_HTML


@pytest.fixture
def empty_html() -> str:
    """An empty-but-well-formed HTML document."""
    return EMPTY_HTML


@pytest.fixture
def amazon_captcha_html() -> str:
    """A static CAPTCHA page used to verify no bypass is attempted."""
    return AMAZON_CAPTCHA_HTML


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enable_scraping(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Enable the scraping fallback for the current test."""
    monkeypatch.setattr(settings, "SCRAPING_ENABLED", True)
    return True


@pytest.fixture
def disable_scraping(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Explicitly disable the scraping fallback for the current test."""
    monkeypatch.setattr(settings, "SCRAPING_ENABLED", False)
    return False


# ---------------------------------------------------------------------------
# Domain-object factories and unit helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> MetadataNormalizer:
    """A ``MetadataNormalizer`` instance."""
    return MetadataNormalizer()


@pytest.fixture
def matcher() -> CandidateMatcher:
    """A ``CandidateMatcher`` instance."""
    return CandidateMatcher()


@pytest.fixture
def book_metadata_factory() -> Callable[..., BookMetadata]:
    """Factory for complete ``BookMetadata`` objects.

    Any field can be overridden::

        metadata = book_metadata_factory(authors=[], page_count=None)
    """
    defaults: dict[str, Any] = {
        "title": "Clean Code",
        "subtitle": "A Handbook of Agile Software Craftsmanship",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "published_date": "2008-08-01",
        "description": (
            "Even bad code can function. But if code isn't clean, it can "
            "bring a development organization to its knees."
        ),
        "isbn_10": VALID_ISBN_10,
        "isbn_13": VALID_ISBN_13,
        "page_count": 464,
        "categories": ["Computers", "Software Engineering"],
        "language": "en",
        "cover_url": (
            "https://books.google.com/books/content?id=zyTCAlFPjgYC"
            "&printsec=frontcover&img=1&zoom=1"
        ),
        "source": "google_books",
        "source_id": "zyTCAlFPjgYC",
        "confidence": 0.0,
    }

    def _make(**overrides: Any) -> BookMetadata:
        data = dict(defaults)
        data.update(overrides)
        return BookMetadata(**data)

    return _make


@pytest.fixture
def sample_book_metadata(
    book_metadata_factory: Callable[..., BookMetadata],
) -> BookMetadata:
    """A single complete, normalised ``BookMetadata`` instance."""
    return book_metadata_factory()


@pytest.fixture
def provider_result_factory() -> Callable[..., ProviderResult]:
    """Factory for ``ProviderResult`` objects used by provider tests."""

    def _make(
        *,
        success: bool = True,
        data: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        raw_response: Optional[dict[str, Any]] = None,
    ) -> ProviderResult:
        return ProviderResult(
            success=success,
            data={"candidates": []} if data is None else data,
            error=error,
            raw_response=raw_response,
        )

    return _make


@pytest.fixture
def metadata_retriever_factory() -> Callable[[], Any]:
    """Factory that creates a fresh ``BookMetadataRetriever`` per call.

    Use this instead of requesting a pre-built retriever when a test enables
    scraping: ``MetadataScraper`` snapshots ``settings.SCRAPING_ENABLED`` in
    its constructor, so the retriever must be created *after*
    ``enable_scraping`` has run.
    """
    from app.books.metadata.service import BookMetadataRetriever

    def _make() -> Any:
        retriever = BookMetadataRetriever()
        # ``MetadataScraper`` snapshots ``settings.SCRAPING_ENABLED`` when it
        # is constructed.  Keep the snapshot in sync even if a test toggles
        # the setting between fixture setup and this factory call.
        retriever._scraper._enabled = settings.SCRAPING_ENABLED
        return retriever

    return _make


@pytest.fixture
def metadata_retriever(
    request: pytest.FixtureRequest,
    metadata_retriever_factory: Callable[[], Any],
) -> Any:
    """A ``BookMetadataRetriever`` wired to the isolated test environment.

    For scraping-enabled tests, prefer :func:`metadata_retriever_factory` and
    construct the retriever after requesting :func:`enable_scraping`.
    """
    retriever = metadata_retriever_factory()
    # Make fixture ordering irrelevant for the common
    # ``enable_scraping`` + ``metadata_retriever`` combination.
    if "enable_scraping" in request.fixturenames:
        retriever._scraper._enabled = True
    return retriever


# ---------------------------------------------------------------------------
# Miscellaneous convenience fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def google_books_provider() -> GoogleBooksProvider:
    """A ``GoogleBooksProvider`` instance (HTTP is fail-closed by default)."""
    return GoogleBooksProvider()


@pytest.fixture
def open_library_provider() -> OpenLibraryProvider:
    """An ``OpenLibraryProvider`` instance (HTTP is fail-closed by default)."""
    return OpenLibraryProvider()


@pytest.fixture
def metadata_scraper(request: pytest.FixtureRequest) -> MetadataScraper:
    """A ``MetadataScraper`` instance (scraping is disabled by default)."""
    scraper = MetadataScraper()
    if "enable_scraping" in request.fixturenames:
        scraper._enabled = True
    return scraper


@pytest.fixture
def fake_httpx_response_factory() -> Callable[..., MockHTTPResponse]:
    """Return the :class:`MockHTTPResponse` constructor for direct use."""
    return MockHTTPResponse


# ---------------------------------------------------------------------------
# Async test helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    """Force the anyio pytest plugin to run async tests on asyncio.

    ``pytest-asyncio`` is not a project dependency, but the ``anyio`` pytest
    plugin (shipped with anyio, which Starlette/FastAPI already require) can
    run ``async def`` tests marked with ``@pytest.mark.anyio``.
    """
    return "asyncio"


@pytest.fixture
def run_async() -> Callable[[Any], Any]:
    """Return a helper that runs a coroutine to completion on a fresh loop.

    Useful for ``async`` provider methods from a synchronous test when a
    plugin such as ``pytest-asyncio`` is unavailable::

        def test_provider(run_async, mock_google_books_http):
            result = run_async(provider.search_by_isbn("9780132350884"))
    """

    def _run(coro: Any) -> Any:
        return asyncio.run(coro)

    return _run


__all__ = [
    # constants
    "METADATA_ENDPOINT",
    "VALID_ISBN_10",
    "VALID_ISBN_13",
    "VALID_ISBN_10_ROADMAP",
    "VALID_ISBN_13_ROADMAP",
    # mock helpers
    "MockHTTPRouter",
    "MockHTTPResponse",
    "MockRequestsRouter",
    "make_requests_response",
    # provider payloads
    "GOOGLE_BOOKS_RESPONSE",
    "OPEN_LIBRARY_ISBN_RESPONSE",
    "OPEN_LIBRARY_SEARCH_RESPONSE",
    # HTML samples
    "GOODREADS_BOOK_HTML",
    "AMAZON_BOOK_HTML",
]
