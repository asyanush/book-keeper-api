"""Unit tests for the optional web-scraping metadata fallback.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 18):

* Extract metadata from valid JSON-LD.
* Extract title and author from supported HTML markup.
* Extract ISBN-10 and ISBN-13.
* Handle missing metadata fields.
* Handle malformed JSON-LD.
* Handle empty HTML.
* Handle HTTP 403, 404, 429 and request timeouts.
* Respect disabled scraping configuration.
* Verify that blocked websites do not crash the application.
* Verify that no CAPTCHA or access-control bypass is attempted.
* Rate limiting (per-domain delays).

All HTTP traffic is served by the ``MockRequestsRouter`` fixture
(``mock_scraper_http``) in ``conftest.py``.  The autouse network guard there
patches ``requests.get`` and ``MetadataScraper._check_robots`` so a forgotten
mock fails the test instead of making a live request.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import requests
from bs4 import BeautifulSoup

from app.books.metadata.providers import scraper as scraper_module
from app.books.metadata.providers.scraper import (
    _RATE_LIMIT_DELAY,
    MetadataScraper,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISBN_13 = "9780132350884"  # Clean Code
ISBN_10 = "0132350882"
ISBN_13_HYPHENATED = "978-0-13-235088-4"

GOODREADS_DOMAIN = re.compile(r"goodreads\.com")
AMAZON_DOMAIN = re.compile(r"amazon\.com")
GOODREADS_ISBN_URL = re.compile(r"goodreads\.com/isbn/")
AMAZON_DP_URL = re.compile(r"amazon\.com/dp/")
GOODREADS_SEARCH_URL = re.compile(r"goodreads\.com/search")
AMAZON_SEARCH_URL = re.compile(r"amazon\.com/s\?")
GOODREADS_DETAIL_URL = re.compile(r"goodreads\.com/book/show/3735293")
AMAZON_DETAIL_URL = re.compile(r"amazon\.com/.*/dp/0132350884")

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def parse(html: str) -> BeautifulSoup:
    """Parse *html* with the same parser used by the scraper."""
    return BeautifulSoup(html, "lxml")


#: The real ``_check_robots`` implementation, captured before the autouse
#: test fixture replaces it with a fail-closed guard.  The robots-specific
#: tests below restore it to exercise the production logic.
_ORIGINAL_CHECK_ROBOTS = MetadataScraper.__dict__["_check_robots"]


@pytest.fixture
def enabled_scraper(enable_scraping: bool) -> MetadataScraper:
    """A scraper instance constructed while scraping is enabled.

    ``MetadataScraper`` snapshots ``settings.SCRAPING_ENABLED`` in its
    constructor, so the instance must be created *after* ``enable_scraping``
    has run.
    """
    assert enable_scraping is True
    scraper = MetadataScraper()
    assert scraper._enabled is True
    return scraper


@pytest.fixture
def disabled_scraper(disable_scraping: bool) -> MetadataScraper:
    """A scraper instance constructed while scraping is disabled."""
    assert disable_scraping is False
    scraper = MetadataScraper()
    assert scraper._enabled is False
    return scraper


def handle_isbn_ok(mock_scraper_http, *, goodreads_html: str, amazon_html: str) -> None:
    """Register successful Goodreads/Amazon ISBN-page responses."""
    mock_scraper_http.add_response(GOODREADS_ISBN_URL, text=goodreads_html)
    mock_scraper_http.add_response(AMAZON_DP_URL, text=amazon_html)


# ===========================================================================
# 1. JSON-LD extraction
# ===========================================================================


class TestJsonLdExtraction:
    """``MetadataScraper.extract_from_json_ld`` behaviour."""

    def test_extracts_book_from_goodreads_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        data = enabled_scraper.extract_from_json_ld(parse(goodreads_html))

        assert data is not None
        assert data["@type"] == "Book"
        assert data["name"] == "Clean Code"
        assert data["isbn"] == ISBN_13

    def test_extracts_product_from_amazon_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        amazon_html: str,
    ) -> None:
        data = enabled_scraper.extract_from_json_ld(parse(amazon_html))

        assert data is not None
        assert data["@type"] == "Product"
        assert data["name"] == (
            "Clean Code: A Handbook of Agile Software Craftsmanship"
        )
        assert data["isbn"] == ISBN_13

    def test_returns_none_for_malformed_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        malformed_json_ld_html: str,
    ) -> None:
        assert enabled_scraper.extract_from_json_ld(
            parse(malformed_json_ld_html)
        ) is None

    def test_returns_none_for_empty_html(
        self,
        enabled_scraper: MetadataScraper,
        empty_html: str,
    ) -> None:
        assert enabled_scraper.extract_from_json_ld(parse(empty_html)) is None

    def test_returns_none_for_captcha_html(
        self,
        enabled_scraper: MetadataScraper,
        amazon_captcha_html: str,
    ) -> None:
        assert enabled_scraper.extract_from_json_ld(
            parse(amazon_captcha_html)
        ) is None


# ===========================================================================
# 2. JSON-LD normalisation
# ===========================================================================


class TestJsonLdNormalisation:
    """``MetadataScraper._normalize_json_ld`` behaviour."""

    def test_normalises_goodreads_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        raw = enabled_scraper.extract_from_json_ld(parse(goodreads_html))
        assert raw is not None

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["title"] == "Clean Code"
        assert result["authors"] == ["Robert C. Martin"]
        assert result["publisher"] == "Prentice Hall"
        assert result["published_date"] == "2008-08-01"
        assert result["isbn_13"] == ISBN_13
        assert result["isbn_10"] is None
        assert result["page_count"] == 464
        assert result["source"] == "scraper"
        assert result["source_id"] == (
            "https://www.goodreads.com/book/show/3735293-clean-code"
        )

    def test_normalises_amazon_json_ld_with_dict_author(
        self,
        enabled_scraper: MetadataScraper,
        amazon_html: str,
    ) -> None:
        raw = enabled_scraper.extract_from_json_ld(parse(amazon_html))
        assert raw is not None

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["title"] == (
            "Clean Code: A Handbook of Agile Software Craftsmanship"
        )
        assert result["authors"] == ["Robert C. Martin"]
        assert result["isbn_13"] == ISBN_13
        assert result["published_date"] == "2008"

    def test_extracts_isbn_13_from_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        raw = enabled_scraper.extract_from_json_ld(parse(goodreads_html))
        assert raw is not None

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["isbn_13"] == ISBN_13
        assert result["isbn_10"] is None

    def test_extracts_isbn_10_from_json_ld(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        html = goodreads_html.replace(ISBN_13, ISBN_10)
        raw = enabled_scraper.extract_from_json_ld(parse(html))
        assert raw is not None

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["isbn_10"] == ISBN_10
        assert result["isbn_13"] is None

    def test_normalises_hyphenated_isbn(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        html = goodreads_html.replace(ISBN_13, ISBN_13_HYPHENATED)
        raw = enabled_scraper.extract_from_json_ld(parse(html))
        assert raw is not None

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["isbn_13"] == ISBN_13
        assert result["isbn_10"] is None

    def test_missing_json_ld_fields_do_not_crash(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        raw = dict(enabled_scraper.extract_from_json_ld(parse(goodreads_html)))
        for key in (
            "author",
            "isbn",
            "publisher",
            "numberOfPages",
            "datePublished",
            "description",
            "url",
            "@id",
        ):
            raw.pop(key, None)

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["title"] == "Clean Code"
        assert result["authors"] == []
        assert result["subtitle"] is None
        assert result["publisher"] is None
        assert result["published_date"] == ""
        assert result["description"] is None
        assert result["isbn_10"] is None
        assert result["isbn_13"] is None
        assert result["page_count"] is None
        assert result["categories"] == []
        assert result["language"] is None
        assert result["cover_url"] is None
        assert result["source_id"] is None
        assert result["confidence"] == 0.0

    def test_handles_string_author_and_publisher(
        self,
        enabled_scraper: MetadataScraper,
    ) -> None:
        raw = {
            "@type": "Book",
            "name": "A Plain Record",
            "author": "Jane Doe",
            "publisher": "Example Press",
            "numberOfPages": "not-a-number",
            "description": ["not", "a", "string"],
        }

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["title"] == "A Plain Record"
        assert result["authors"] == ["Jane Doe"]
        assert result["publisher"] == "Example Press"
        assert result["page_count"] is None
        assert result["description"] is None

    def test_handles_mixed_author_list(
        self,
        enabled_scraper: MetadataScraper,
    ) -> None:
        raw = {
            "@type": "Book",
            "name": "Collaborative Work",
            "author": [
                {"@type": "Person", "name": "First Author"},
                "Second Author",
                {"@type": "Person"},  # no name → ignored
            ],
        }

        result = enabled_scraper._normalize_json_ld(raw)

        assert result["authors"] == ["First Author", "Second Author"]


# ===========================================================================
# 3. HTML fallback extraction
# ===========================================================================


class TestHtmlExtraction:
    """``MetadataScraper.extract_from_html`` behaviour."""

    def test_extracts_title_and_author_from_goodreads_markup(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        result = enabled_scraper.extract_from_html(parse(goodreads_html))

        assert result is not None
        assert result["title"] == "Clean Code"
        assert result["authors"] == ["Robert C. Martin"]
        assert result["publisher"] == "Prentice Hall"
        assert result["isbn_13"] == ISBN_13
        assert result["description"].startswith("Even bad code can function.")

    def test_extracts_title_from_amazon_markup(
        self,
        enabled_scraper: MetadataScraper,
        amazon_html: str,
    ) -> None:
        result = enabled_scraper.extract_from_html(parse(amazon_html))

        assert result is not None
        assert result["title"] == (
            "Clean Code: A Handbook of Agile Software Craftsmanship"
        )
        # Amazon's author data lives in JSON-LD, not in the fallback markup.
        assert "authors" not in result

    def test_extracts_isbn_13_from_meta_tags(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        result = enabled_scraper.extract_from_html(parse(goodreads_html))

        assert result is not None
        assert result["isbn_13"] == ISBN_13
        assert "isbn_10" not in result

    def test_extracts_isbn_10_from_meta_tags(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        html = goodreads_html.replace(ISBN_13, ISBN_10)
        result = enabled_scraper.extract_from_html(parse(html))

        assert result is not None
        assert result["isbn_10"] == ISBN_10
        assert "isbn_13" not in result

    def test_normalises_hyphenated_isbn_in_meta_tags(
        self,
        enabled_scraper: MetadataScraper,
        goodreads_html: str,
    ) -> None:
        html = goodreads_html.replace(ISBN_13, ISBN_13_HYPHENATED)
        result = enabled_scraper.extract_from_html(parse(html))

        assert result is not None
        assert result["isbn_13"] == ISBN_13

    def test_falls_back_to_meta_title_when_json_ld_is_malformed(
        self,
        enabled_scraper: MetadataScraper,
        malformed_json_ld_html: str,
    ) -> None:
        result = enabled_scraper.extract_from_html(parse(malformed_json_ld_html))

        assert result is not None
        assert result["title"] == "Fallback Title From Meta Tags"
        assert "authors" not in result
        assert "isbn_13" not in result

    def test_returns_none_for_empty_html(
        self,
        enabled_scraper: MetadataScraper,
        empty_html: str,
    ) -> None:
        assert enabled_scraper.extract_from_html(parse(empty_html)) is None

    def test_captcha_page_yields_no_book_identifiers(
        self,
        enabled_scraper: MetadataScraper,
        amazon_captcha_html: str,
    ) -> None:
        result = enabled_scraper.extract_from_html(parse(amazon_captcha_html))

        # A CAPTCHA page may expose a generic <title>; it must never yield
        # book identifiers or authors.
        if result is not None:
            assert "isbn_10" not in result
            assert "isbn_13" not in result
            assert "authors" not in result
            assert "publisher" not in result


# ===========================================================================
# 4. search_by_isbn
# ===========================================================================


class TestSearchByIsbn:
    """``MetadataScraper.search_by_isbn`` behaviour."""

    def test_returns_candidates_from_both_domains(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
        amazon_html: str,
    ) -> None:
        handle_isbn_ok(
            mock_scraper_http,
            goodreads_html=goodreads_html,
            amazon_html=amazon_html,
        )

        candidates = enabled_scraper.search_by_isbn(ISBN_13)

        assert len(candidates) == 2
        by_source = {candidate.source: candidate for candidate in candidates}
        assert set(by_source) == {"goodreads", "amazon"}

        goodreads = by_source["goodreads"]
        assert goodreads.title == "Clean Code"
        assert goodreads.authors == ["Robert C. Martin"]
        assert goodreads.publisher == "Prentice Hall"
        assert goodreads.published_date == "2008-08-01"
        assert goodreads.isbn_13 == ISBN_13
        assert goodreads.isbn_10 is None
        assert goodreads.page_count == 464
        assert goodreads.source_id == (
            "https://www.goodreads.com/book/show/3735293-clean-code"
        )

        amazon = by_source["amazon"]
        assert amazon.title == (
            "Clean Code: A Handbook of Agile Software Craftsmanship"
        )
        assert amazon.authors == ["Robert C. Martin"]
        assert amazon.isbn_13 == ISBN_13
        assert amazon.published_date == "2008"

        urls = mock_scraper_http.requested_urls
        assert any(f"goodreads.com/isbn/{ISBN_13}" in url for url in urls)
        assert any(f"amazon.com/dp/{ISBN_13}" in url for url in urls)

    def test_one_blocked_domain_does_not_prevent_the_other(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        amazon_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, status_code=403)
        mock_scraper_http.add_response(AMAZON_DP_URL, text=amazon_html)

        candidates = enabled_scraper.search_by_isbn(ISBN_13)

        assert len(candidates) == 1
        assert candidates[0].source == "amazon"
        assert candidates[0].title == (
            "Clean Code: A Handbook of Agile Software Craftsmanship"
        )

    def test_missing_isbn_page_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, status_code=404)
        mock_scraper_http.add_response(AMAZON_DP_URL, status_code=404)

        assert enabled_scraper.search_by_isbn(ISBN_13) == []

    def test_empty_html_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        empty_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, text=empty_html)
        mock_scraper_http.add_response(AMAZON_DP_URL, text=empty_html)

        assert enabled_scraper.search_by_isbn(ISBN_13) == []

    def test_malformed_json_ld_falls_back_without_crashing(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        malformed_json_ld_html: str,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_ISBN_URL, text=malformed_json_ld_html
        )
        mock_scraper_http.add_response(
            AMAZON_DP_URL, text=malformed_json_ld_html
        )

        candidates = enabled_scraper.search_by_isbn(ISBN_13)

        assert len(candidates) == 2
        for candidate in candidates:
            assert candidate.title == "Fallback Title From Meta Tags"
            assert candidate.authors == []
            assert candidate.isbn_10 is None
            assert candidate.isbn_13 is None

    @pytest.mark.parametrize("status_code", [403, 404, 429, 500])
    def test_http_error_statuses_return_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        status_code: int,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_ISBN_URL, status_code=status_code
        )
        mock_scraper_http.add_response(AMAZON_DP_URL, status_code=status_code)

        assert enabled_scraper.search_by_isbn(ISBN_13) == []

    def test_request_timeout_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
    ) -> None:
        mock_scraper_http.add_exception(
            requests.Timeout("timed out"), GOODREADS_ISBN_URL
        )
        mock_scraper_http.add_exception(
            requests.Timeout("timed out"), AMAZON_DP_URL
        )

        assert enabled_scraper.search_by_isbn(ISBN_13) == []

    def test_connection_error_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
    ) -> None:
        mock_scraper_http.add_exception(
            requests.ConnectionError("connection refused"), GOODREADS_ISBN_URL
        )
        mock_scraper_http.add_exception(
            requests.ConnectionError("connection refused"), AMAZON_DP_URL
        )

        assert enabled_scraper.search_by_isbn(ISBN_13) == []

    def test_robots_disallowed_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
        amazon_html: str,
    ) -> None:
        mock_scraper_http.robots_allowed = False
        handle_isbn_ok(
            mock_scraper_http,
            goodreads_html=goodreads_html,
            amazon_html=amazon_html,
        )

        assert enabled_scraper.search_by_isbn(ISBN_13) == []
        assert mock_scraper_http.request_count == 0

    def test_disabled_scraping_makes_no_requests(
        self,
        disabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
        amazon_html: str,
    ) -> None:
        handle_isbn_ok(
            mock_scraper_http,
            goodreads_html=goodreads_html,
            amazon_html=amazon_html,
        )

        assert disabled_scraper.search_by_isbn(ISBN_13) == []
        assert mock_scraper_http.request_count == 0

    def test_make_request_returns_none_when_disabled(
        self,
        disabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, text=goodreads_html)

        assert (
            disabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is None
        )
        assert mock_scraper_http.request_count == 0

    def test_unexpected_source_error_is_contained(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        amazon_html: str,
    ) -> None:
        def boom(_isbn: str) -> dict[str, Any]:
            raise RuntimeError("unexpected scraper failure")

        mock_scraper_http.add_response(AMAZON_DP_URL, text=amazon_html)
        enabled_scraper._scrape_goodreads_by_isbn = boom  # type: ignore[method-assign]

        candidates = enabled_scraper.search_by_isbn(ISBN_13)

        assert len(candidates) == 1
        assert candidates[0].source == "amazon"

    def test_captcha_page_does_not_attempt_bypass(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        amazon_captcha_html: str,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_ISBN_URL, text=amazon_captcha_html
        )
        mock_scraper_http.add_response(
            AMAZON_DP_URL, text=amazon_captcha_html
        )

        # A CAPTCHA page must never raise out of the fallback.
        candidates = enabled_scraper.search_by_isbn(ISBN_13)

        urls = [url.lower() for url in mock_scraper_http.requested_urls]
        # The CAPTCHA challenge endpoint is never contacted or submitted.
        assert all("captcha" not in url for url in urls)
        assert all("validate" not in url for url in urls)
        # Exactly one GET per source: no retry loop trying to defeat the
        # challenge.
        assert mock_scraper_http.request_count == 2
        assert all("method" not in entry["kwargs"] for entry in mock_scraper_http.requests)
        # Nothing from the challenge is treated as real book metadata.
        for candidate in candidates:
            assert candidate.isbn_10 is None
            assert candidate.isbn_13 is None
            assert candidate.authors == []


# ===========================================================================
# 5. search_by_title_author
# ===========================================================================


class TestSearchByTitleAuthor:
    """``MetadataScraper.search_by_title_author`` behaviour."""

    def test_returns_candidates_from_both_domains(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_search_html: str,
        amazon_search_html: str,
        goodreads_html: str,
        amazon_html: str,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_SEARCH_URL, text=goodreads_search_html
        )
        mock_scraper_http.add_response(GOODREADS_DETAIL_URL, text=goodreads_html)
        mock_scraper_http.add_response(
            AMAZON_SEARCH_URL, text=amazon_search_html
        )
        mock_scraper_http.add_response(AMAZON_DETAIL_URL, text=amazon_html)

        candidates = enabled_scraper.search_by_title_author(
            "Clean Code", "Robert C. Martin"
        )

        assert len(candidates) == 2
        by_source = {candidate.source: candidate for candidate in candidates}
        assert set(by_source) == {"goodreads", "amazon"}
        assert by_source["goodreads"].title == "Clean Code"
        assert by_source["goodreads"].authors == ["Robert C. Martin"]
        assert by_source["amazon"].isbn_13 == ISBN_13

        urls = mock_scraper_http.requested_urls
        assert any("goodreads.com/search" in url for url in urls)
        assert any("amazon.com/s?" in url for url in urls)
        assert any("3735293" in url for url in urls)
        assert any("0132350884" in url for url in urls)

    def test_title_only_search_works(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_search_html: str,
        goodreads_html: str,
        amazon_search_html: str,
        amazon_html: str,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_SEARCH_URL, text=goodreads_search_html
        )
        mock_scraper_http.add_response(GOODREADS_DETAIL_URL, text=goodreads_html)
        mock_scraper_http.add_response(
            AMAZON_SEARCH_URL, text=amazon_search_html
        )
        mock_scraper_http.add_response(AMAZON_DETAIL_URL, text=amazon_html)

        candidates = enabled_scraper.search_by_title_author("Clean Code")

        assert len(candidates) == 2
        goodreads_search_url = next(
            url
            for url in mock_scraper_http.requested_urls
            if "goodreads.com/search" in url
        )
        assert "q=Clean%20Code" in goodreads_search_url
        assert "Robert" not in goodreads_search_url

    def test_search_page_without_results_returns_empty(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        empty_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_SEARCH_URL, text=empty_html)
        mock_scraper_http.add_response(AMAZON_SEARCH_URL, text=empty_html)

        assert (
            enabled_scraper.search_by_title_author(
                "Clean Code", "Robert C. Martin"
            )
            == []
        )

    def test_robots_disallowed_returns_empty_results(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_search_html: str,
        amazon_search_html: str,
    ) -> None:
        mock_scraper_http.robots_allowed = False
        mock_scraper_http.add_response(
            GOODREADS_SEARCH_URL, text=goodreads_search_html
        )
        mock_scraper_http.add_response(
            AMAZON_SEARCH_URL, text=amazon_search_html
        )

        assert (
            enabled_scraper.search_by_title_author(
                "Clean Code", "Robert C. Martin"
            )
            == []
        )
        assert mock_scraper_http.request_count == 0

    def test_disabled_scraping_makes_no_requests(
        self,
        disabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_search_html: str,
        amazon_search_html: str,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_SEARCH_URL, text=goodreads_search_html
        )
        mock_scraper_http.add_response(
            AMAZON_SEARCH_URL, text=amazon_search_html
        )

        assert (
            disabled_scraper.search_by_title_author(
                "Clean Code", "Robert C. Martin"
            )
            == []
        )
        assert mock_scraper_http.request_count == 0


# ===========================================================================
# 6. robots.txt handling
# ===========================================================================


class TestRobotsTxt:
    """``MetadataScraper._check_robots`` compliance behaviour."""

    @pytest.fixture(autouse=True)
    def _restore_real_robots_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Use the real ``_check_robots`` implementation for these tests."""
        monkeypatch.setattr(
            MetadataScraper, "_check_robots", _ORIGINAL_CHECK_ROBOTS
        )

    def test_allows_url_when_robots_permits(
        self,
        enabled_scraper: MetadataScraper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class AllowingParser:
            def set_url(self, _url: str) -> None:
                pass

            def read(self) -> None:
                pass

            def can_fetch(self, _user_agent: str, _url: str) -> bool:
                return True

        monkeypatch.setattr(scraper_module, "RobotFileParser", AllowingParser)

        assert enabled_scraper._check_robots(
            f"https://www.goodreads.com/isbn/{ISBN_13}"
        ) is True

    def test_disallows_url_when_robots_forbids(
        self,
        enabled_scraper: MetadataScraper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class DenyingParser:
            def set_url(self, _url: str) -> None:
                pass

            def read(self) -> None:
                pass

            def can_fetch(self, _user_agent: str, _url: str) -> bool:
                return False

        monkeypatch.setattr(scraper_module, "RobotFileParser", DenyingParser)

        assert enabled_scraper._check_robots(
            f"https://www.goodreads.com/isbn/{ISBN_13}"
        ) is False

    def test_unreachable_robots_txt_fails_closed(
        self,
        enabled_scraper: MetadataScraper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class BrokenParser:
            def set_url(self, _url: str) -> None:
                pass

            def read(self) -> None:
                raise OSError("robots.txt unavailable")

            def can_fetch(self, _user_agent: str, _url: str) -> bool:
                return True

        monkeypatch.setattr(scraper_module, "RobotFileParser", BrokenParser)

        assert enabled_scraper._check_robots(
            f"https://www.goodreads.com/isbn/{ISBN_13}"
        ) is False


# ===========================================================================
# 7. HTTP request infrastructure and rate limiting
# ===========================================================================


class TestRequestInfrastructure:
    """``_make_request`` success/error handling and per-domain rate limits."""

    def test_successful_request_returns_response(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, text=goodreads_html)

        response = enabled_scraper._make_request(
            f"https://www.goodreads.com/isbn/{ISBN_13}"
        )

        assert response is not None
        assert response.status_code == 200
        assert "Clean Code" in response.text
        assert mock_scraper_http.request_count == 1
        entry = mock_scraper_http.requests[0]
        assert entry["kwargs"]["timeout"] == scraper_module._DEFAULT_TIMEOUT
        assert "User-Agent" in entry["kwargs"]["headers"]

    @pytest.mark.parametrize("status_code", [403, 404, 429])
    def test_http_error_statuses_return_none(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        status_code: int,
    ) -> None:
        mock_scraper_http.add_response(
            GOODREADS_ISBN_URL, status_code=status_code
        )

        assert (
            enabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is None
        )

    def test_timeout_returns_none(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
    ) -> None:
        mock_scraper_http.add_exception(
            requests.Timeout("timed out"), GOODREADS_ISBN_URL
        )

        assert (
            enabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is None
        )

    def test_request_exception_returns_none(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
    ) -> None:
        mock_scraper_http.add_exception(
            requests.RequestException("broken pipe"), GOODREADS_ISBN_URL
        )

        assert (
            enabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is None
        )

    def test_same_domain_requests_are_rate_limited(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        monkeypatch: pytest.MonkeyPatch,
        empty_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_DOMAIN, text=empty_html)
        sleeps: list[float] = []

        def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(scraper_module.time, "sleep", record_sleep)

        url = f"https://www.goodreads.com/isbn/{ISBN_13}"
        assert enabled_scraper._make_request(url) is not None
        assert enabled_scraper._make_request(url) is not None

        assert len(sleeps) == 1
        assert 0 < sleeps[0] <= _RATE_LIMIT_DELAY

    def test_different_domains_are_not_rate_limited(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        monkeypatch: pytest.MonkeyPatch,
        empty_html: str,
    ) -> None:
        mock_scraper_http.add_response(GOODREADS_DOMAIN, text=empty_html)
        mock_scraper_http.add_response(AMAZON_DOMAIN, text=empty_html)
        sleeps: list[float] = []

        monkeypatch.setattr(
            scraper_module.time, "sleep", lambda seconds: sleeps.append(seconds)
        )

        assert (
            enabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is not None
        )
        assert (
            enabled_scraper._make_request(f"https://www.amazon.com/dp/{ISBN_13}")
            is not None
        )

        assert sleeps == []

    def test_robots_disallowed_request_returns_none(
        self,
        enabled_scraper: MetadataScraper,
        mock_scraper_http,
        goodreads_html: str,
    ) -> None:
        mock_scraper_http.robots_allowed = False
        mock_scraper_http.add_response(GOODREADS_ISBN_URL, text=goodreads_html)

        assert (
            enabled_scraper._make_request(
                f"https://www.goodreads.com/isbn/{ISBN_13}"
            )
            is None
        )
        assert mock_scraper_http.request_count == 0
