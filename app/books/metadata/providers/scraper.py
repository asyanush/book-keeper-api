"""Web scraping fallback for book metadata.

Provides an optional, controlled web-scraping layer that extracts
book metadata from public-facing pages on Goodreads and Amazon when
the primary API providers return insufficient results.

Scraping is only enabled when ``settings.SCRAPING_ENABLED`` is
``True``.  All scrapers respect ``robots.txt``, rate limits, and
access restrictions.  No CAPTCHAs, logins, or anti-bot measures are
bypassed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from app.books.metadata.schemas import BookMetadata
from app.core.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level constants
# ------------------------------------------------------------------

_DEFAULT_TIMEOUT = 15  # seconds
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BookKeeperBot/1.0; "
        "+https://github.com/book-keeper-api)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Minimum delay between requests to the same domain (seconds)
_RATE_LIMIT_DELAY = 2.0

# Recognised JSON-LD types for books
_BOOK_TYPES = {
    "Book",
    "http://schema.org/Book",
    "https://schema.org/Book",
    "Product",
    "http://schema.org/Product",
    "https://schema.org/Product",
}


class MetadataScraper:
    """Optional web-scraping fallback for metadata retrieval.

    Scrapes public book pages to extract structured metadata
    (JSON-LD where available, then HTML fallback).  Designed to
    fail gracefully – a blocked or failing source never causes
    the metadata pipeline to crash.
    """

    def __init__(self) -> None:
        self._enabled = settings.SCRAPING_ENABLED
        self._last_request: Dict[str, float] = {}  # domain → last request time

    # ------------------------------------------------------------------
    # HTTP request infrastructure
    # ------------------------------------------------------------------

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make an HTTP GET request with proper headers and error handling.

        Respects per-domain rate limits.  Returns ``None`` on any
        failure (timeout, connection error, HTTP error) so callers
        can continue gracefully.

        Args:
            url: Full URL to request.

        Returns:
            ``requests.Response`` on success, ``None`` on failure.
        """
        if not self._enabled:
            logger.debug("Scraping is disabled; skipping request")
            return None

        parsed = urlparse(url)
        domain = parsed.netloc

        # --- Rate limiting ---
        now = time.monotonic()
        last = self._last_request.get(domain, 0)
        elapsed = now - last
        if elapsed < _RATE_LIMIT_DELAY:
            to_wait = _RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate-limiting {domain}: waiting {to_wait:.1f}s")
            time.sleep(to_wait)

        self._last_request[domain] = time.monotonic()

        # --- Check robots.txt ---
        if not self._check_robots(url):
            logger.info(f"robots.txt disallows: {url}")
            return None

        try:
            response = requests.get(
                url,
                headers=_DEFAULT_HEADERS,
                timeout=_DEFAULT_TIMEOUT,
            )

            if response.status_code == 403:
                logger.info(f"Access forbidden (403) for {url}")
                return None

            if response.status_code == 404:
                logger.debug(f"Page not found (404): {url}")
                return None

            if response.status_code == 429:
                logger.warning(f"Rate limited (429) by {domain}")
                return None

            response.raise_for_status()
            return response

        except requests.Timeout:
            logger.warning(f"Request timed out: {url}")
            return None
        except requests.ConnectionError as exc:
            logger.warning(f"Connection error for {url}: {exc}")
            return None
        except requests.RequestException as exc:
            logger.warning(f"Request failed for {url}: {exc}")
            return None

    def _check_robots(self, url: str) -> bool:
        """Check robots.txt for the given URL.

        Returns ``True`` if the URL is allowed, ``False`` otherwise.
        If robots.txt cannot be fetched, conservatively returns
        ``False``.
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], url)
        except Exception as exc:
            logger.warning(f"Could not check robots.txt for {parsed.netloc}: {exc}")
            return False

    # ------------------------------------------------------------------
    # JSON-LD extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_json_ld(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract book metadata from JSON-LD structured data.

        Searches ``<script type="application/ld+json">`` tags for
        schema.org ``Book`` or ``Product`` types and returns the
        first matching block as a dictionary.

        Args:
            soup: Parsed HTML ``BeautifulSoup`` instance.

        Returns:
            Parsed JSON-LD dict, or ``None``.
        """
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue

            # JSON-LD may be a list of objects or a single object
            if isinstance(data, list):
                items = data
            else:
                items = [data]

            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("@type", "")
                # Normalise: http → https and strip trailing slash
                item_type_norm = (
                    item_type.replace("http://", "https://").rstrip("/")
                )
                if item_type_norm in _BOOK_TYPES:
                    return item

        return None

    # ------------------------------------------------------------------
    # HTML fallback extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_html(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Extract metadata from HTML markup as a fallback to JSON-LD.

        Looks for common patterns used by Goodreads and Amazon,
        including ``<meta>`` tags, ``<title>`` elements, and known
        class-based selectors.  Returns a partial candidate dict
        with whatever can be reliably extracted.

        Args:
            soup: Parsed HTML ``BeautifulSoup`` instance.

        Returns:
            Dict of extracted metadata, or ``None`` if nothing
            useful was found.
        """
        result: Dict[str, Any] = {}

        # --- Page title ---
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            page_title = title_tag.string.strip() if title_tag.string else ""
            if page_title:
                result["title"] = page_title

        # --- Meta tags ---
        meta_map: Dict[str, str] = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property") or ""
            content = meta.get("content", "")
            if name and content:
                meta_map[name.lower()] = content

        # og:title often has a cleaner title
        if "og:title" in meta_map:
            result["title"] = meta_map["og:title"]

        if "og:description" in meta_map:
            result["description"] = meta_map["og:description"]

        if "books:isbn" in meta_map:
            isbn_val = meta_map["books:isbn"]
            cleaned = isbn_val.replace("-", "").replace(" ", "")
            if len(cleaned) == 10:
                result["isbn_10"] = cleaned
            elif len(cleaned) == 13:
                result["isbn_13"] = cleaned

        # --- Author (og:book:author or author meta) ---
        author = meta_map.get("og:book:author") or meta_map.get("author")
        if author:
            result["authors"] = [a.strip() for a in author.split(",") if a.strip()]

        # --- Publisher ---
        publisher = meta_map.get("og:book:publisher") or meta_map.get("publisher")
        if publisher:
            result["publisher"] = publisher.strip()

        return result if result else None

    # ------------------------------------------------------------------
    # Goodreads-specific scraping
    # ------------------------------------------------------------------

    def _scrape_goodreads_by_isbn(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Search Goodreads for a book by ISBN.

        Goodreads has a dedicated ISBN landing page at
        ``/isbn/{isbn}`` which redirects to the book page.
        """
        url = f"https://www.goodreads.com/isbn/{isbn}"
        response = self._make_request(url)
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Try JSON-LD first
        ld_data = self.extract_from_json_ld(soup)
        if ld_data:
            return self._normalize_json_ld(ld_data)

        # Fall back to HTML extraction
        return self.extract_from_html(soup)

    def _scrape_goodreads_by_title_author(
        self, title: str, author: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Search Goodreads by title and optional author.

        Uses Goodreads' public search page.
        """
        query = title
        if author:
            query = f"{title} {author}"
        url = f"https://www.goodreads.com/search?q={requests.utils.quote(query)}"
        response = self._make_request(url)
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Try to find the first book result and follow it
        book_link = soup.select_one("a.bookTitle")
        if book_link:
            href = book_link.get("href", "")
            if href:
                detail_url = urljoin("https://www.goodreads.com", href)
                detail_response = self._make_request(detail_url)
                if detail_response is not None:
                    detail_soup = BeautifulSoup(detail_response.text, "lxml")
                    ld_data = self.extract_from_json_ld(detail_soup)
                    if ld_data:
                        return self._normalize_json_ld(ld_data)
                    return self.extract_from_html(detail_soup)

        return None

    # ------------------------------------------------------------------
    # Amazon-specific scraping
    # ------------------------------------------------------------------

    def _scrape_amazon_by_isbn(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Search Amazon for a book by ISBN."""
        url = f"https://www.amazon.com/dp/{isbn}"
        response = self._make_request(url)
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Try JSON-LD first
        ld_data = self.extract_from_json_ld(soup)
        if ld_data:
            return self._normalize_json_ld(ld_data)

        return self.extract_from_html(soup)

    def _scrape_amazon_by_title_author(
        self, title: str, author: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Search Amazon by title and optional author."""
        query = title
        if author:
            query = f"{title} {author}"
        url = f"https://www.amazon.com/s?k={requests.utils.quote(query)}&i=stripbooks"
        response = self._make_request(url)
        if response is None:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Try to find the first search result link
        result_link = soup.select_one("a.a-link-normal.s-no-outline")
        if result_link:
            href = result_link.get("href", "")
            if href:
                detail_url = urljoin("https://www.amazon.com", href)
                detail_response = self._make_request(detail_url)
                if detail_response is not None:
                    detail_soup = BeautifulSoup(detail_response.text, "lxml")
                    ld_data = self.extract_from_json_ld(detail_soup)
                    if ld_data:
                        return self._normalize_json_ld(ld_data)
                    return self.extract_from_html(detail_soup)

        return None

    # ------------------------------------------------------------------
    # JSON-LD normalisation (schema.org → BookMetadata-like dict)
    # ------------------------------------------------------------------

    def _normalize_json_ld(self, ld_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert schema.org JSON-LD into our internal candidate dict.

        Args:
            ld_data: Raw JSON-LD object for a Book/Product.

        Returns:
            Dict with keys matching the ``BookMetadata`` schema.
        """
        authors: List[str] = []
        author_data = ld_data.get("author")
        if isinstance(author_data, list):
            for a in author_data:
                if isinstance(a, dict):
                    name = a.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(a, str):
                    authors.append(a)
        elif isinstance(author_data, dict):
            name = author_data.get("name", "")
            if name:
                authors.append(name)
        elif isinstance(author_data, str):
            authors.append(author_data)

        # Extract ISBN from JSON-LD
        isbn_10: Optional[str] = None
        isbn_13: Optional[str] = None
        isbn_val = ld_data.get("isbn", "")
        if isbn_val:
            cleaned = isbn_val.replace("-", "").replace(" ", "")
            if len(cleaned) == 10:
                isbn_10 = cleaned
            elif len(cleaned) == 13:
                isbn_13 = cleaned

        # Publisher
        publisher: Optional[str] = None
        publisher_data = ld_data.get("publisher")
        if isinstance(publisher_data, dict):
            publisher = publisher_data.get("name", "")
        elif isinstance(publisher_data, str):
            publisher = publisher_data

        # Description
        desc = ld_data.get("description", "")
        if not isinstance(desc, str):
            desc = ""

        # Number of pages
        page_count: Optional[int] = None
        np_raw = ld_data.get("numberOfPages")
        if np_raw is not None:
            try:
                page_count = int(np_raw)
            except (TypeError, ValueError):
                pass

        return {
            "title": ld_data.get("name", ""),
            "subtitle": None,
            "authors": authors,
            "publisher": publisher or None,
            "published_date": ld_data.get("datePublished", ""),
            "description": desc or None,
            "isbn_10": isbn_10,
            "isbn_13": isbn_13,
            "page_count": page_count,
            "categories": [],
            "language": None,
            "cover_url": None,
            "source": "scraper",
            "source_id": ld_data.get("url") or ld_data.get("@id"),
            "confidence": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_by_isbn(self, isbn: str) -> List[BookMetadata]:
        """Search for book metadata by ISBN via web scraping.

        Tries Goodreads and Amazon in sequence.  Failures from
        individual sources are logged but do not prevent other
        sources from being tried.

        Args:
            isbn: Normalised ISBN (digit-only, no hyphens/spaces).

        Returns:
            List of ``BookMetadata`` candidates (may be empty).
        """
        if not self._enabled:
            return []

        candidates: List[BookMetadata] = []

        # Goodreads
        try:
            raw = self._scrape_goodreads_by_isbn(isbn)
            if raw:
                raw["source"] = "goodreads"
                try:
                    candidates.append(BookMetadata(**raw))
                except Exception as exc:
                    logger.warning(f"Failed to build BookMetadata from Goodreads data: {exc}")
        except Exception as exc:
            logger.warning(f"Goodreads scrape error (ISBN={isbn}): {exc}")

        # Amazon
        try:
            raw = self._scrape_amazon_by_isbn(isbn)
            if raw:
                raw["source"] = "amazon"
                try:
                    candidates.append(BookMetadata(**raw))
                except Exception as exc:
                    logger.warning(f"Failed to build BookMetadata from Amazon data: {exc}")
        except Exception as exc:
            logger.warning(f"Amazon scrape error (ISBN={isbn}): {exc}")

        return candidates

    def search_by_title_author(
        self, title: str, author: Optional[str] = None
    ) -> List[BookMetadata]:
        """Search for book metadata by title / author via web scraping.

        Args:
            title: Book title.
            author: Author name, optional.

        Returns:
            List of ``BookMetadata`` candidates (may be empty).
        """
        if not self._enabled:
            return []

        candidates: List[BookMetadata] = []

        # Goodreads
        try:
            raw = self._scrape_goodreads_by_title_author(title, author)
            if raw:
                raw["source"] = "goodreads"
                try:
                    candidates.append(BookMetadata(**raw))
                except Exception as exc:
                    logger.warning(f"Failed to build BookMetadata from Goodreads data: {exc}")
        except Exception as exc:
            logger.warning(f"Goodreads scrape error (title={title!r}): {exc}")

        # Amazon
        try:
            raw = self._scrape_amazon_by_title_author(title, author)
            if raw:
                raw["source"] = "amazon"
                try:
                    candidates.append(BookMetadata(**raw))
                except Exception as exc:
                    logger.warning(f"Failed to build BookMetadata from Amazon data: {exc}")
        except Exception as exc:
            logger.warning(f"Amazon scrape error (title={title!r}): {exc}")

        return candidates