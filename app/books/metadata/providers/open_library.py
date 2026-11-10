"""Open Library API provider implementation."""

import asyncio
import httpx
import logging
from typing import Optional, List, Dict, Any

from app.books.metadata.providers.base import MetadataProvider, ProviderResult
from app.core.config import settings

logger = logging.getLogger(__name__)

COVER_BASE_URL = "https://covers.openlibrary.org/b/id"


class OpenLibraryProvider(MetadataProvider):
    """Open Library API provider for book metadata retrieval.

    Uses the Open Library REST API to look up book metadata by ISBN,
    search by title/author, and fetch detailed edition/work information.
    """

    @property
    def name(self) -> str:
        return "open_library"

    def __init__(self):
        self.base_url = settings.OPEN_LIBRARY_API_URL
        self.timeout = settings.METADATA_TIMEOUT
        self.max_retries = settings.METADATA_MAX_RETRIES

    # ------------------------------------------------------------------
    # HTTP client & request infrastructure
    # ------------------------------------------------------------------

    def _build_client(self) -> httpx.AsyncClient:
        """Create an async HTTP client with timeout configuration."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )

    def _build_cover_url(self, cover_id: int, size: str = "L") -> Optional[str]:
        """Build cover image URL from cover ID.

        Args:
            cover_id: Numeric cover identifier from Open Library.
            size: Image size suffix ('S', 'M', 'L').  Defaults to 'L' (large).

        Returns:
            Full cover URL string, or None if cover_id is falsy.
        """
        if cover_id:
            return f"{COVER_BASE_URL}/{cover_id}-{size}.jpg"
        return None

    async def _make_request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> ProviderResult:
        """Make HTTP request to Open Library with retry logic.

        Handles HTTP 429 (rate limit), HTTP 5xx (server errors),
        HTTP 404 (not found → empty results), HTTP 400 (bad request),
        timeouts, connection errors, and malformed JSON.

        Args:
            url: Full request URL.
            params: Query parameters.

        Returns:
            ProviderResult with success flag, parsed data, and
            optionally an error message.
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._build_client() as client:
                    response = await client.get(url, params=params or {})

                # --- rate limiting (429) ----------------------------------
                if response.status_code == 429:
                    logger.warning(
                        f"Open Library API rate limited "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(
                        success=False,
                        error="Rate limit exceeded",
                    )

                # --- server errors (5xx) ----------------------------------
                if response.status_code >= 500:
                    logger.warning(
                        f"Open Library API server error: {response.status_code} "
                        f"(attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(
                        success=False,
                        error=f"Server error: {response.status_code}",
                    )

                # --- not found (404) → empty results ----------------------
                if response.status_code == 404:
                    return ProviderResult(
                        success=True,
                        data={"candidates": []},
                    )

                # --- bad request (400) ------------------------------------
                if response.status_code == 400:
                    return ProviderResult(
                        success=False,
                        error=f"Bad request: {response.text}",
                    )

                response.raise_for_status()

                # --- parse JSON -------------------------------------------
                try:
                    data = response.json()
                except Exception as e:
                    return ProviderResult(
                        success=False,
                        error=f"Invalid JSON response: {e}",
                    )

                return ProviderResult(success=True, data=data, raw_response=data)

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Open Library API timeout (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Open Library API request error: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue

        return ProviderResult(
            success=False,
            error=f"Request failed after retries: {last_error}",
        )

    # ------------------------------------------------------------------
    # Author resolution
    # ------------------------------------------------------------------

    async def _resolve_author_name(
        self, client: httpx.AsyncClient, author_key: str
    ) -> Optional[str]:
        """Resolve a single author key to a human-readable name.

        Calls ``/authors/{key}.json`` and returns the ``name`` field.

        Args:
            client: Reusable HTTP client.
            author_key: Author key (e.g. ``/authors/OL123A`` or ``OL123A``).

        Returns:
            Author name string, or None on failure.
        """
        clean_key = author_key.strip("/")
        url = f"{self.base_url}/{clean_key}.json"

        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("name") or data.get("personal_name")
        except Exception as e:
            logger.warning(f"Failed to resolve author {author_key}: {e}")
        return None

    async def _resolve_authors(
        self, client: httpx.AsyncClient, author_refs: List[Any]
    ) -> List[str]:
        """Resolve a list of author references into author name strings.

        Accepts mixed formats (dict objects with ``key``/``name`` fields,
        raw strings, or keys like ``/authors/OL123A``).  Only keys that
        cannot be resolved locally trigger HTTP requests, and those
        requests are issued concurrently.

        Args:
            client: Reusable HTTP client.
            author_refs: Raw author list from Open Library edition data.

        Returns:
            Deduplicated list of author names (order preserved).
        """
        author_names: List[str] = []
        keys_to_resolve: List[str] = []

        for ref in author_refs:
            if isinstance(ref, dict):
                # Edition-author object: {"key": "/authors/OL123A"}
                name = ref.get("name")
                key = ref.get("key", "")
                if name:
                    author_names.append(name)
                elif key:
                    keys_to_resolve.append(key)
            elif isinstance(ref, str):
                if "/authors/" in ref:
                    keys_to_resolve.append(ref)
                else:
                    author_names.append(ref)

        # Resolve unresolved keys concurrently
        if keys_to_resolve:
            tasks = [
                self._resolve_author_name(client, key) for key in keys_to_resolve
            ]
            resolved = await asyncio.gather(*tasks)
            for name in resolved:
                if name:
                    author_names.append(name)

        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for name in author_names:
            if name not in seen:
                seen.add(name)
                deduped.append(name)

        return deduped

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    def _extract_isbns(
        self, data: Dict[str, Any]
    ) -> tuple:
        """Extract ISBN-10 and ISBN-13 from Open Library data.

        Looks for dedicated ``isbn_10`` / ``isbn_13`` lists first,
        then falls back to the generic ``isbn`` list.

        Returns:
            Tuple of ``(isbn_10, isbn_13)`` – either may be None.
        """
        isbn_10: Optional[str] = None
        isbn_13: Optional[str] = None

        isbn_10_list: Any = data.get("isbn_10", [])
        isbn_13_list: Any = data.get("isbn_13", [])

        if isbn_10_list:
            if isinstance(isbn_10_list, list) and isbn_10_list:
                isbn_10 = str(isbn_10_list[0])
            else:
                isbn_10 = str(isbn_10_list)

        if isbn_13_list:
            if isinstance(isbn_13_list, list) and isbn_13_list:
                isbn_13 = str(isbn_13_list[0])
            else:
                isbn_13 = str(isbn_13_list)

        # Fallback to generic ``isbn`` list
        if not isbn_10 and not isbn_13:
            isbns: Any = data.get("isbn", [])
            if isbns:
                for isbn in isbns:
                    isbn_str = str(isbn)
                    if len(isbn_str) == 10 and not isbn_10:
                        isbn_10 = isbn_str
                    elif len(isbn_str) == 13 and not isbn_13:
                        isbn_13 = isbn_str

        return isbn_10, isbn_13

    def _parse_language(self, languages: List[Any]) -> Optional[str]:
        """Parse language code from Open Library language structures.

        Open Library represents languages as objects like
        ``{"key": "/languages/eng"}`` or plain strings like ``"eng"``.

        Returns:
            Language code (e.g. ``"eng"``) or None.
        """
        if not languages:
            return None
        for lang in languages:
            if isinstance(lang, dict):
                key: str = lang.get("key", "")
                parts = key.rstrip("/").split("/")
                if parts:
                    return parts[-1]
            elif isinstance(lang, str):
                return lang
        return None

    def _extract_edition_id(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the edition identifier from a ``key`` field.

        Key format: ``/books/OL123M`` → returns ``OL123M``.
        """
        key: str = data.get("key", "")
        if key:
            parts = key.rstrip("/").split("/")
            if parts:
                return parts[-1]
        return None

    def _extract_work_id(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the work identifier from edition ``works`` array.

        Works format: ``[{"key": "/works/OL456W"}]`` → returns ``OL456W``.
        """
        works: Any = data.get("works", [])
        if works:
            work = works[0]
            if isinstance(work, dict):
                key: str = work.get("key", "")
                parts = key.rstrip("/").split("/")
                if parts:
                    return parts[-1]
        return None

    def _extract_subjects(self, subjects: Any) -> List[str]:
        """Normalise subjects / categories list.

        Handles plain string lists and object lists
        (e.g. ``{"name": "Fiction"}``).

        Returns:
            Flat list of subject strings.
        """
        if not subjects or not isinstance(subjects, list):
            return []

        result: List[str] = []
        for s in subjects:
            if isinstance(s, dict):
                name = s.get("name", "")
                if name:
                    result.append(name)
            elif isinstance(s, str):
                result.append(s)
        return result

    # ------------------------------------------------------------------
    # Main parsing pipeline
    # ------------------------------------------------------------------

    async def _parse_edition_data(
        self, client: httpx.AsyncClient, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse raw Open Library edition / work data into a candidate dict.

        Resolves author keys to names, builds cover URLs, extracts
        identifiers, and normalises all fields to match the
        ``BookMetadata`` schema.

        Args:
            client: Reusable HTTP client (for author resolution).
            data: Raw JSON response from Open Library.

        Returns:
            Candidate dictionary with keys matching ``BookMetadata``.
        """
        # Authors -----------------------------------------------------------
        author_refs: Any = data.get("authors", [])
        authors = await self._resolve_authors(client, author_refs)

        # ISBNs -------------------------------------------------------------
        isbn_10, isbn_13 = self._extract_isbns(data)

        # Cover -------------------------------------------------------------
        covers: Any = data.get("covers", [])
        cover_id = covers[0] if covers else None
        cover_url = self._build_cover_url(cover_id) if cover_id else None

        # Identifiers -------------------------------------------------------
        edition_id = self._extract_edition_id(data)
        work_id = self._extract_work_id(data)

        # Subjects / categories ---------------------------------------------
        categories = self._extract_subjects(data.get("subjects", []))

        # Publisher ---------------------------------------------------------
        publishers: Any = data.get("publishers", [])
        if isinstance(publishers, list) and publishers:
            publisher = publishers[0]
        elif publishers:
            publisher = str(publishers)
        else:
            publisher = None

        # Description -------------------------------------------------------
        description: Any = data.get("description")
        if isinstance(description, dict):
            # Work descriptions are often {"type": "/type/text", "value": "..."}
            description = description.get("value")
        elif not isinstance(description, (str, type(None))):
            description = None

        return {
            "title": data.get("title"),
            "subtitle": data.get("subtitle"),
            "authors": authors,
            "publisher": publisher,
            "published_date": data.get("publish_date"),
            "description": description,
            "isbn_10": isbn_10,
            "isbn_13": isbn_13,
            "page_count": data.get("number_of_pages"),
            "categories": categories,
            "language": self._parse_language(data.get("languages", [])),
            "cover_url": cover_url,
            "source": self.name,
            "source_id": edition_id or work_id,
            "confidence": 0.0,  # Will be set by the matcher
        }

    def _parse_search_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single search-result document into a candidate dict.

        Open Library search results already include author names and
        most metadata inline, so no additional HTTP requests are made
        for each search result.

        Args:
            doc: A single ``docs[]`` entry from ``/search.json``.

        Returns:
            Candidate dictionary.
        """
        # Authors are already resolved to names in search results
        author_names: Any = doc.get("author_name", [])

        # ISBNs from search results
        isbn_10: Optional[str] = None
        isbn_13: Optional[str] = None
        isbns: Any = doc.get("isbn", [])
        if isbns:
            for isbn in isbns:
                isbn_str = str(isbn)
                if len(isbn_str) == 10 and not isbn_10:
                    isbn_10 = isbn_str
                elif len(isbn_str) == 13 and not isbn_13:
                    isbn_13 = isbn_str

        # Cover
        cover_id = doc.get("cover_i")
        cover_url = self._build_cover_url(cover_id) if cover_id else None

        # Source identifier – use first edition key if available
        edition_keys: Any = doc.get("edition_key", [])
        source_id = edition_keys[0] if edition_keys else None

        # Publisher
        publishers: Any = doc.get("publisher", [])
        publisher = publishers[0] if publishers else None

        # Publication date (search results only have year)
        publish_year = doc.get("first_publish_year")
        published_date = str(publish_year) if publish_year else None

        # Categories / subjects
        categories: Any = doc.get("subject", [])

        # Language
        languages: Any = doc.get("language", [])
        language = languages[0] if languages else None

        return {
            "title": doc.get("title"),
            "subtitle": doc.get("subtitle"),
            "authors": author_names if isinstance(author_names, list) else [],
            "publisher": publisher,
            "published_date": published_date,
            "description": None,  # Search results do not include descriptions
            "isbn_10": isbn_10,
            "isbn_13": isbn_13,
            "page_count": None,  # Search results do not reliably include page count
            "categories": categories if isinstance(categories, list) else [],
            "language": language,
            "cover_url": cover_url,
            "source": self.name,
            "source_id": source_id,
            "confidence": 0.0,  # Will be set by the matcher
        }

    # ------------------------------------------------------------------
    # Provider interface methods
    # ------------------------------------------------------------------

    async def search_by_isbn(self, isbn: str) -> ProviderResult:
        """Search for a book by ISBN using the Open Library API.

        Hits ``/isbn/{isbn}.json`` and returns a single candidate
        built from the edition data.

        Args:
            isbn: Normalised ISBN-10 or ISBN-13 (no hyphens/spaces).

        Returns:
            ProviderResult with ``data.candidates`` list.
        """
        url = f"{self.base_url}/isbn/{isbn}.json"
        result = await self._make_request(url)

        if not result.success:
            return result

        # 404 / empty-candidates response already pre-formatted
        if result.data is None or "candidates" in (result.data or {}):
            return result

        # Guard against empty-but-not-404 responses (e.g. ``{}``)
        if not result.data:
            return ProviderResult(success=True, data={"candidates": []})

        # Resolve authors and build the candidate
        async with self._build_client() as client:
            candidate = await self._parse_edition_data(client, result.data)

        return ProviderResult(
            success=True,
            data={"candidates": [candidate]},
            raw_response=result.raw_response,
        )

    async def search_by_title_author(
        self, title: str, author: Optional[str] = None
    ) -> ProviderResult:
        """Search for books by title and optional author.

        Uses ``/search.json`` with ``title`` and ``author`` parameters.
        Search results already include author names, so no additional
        HTTP requests are needed for each result item.

        Args:
            title: Book title (required).
            author: Author name (optional).

        Returns:
            ProviderResult with ``data.candidates`` list.
        """
        params: Dict[str, Any] = {"title": title, "limit": 20}
        if author:
            params["author"] = author

        # If only author is somehow supplied without title, fall back
        # to a general query (this path is defensive – the base class
        # signature guarantees *title* is always provided).
        url = f"{self.base_url}/search.json"
        if not title.strip() and author:
            params = {"q": author, "limit": 20}

        result = await self._make_request(url, params=params)

        if not result.success:
            return result

        # 404 / empty-candidates response already pre-formatted
        if result.data is None or "candidates" in (result.data or {}):
            return result

        docs: Any = (result.data or {}).get("docs", [])
        if not docs:
            return ProviderResult(success=True, data={"candidates": []})

        candidates = [self._parse_search_doc(doc) for doc in docs]

        return ProviderResult(
            success=True,
            data={"candidates": candidates},
            raw_response=result.raw_response,
        )

    async def get_book_details(self, provider_id: str) -> ProviderResult:
        """Get detailed book information by Open Library edition or work ID.

        Accepts both bare OLIDs (``OL123M``, ``OL456W``) and full key
        paths (``/books/OL123M``, ``/works/OL456W``).  Dispatches to
        the correct endpoint based on the ID suffix.

        Args:
            provider_id: Open Library edition or work identifier.

        Returns:
            ProviderResult with ``data.candidates`` list.
        """
        clean_id = provider_id.strip("/")

        # Determine endpoint from the ID structure
        if "M" in clean_id.upper() and "/works/" not in clean_id.lower():
            # Edition ID (ends with 'M', e.g. OL123M)
            if "/" not in clean_id:
                url = f"{self.base_url}/books/{clean_id}.json"
            else:
                url = f"{self.base_url}/{clean_id}.json"
        elif "W" in clean_id.upper() and "/books/" not in clean_id.lower():
            # Work ID (ends with 'W', e.g. OL456W)
            if "/" not in clean_id:
                url = f"{self.base_url}/works/{clean_id}.json"
            else:
                url = f"{self.base_url}/{clean_id}.json"
        else:
            # Ambiguous – try edition first
            if "/" not in clean_id:
                url = f"{self.base_url}/books/{clean_id}.json"
            else:
                url = f"{self.base_url}/{clean_id}.json"

        result = await self._make_request(url)

        if not result.success:
            return result

        # 404 / empty-candidates response already pre-formatted
        if result.data is None or "candidates" in (result.data or {}):
            return result

        # Guard against empty response
        if not result.data:
            return ProviderResult(success=True, data={"candidates": []})

        async with self._build_client() as client:
            candidate = await self._parse_edition_data(client, result.data)

        return ProviderResult(
            success=True,
            data={"candidates": [candidate]},
            raw_response=result.raw_response,
        )