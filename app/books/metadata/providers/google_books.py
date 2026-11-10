"""Google Books API provider implementation."""

import httpx
import logging
from typing import Optional, List, Dict, Any

from app.books.metadata.providers.base import MetadataProvider, ProviderResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class GoogleBooksProvider(MetadataProvider):
    """Google Books API provider for book metadata retrieval."""

    @property
    def name(self) -> str:
        return "google_books"

    def __init__(self):
        self.base_url = settings.GOOGLE_BOOKS_API_URL
        self.api_key = settings.GOOGLE_BOOKS_API_KEY
        self.timeout = settings.METADATA_TIMEOUT
        self.max_retries = settings.METADATA_MAX_RETRIES

    def _build_client(self) -> httpx.AsyncClient:
        """Create an async HTTP client with timeout and retry configuration."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )

    async def _make_request(
        self, params: Dict[str, Any]
    ) -> ProviderResult:
        """Make HTTP request to Google Books API with retry logic."""
        if self.api_key:
            params["key"] = self.api_key

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._build_client() as client:
                    response = await client.get(self.base_url, params=params)

                if response.status_code == 429:
                    logger.warning(
                        f"Google Books API rate limited (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    if attempt < self.max_retries:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(
                        success=False,
                        error="Rate limit exceeded",
                    )

                if response.status_code >= 500:
                    logger.warning(
                        f"Google Books API server error: {response.status_code} (attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(
                        success=False,
                        error=f"Server error: {response.status_code}",
                    )

                if response.status_code == 404:
                    return ProviderResult(
                        success=True,
                        data={"items": []},
                    )

                if response.status_code == 400:
                    return ProviderResult(
                        success=False,
                        error=f"Bad request: {response.text}",
                    )

                response.raise_for_status()

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
                logger.warning(f"Google Books API timeout (attempt {attempt + 1})")
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"Google Books API request error: {e}")
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue

        return ProviderResult(
            success=False,
            error=f"Request failed after retries: {last_error}",
        )

    def _extract_volume_info(self, volume: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize volume info from Google Books response."""
        info = volume.get("volumeInfo", {})
        sale_info = volume.get("saleInfo", {})
        access_info = volume.get("accessInfo", {})

        # Extract ISBNs
        isbn_10 = None
        isbn_13 = None
        for identifier in info.get("industryIdentifiers", []):
            if identifier.get("type") == "ISBN_10":
                isbn_10 = identifier.get("identifier")
            elif identifier.get("type") == "ISBN_13":
                isbn_13 = identifier.get("identifier")

        # Extract cover image URLs
        cover_url = None
        image_links = info.get("imageLinks", {})
        if image_links:
            # Prefer larger images
            for key in ["extraLarge", "large", "medium", "small", "thumbnail", "smallThumbnail"]:
                if key in image_links:
                    cover_url = image_links[key]
                    break

        return {
            "title": info.get("title"),
            "subtitle": info.get("subtitle"),
            "authors": info.get("authors", []),
            "publisher": info.get("publisher"),
            "published_date": info.get("publishedDate"),
            "description": info.get("description"),
            "isbn_10": isbn_10,
            "isbn_13": isbn_13,
            "page_count": info.get("pageCount"),
            "categories": info.get("categories", []),
            "language": info.get("language"),
            "cover_url": cover_url,
            "source": self.name,
            "source_id": volume.get("id"),
            "confidence": 0.0,  # Will be set by matcher
        }

    async def search_by_isbn(self, isbn: str) -> ProviderResult:
        """Search for a book by ISBN using Google Books API."""
        params = {"q": f"isbn:{isbn}"}
        result = await self._make_request(params)

        if not result.success:
            return result

        items = result.data.get("items", []) if result.data else []
        if not items:
            return ProviderResult(success=True, data={"candidates": []})

        candidates = [self._extract_volume_info(item) for item in items]
        return ProviderResult(success=True, data={"candidates": candidates}, raw_response=result.raw_response)

    async def search_by_title_author(
        self, title: str, author: Optional[str] = None
    ) -> ProviderResult:
        """Search for books by title and optional author."""
        query_parts = [f'intitle:"{title}"']
        if author:
            query_parts.append(f'inauthor:"{author}"')
        params = {"q": " ".join(query_parts)}
        result = await self._make_request(params)

        if not result.success:
            return result

        items = result.data.get("items", []) if result.data else []
        if not items:
            return ProviderResult(success=True, data={"candidates": []})

        candidates = [self._extract_volume_info(item) for item in items]
        return ProviderResult(success=True, data={"candidates": candidates}, raw_response=result.raw_response)

    async def get_book_details(self, provider_id: str) -> ProviderResult:
        """Get detailed book information by Google Books volume ID."""
        url = f"{self.base_url}/{provider_id}"
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                params = {}
                if self.api_key:
                    params["key"] = self.api_key

                async with self._build_client() as client:
                    response = await client.get(url, params=params)

                if response.status_code == 429:
                    if attempt < self.max_retries:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(success=False, error="Rate limit exceeded")

                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ProviderResult(success=False, error=f"Server error: {response.status_code}")

                if response.status_code == 404:
                    return ProviderResult(success=True, data={"candidates": []})

                response.raise_for_status()

                try:
                    data = response.json()
                except Exception as e:
                    return ProviderResult(success=False, error=f"Invalid JSON response: {e}")

                candidate = self._extract_volume_info(data)
                return ProviderResult(
                    success=True,
                    data={"candidates": [candidate]},
                    raw_response=data,
                )

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue

        return ProviderResult(
            success=False,
            error=f"Request failed after retries: {last_error}",
        )