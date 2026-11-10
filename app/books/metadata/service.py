"""Metadata retrieval orchestrator.

Coordinates the full retrieval pipeline:

1. Check cache
2. Query primary API providers (Google Books, Open Library)
3. Normalise and match candidates
4. Optionally invoke scraping fallback
5. Select a single candidate when confidence is unambiguous
6. Cache successful results

Provider failures are handled independently – one failing provider
never prevents another from contributing candidates.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.books.exceptions import MetadataServiceUnavailable
from app.books.metadata.cache import MetadataCache
from app.books.metadata.matcher import CandidateMatcher
from app.books.metadata.normalizer import MetadataNormalizer
from app.books.metadata.providers.base import ProviderResult
from app.books.metadata.providers.google_books import GoogleBooksProvider
from app.books.metadata.providers.open_library import OpenLibraryProvider
from app.books.metadata.providers.scraper import MetadataScraper
from app.books.metadata.schemas import (
    BookMetadata,
    MetadataCandidateResponse,
    MetadataSearchRequest,
    MetadataSearchResponse,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _provider_label(provider_name: str) -> str:
    """Return a human-readable label bundle for internal status tracking."""
    return provider_name


def _build_candidate_response(candidate: BookMetadata) -> MetadataCandidateResponse:
    """Convert a ``BookMetadata`` instance into an API response model."""
    return MetadataCandidateResponse(
        title=candidate.title,
        subtitle=candidate.subtitle,
        authors=candidate.authors,
        publisher=candidate.publisher,
        published_date=candidate.published_date,
        isbn_13=candidate.isbn_13,
        isbn_10=candidate.isbn_10,
        page_count=candidate.page_count,
        categories=candidate.categories,
        language=candidate.language,
        cover_url=candidate.cover_url,
        source=candidate.source,
        confidence=candidate.confidence,
    )


# ------------------------------------------------------------------
# Provider result collection helpers
# ------------------------------------------------------------------


def _extract_candidates_from_result(
    result: ProviderResult,
) -> List[Dict[str, Any]]:
    """Safely extract raw candidate dicts from a ``ProviderResult``."""
    if not result.success or not result.data:
        return []
    candidates = result.data.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    return candidates


# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------


class BookMetadataRetriever:
    """Top-level orchestrator for book metadata retrieval.

    Call :meth:`retrieve` with an ISBN, title, and/or author to
    get a fully assembled ``MetadataSearchResponse`` with ranked
    candidates and (optionally) a single selected candidate.
    """

    def __init__(self) -> None:
        self._google = GoogleBooksProvider()
        self._open_library = OpenLibraryProvider()
        self._normalizer = MetadataNormalizer()
        self._matcher = CandidateMatcher()
        self._cache = MetadataCache()
        self._scraper = MetadataScraper()
        self._strategy = settings.METADATA_STRATEGY.upper()

    # ------------------------------------------------------------------
    # Single-provider query helpers
    # ------------------------------------------------------------------

    async def _query_google(
        self,
        isbn: Optional[str],
        title: Optional[str],
        author: Optional[str],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Query Google Books and return (provider_reachable, candidates)."""
        try:
            if isbn:
                result = await self._google.search_by_isbn(isbn)
            else:
                result = await self._google.search_by_title_author(
                    title or "", author
                )
            return True, _extract_candidates_from_result(result)
        except Exception as exc:
            logger.warning(f"Google Books provider failed: {exc}")
            return False, []

    async def _query_open_library(
        self,
        isbn: Optional[str],
        title: Optional[str],
        author: Optional[str],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Query Open Library and return (provider_reachable, candidates)."""
        try:
            if isbn:
                result = await self._open_library.search_by_isbn(isbn)
            else:
                result = await self._open_library.search_by_title_author(
                    title or "", author
                )
            return True, _extract_candidates_from_result(result)
        except Exception as exc:
            logger.warning(f"Open Library provider failed: {exc}")
            return False, []

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_all(
        self, raw_candidates: List[Dict[str, Any]]
    ) -> List[BookMetadata]:
        """Normalise a list of raw candidate dicts into ``BookMetadata``."""
        norm: List[BookMetadata] = []
        for raw in raw_candidates:
            try:
                norm.append(self._normalizer.normalize_candidate(raw))
            except Exception as exc:
                logger.warning(
                    f"Failed to normalise candidate from {raw.get('source', '?')}: {exc}"
                )
        return norm

    # ------------------------------------------------------------------
    # Scraping fallback
    # ------------------------------------------------------------------

    def _run_scraper(
        self,
        isbn: Optional[str],
        title: Optional[str],
        author: Optional[str],
    ) -> List[BookMetadata]:
        """Invoke the scraper (synchronous) and return normalised results."""
        if not settings.SCRAPING_ENABLED:
            return []

        try:
            if isbn:
                scraped = self._scraper.search_by_isbn(isbn)
            elif title:
                scraped = self._scraper.search_by_title_author(title, author)
            else:
                return []
        except Exception as exc:
            logger.warning(f"Scraper failed: {exc}")
            return []

        # Normalise scraper results
        normalized: List[BookMetadata] = []
        for candidate in scraped:
            try:
                normalized.append(self._normalizer.normalize_candidate(
                    candidate.model_dump(exclude_none=False)
                ))
            except Exception as exc:
                logger.warning(f"Failed to normalise scraper candidate: {exc}")
        return normalized

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_candidate(
        candidates: List[BookMetadata],
    ) -> Optional[BookMetadata]:
        """Select a single candidate when confidence is clearly highest.

        A candidate is only selected when:
        - Its confidence is > 0.85
        - It is significantly higher than the next-best candidate
          (at least 0.15 above the runner-up, or the runner-up is
          below 0.6)

        Returns ``None`` when the situation is ambiguous.
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            sole = candidates[0]
            if sole.confidence > 0.85:
                return sole
            return None

        top = candidates[0]
        runner_up = candidates[1]

        # Top confidence must be > 0.85
        if top.confidence <= 0.85:
            return None

        # Difference from runner-up must be significant
        gap = top.confidence - runner_up.confidence
        if gap >= 0.15 or runner_up.confidence < 0.6:
            return top

        return None

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        isbn: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
    ) -> MetadataSearchResponse:
        """Run the full metadata retrieval pipeline.

        Args:
            isbn: Normalised ISBN (no hyphens/spaces), optional.
            title: Book title, optional.
            author: Author name, optional.

        Returns:
            ``MetadataSearchResponse`` with status, query, ranked
            candidates, and (when appropriate) a selected candidate.
        """
        query = MetadataSearchRequest(
            isbn=isbn,
            title=title,
            author=author,
        )

        # ------------------------------------------------------------------
        # 1. Check cache
        # ------------------------------------------------------------------
        cached = self._cache.get(isbn=isbn, title=title, author=author)
        if cached:
            # Cached results were already matched/normalised when stored,
            # but we still need to re-run selection since the selection
            # criteria are stateful.
            selected = self._select_candidate(cached)
            return MetadataSearchResponse(
                status="success",
                query=query,
                candidates=[_build_candidate_response(c) for c in cached],
                selected_candidate=(
                    _build_candidate_response(selected) if selected else None
                ),
            )

        # ------------------------------------------------------------------
        # 2. Determine which providers to query
        # ------------------------------------------------------------------
        # Per spec: prefer ISBN search on Google Books first when ISBN
        # is available, then Open Library.  For title/author queries
        # both providers are queried.
        raw_candidates: Dict[str, List[Dict[str, Any]]] = {}
        tasks: List[tuple] = []

        strategy = self._strategy

        # Build provider query tasks based on strategy
        if strategy in ("HYBRID", "GOOGLE_ONLY"):
            tasks.append(("google_books", self._query_google(isbn, title, author)))
        if strategy in ("HYBRID", "OPEN_LIBRARY_ONLY"):
            tasks.append(("open_library", self._query_open_library(isbn, title, author)))

        # Run all enabled provider queries concurrently
        provider_reachable = False  # Track if ANY provider was reachable
        if tasks:
            names, coros = zip(*tasks)
            results = await asyncio.gather(*coros, return_exceptions=True)
            for name, res in zip(names, results):
                if isinstance(res, Exception):
                    logger.warning(f"Provider '{name}' raised: {res}")
                    raw_candidates[name] = []
                elif isinstance(res, tuple) and len(res) == 2:
                    reachable, candidates = res
                    raw_candidates[name] = candidates
                    if reachable:
                        provider_reachable = True
                else:
                    raw_candidates[name] = []

        # ------------------------------------------------------------------
        # 3. Normalise all collected candidates
        # ------------------------------------------------------------------
        all_normalized: List[BookMetadata] = []
        for provider_name, raws in raw_candidates.items():
            normalized = self._normalize_all(raws)
            all_normalized.extend(normalized)

        # ------------------------------------------------------------------
        # 4. Match candidates (score + deduplicate)
        # ------------------------------------------------------------------
        query_dict: Dict[str, Any] = {}
        if isbn:
            query_dict["isbn"] = isbn
        if title:
            query_dict["title"] = title
        if author:
            query_dict["author"] = author

        matched = self._matcher.match_candidates(query_dict, all_normalized)

        # ------------------------------------------------------------------
        # 5. Scraping fallback (only if API results insufficient)
        # ------------------------------------------------------------------
        if settings.SCRAPING_ENABLED:
            # Invoke scraping only if we have fewer than 2 matched
            # candidates or the top confidence is below 0.5.
            need_scrape = len(matched) < 2
            if matched and matched[0].confidence >= 0.7:
                need_scrape = False

            if need_scrape:
                scraped = self._run_scraper(isbn, title, author)
                if scraped:
                    # Score scraped candidates and merge
                    scraped_matched = self._matcher.match_candidates(
                        query_dict, scraped
                    )
                    matched.extend(scraped_matched)
                    # Re-deduplicate after merging
                    matched = self._matcher.detect_duplicates(matched)
                    # Re-sort by confidence descending
                    matched.sort(
                        key=lambda c: (c.confidence, c.title or ""),
                        reverse=True,
                    )

        # ------------------------------------------------------------------
        # 6. Check for complete provider failure (503 condition)
        # ------------------------------------------------------------------
        if not matched and not provider_reachable:
            # Cache was empty (we already checked) and no provider was
            # reachable – the caller cannot determine whether results
            # exist or not.  Signal a service outage.
            raise MetadataServiceUnavailable()

        # ------------------------------------------------------------------
        # 7. Select candidate (or not, if ambiguous)
        # ------------------------------------------------------------------
        selected = self._select_candidate(matched)

        # ------------------------------------------------------------------
        # 8. Cache successful results
        # ------------------------------------------------------------------
        if matched:
            # Determine the provider for cache attribution – use the
            # provider that contributed the most high-confidence results.
            provider_counts: Dict[str, int] = {}
            for c in matched:
                provider_counts[c.source] = provider_counts.get(c.source, 0) + 1
            primary_provider = max(provider_counts, key=provider_counts.get)
            self._cache.set(
                isbn=isbn,
                title=title,
                author=author,
                candidates=matched,
                provider=primary_provider,
            )

        # ------------------------------------------------------------------
        # 7. Build and return response
        # ------------------------------------------------------------------
        return MetadataSearchResponse(
            status="success",
            query=query,
            candidates=[_build_candidate_response(c) for c in matched],
            selected_candidate=(
                _build_candidate_response(selected) if selected else None
            ),
        )


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------


async def retrieve_book_metadata(
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> dict:
    """Retrieve book metadata from external providers with caching.

    Convenience wrapper around :class:`BookMetadataRetriever` that
    returns a plain dict suitable for direct API response handling.

    Args:
        isbn: Normalised ISBN (no hyphens/spaces), optional.
        title: Book title, optional.
        author: Author name, optional.

    Returns:
        Dict with keys ``status``, ``query``, ``candidates``, and
        ``selected_candidate``.

    Raises:
        MetadataServiceUnavailable: When all external providers fail
            and no cached results exist.
    """
    retriever = BookMetadataRetriever()
    response = await retriever.retrieve(isbn=isbn, title=title, author=author)
    return response.model_dump(exclude_none=False)