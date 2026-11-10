"""Metadata normalization module.

Normalises raw provider candidate dictionaries into consistent
``BookMetadata`` instances.  Handles ISBN checksum validation,
Unicode normalisation, whitespace collapse, and de-duplication of
lists such as authors and categories.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Union

from app.books.metadata.schemas import BookMetadata

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_NON_DIGIT = re.compile(r"[^\d]")


def _digits(text: str) -> str:
    """Return only the digit characters from *text*."""
    return _NON_DIGIT.sub("", text)


# ------------------------------------------------------------------
# MetadataNormalizer
# ------------------------------------------------------------------


class MetadataNormalizer:
    """Normalises raw provider candidate data into ``BookMetadata``.

    Each normalisation method is independent so callers can compose
    them as needed.  The primary entry point for a full candidate
    is :meth:`normalize_candidate`.
    """

    # ------------------------------------------------------------------
    # ISBN validation & normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_isbn10(isbn: str) -> bool:
        """Validate an ISBN-10 checksum.

        The checksum digit may be 0-9 or 'X' (representing 10).

        Args:
            isbn: Raw ISBN-10 string (may contain hyphens/spaces).

        Returns:
            ``True`` when the checksum is valid.
        """
        cleaned = _digits(isbn.upper())

        if len(cleaned) != 10:
            return False

        total = 0
        for i, ch in enumerate(cleaned):
            weight = 10 - i
            if ch == "X":
                if i != 9:  # 'X' is only valid in the final position
                    return False
                digit = 10
            else:
                digit = int(ch)
            total += weight * digit

        return total % 11 == 0

    @staticmethod
    def validate_isbn13(isbn: str) -> bool:
        """Validate an ISBN-13 checksum.

        Uses the EAN-13 algorithm: alternating weights 1 and 3,
        checksum must be divisible by 10.

        Args:
            isbn: Raw ISBN-13 string (may contain hyphens/spaces).

        Returns:
            ``True`` when the checksum is valid.
        """
        cleaned = _digits(isbn)

        if len(cleaned) != 13:
            return False

        total = 0
        for i, ch in enumerate(cleaned):
            digit = int(ch)
            weight = 1 if i % 2 == 0 else 3
            total += weight * digit

        return total % 10 == 0

    @classmethod
    def normalize_isbn(cls, isbn: str) -> Optional[str]:
        """Normalize an ISBN by removing hyphens and spaces, then validate.

        Returns the digit-only string if the checksum is valid for
        either ISBN-10 or ISBN-13 length, otherwise ``None``.

        Args:
            isbn: Raw ISBN string.

        Returns:
            Normalised digit-only ISBN, or ``None`` if invalid.
        """
        cleaned = _digits(isbn)

        if not cleaned:
            return None

        if len(cleaned) == 10:
            if cls.validate_isbn10(cleaned):
                return cleaned
            logger.debug(f"ISBN-10 checksum failed for: {isbn!r}")
            return None

        if len(cleaned) == 13:
            if cls.validate_isbn13(cleaned):
                return cleaned
            logger.debug(f"ISBN-13 checksum failed for: {isbn!r}")
            return None

        logger.debug(f"Unrecognised ISBN length ({len(cleaned)}): {isbn!r}")
        return None

    # ------------------------------------------------------------------
    # Text normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse whitespace and normalise Unicode to NFKC form.

        Args:
            text: Arbitrary string.

        Returns:
            Cleaned string with collapsed whitespace and NFKC normalisation.
        """
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    # ------------------------------------------------------------------
    # List normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_authors(authors: List[str]) -> List[str]:
        """Deduplicate authors while preserving order and trimming whitespace.

        Args:
            authors: Raw author name list.

        Returns:
            Deduplicated, trimmed list of non-empty author names.
        """
        seen: set = set()
        result: List[str] = []
        for author in authors:
            trimmed = author.strip()
            if trimmed and trimmed not in seen:
                seen.add(trimmed)
                result.append(trimmed)
        return result

    @staticmethod
    def normalize_categories(categories: List[str]) -> List[str]:
        """Deduplicate categories, trim, and normalise case.

        Categories are title-cased for consistency while preserving
        acronyms (e.g. 'AI' stays 'AI', 'Computers' stays 'Computers').

        Args:
            categories: Raw category / subject name list.

        Returns:
            Deduplicated, trimmed, title-cased list of non-empty categories.
        """
        seen: set = set()
        result: List[str] = []
        for cat in categories:
            trimmed = cat.strip()
            if not trimmed:
                continue
            # Normalise to lowercase for dedup comparison but keep
            # original case in the result.
            lower = trimmed.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(trimmed)
        return result

    # ------------------------------------------------------------------
    # Date normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_date(date_str: str) -> Optional[str]:
        """Normalise a publication date while preserving original precision.

        Does *not* invent missing parts.  Recognised patterns:

        * ``YYYY-MM-DD`` → kept as-is
        * ``YYYY-MM``   → kept as-is
        * ``YYYY``      → kept as-is
        * Everything else is normalised for whitespace and returned.

        Args:
            date_str: Raw date string from a provider.

        Returns:
            Normalised date string or ``None`` if the input is empty.
        """
        if not date_str:
            return None

        cleaned = MetadataNormalizer.normalize_whitespace(date_str)
        if not cleaned:
            return None

        # Common identifiable patterns – preserve as-is:
        #   YYYY-MM-DD, YYYY-MM, YYYY
        if re.match(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$", cleaned):
            return cleaned

        # For anything else (e.g. "Jan 2023", "2023-01-01T00:00:00Z"),
        # try to extract a reasonable date component.  We do *not*
        # fabricate parts that are absent.
        m = re.match(r"^(\d{4})", cleaned)
        if m:
            # Strip time portion if present
            date_part = cleaned.split("T")[0].split(" ")[0]
            if re.match(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?", date_part):
                return date_part
            return m.group(1)

        return cleaned

    # ------------------------------------------------------------------
    # Candidate normalisation (main entry point)
    # ------------------------------------------------------------------

    @classmethod
    def normalize_candidate(cls, candidate: Dict[str, Any]) -> BookMetadata:
        """Normalize a raw provider candidate dict into a ``BookMetadata``.

        Processes every field through the appropriate normaliser while
        preserving provider-specific metadata and provenance.  Missing
        fields are left as ``None`` / empty lists – nothing is
        fabricated.

        Args:
            candidate: Raw candidate dictionary from a provider.

        Returns:
            Normalised ``BookMetadata`` instance.
        """
        # ISBNs – run through checksum validation
        isbn_10 = candidate.get("isbn_10")
        isbn_13 = candidate.get("isbn_13")

        if isinstance(isbn_10, str):
            isbn_10 = cls.normalize_isbn(isbn_10)
        if isinstance(isbn_13, str):
            isbn_13 = cls.normalize_isbn(isbn_13)

        # Text fields that benefit from whitespace / Unicode normalisation
        title = candidate.get("title")
        subtitle = candidate.get("subtitle")
        publisher = candidate.get("publisher")
        description = candidate.get("description")
        language = candidate.get("language")

        title = cls.normalize_whitespace(title) if isinstance(title, str) else None
        subtitle = cls.normalize_whitespace(subtitle) if isinstance(subtitle, str) else None
        publisher = cls.normalize_whitespace(publisher) if isinstance(publisher, str) else None
        description = cls.normalize_whitespace(description) if isinstance(description, str) else None
        language = cls.normalize_whitespace(language) if isinstance(language, str) else None

        # Lists
        authors_raw = candidate.get("authors", [])
        authors = cls.normalize_authors(authors_raw if isinstance(authors_raw, list) else [])

        categories_raw = candidate.get("categories", [])
        categories = cls.normalize_categories(categories_raw if isinstance(categories_raw, list) else [])

        # Date
        published_date = candidate.get("published_date")
        published_date = cls.normalize_date(published_date) if isinstance(published_date, str) else None

        # Page count – coerce to int or leave as None
        page_count = candidate.get("page_count")
        if isinstance(page_count, (int, float)):
            page_count = int(page_count) if page_count > 0 else None
        else:
            page_count = None

        # Cover URL – keep as-is, no normalisation needed
        cover_url = candidate.get("cover_url")

        # Source attribution (required – always preserve)
        source = candidate.get("source", "unknown")
        source_id = candidate.get("source_id")

        # Confidence – preserve existing value; matcher sets it later
        confidence = candidate.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        return BookMetadata(
            title=title,
            subtitle=subtitle,
            authors=authors,
            publisher=publisher,
            published_date=published_date,
            description=description,
            isbn_10=isbn_10,
            isbn_13=isbn_13,
            page_count=page_count,
            categories=categories,
            language=language,
            cover_url=cover_url,
            source=source,
            source_id=source_id,
            confidence=float(confidence),
        )