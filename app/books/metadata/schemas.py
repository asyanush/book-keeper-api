"""Pydantic schemas for book metadata retrieval."""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class BookMetadata(BaseModel):
    """Normalized book metadata from any provider."""

    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    description: Optional[str] = None
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    page_count: Optional[int] = None
    categories: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    cover_url: Optional[str] = None
    source: str
    source_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("isbn_10", "isbn_13", mode="before")
    @classmethod
    def normalize_isbn(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Remove hyphens and spaces
        cleaned = v.replace("-", "").replace(" ", "")
        return cleaned if cleaned else None

    @field_validator("authors", "categories", mode="before")
    @classmethod
    def deduplicate_list(cls, v: List[str]) -> List[str]:
        if not v:
            return []
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for item in v:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @field_validator("title", "subtitle", "publisher", "description", "language", mode="before")
    @classmethod
    def normalize_string(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Normalize whitespace and Unicode
        import unicodedata
        normalized = unicodedata.normalize("NFKC", v)
        # Collapse multiple whitespace into single space
        import re
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized if normalized else None


class MetadataSearchRequest(BaseModel):
    """Request schema for metadata search endpoint."""

    isbn: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None

    @field_validator("isbn", mode="before")
    @classmethod
    def normalize_isbn_input(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.replace("-", "").replace(" ", "")
        return cleaned if cleaned else None

    @field_validator("isbn")
    @classmethod
    def validate_isbn_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISBN checksum when an ISBN is supplied.

        Supports ISBN-10 and ISBN-13 with valid checksums.  Returns
        the normalised digit-only ISBN or raises ``ValueError`` when
        the checksum is invalid.
        """
        if v is None:
            return None

        # Lazy import to avoid circular import issues at module level.
        from app.books.metadata.normalizer import MetadataNormalizer

        # Must be all digits (possibly with trailing 'X' for ISBN-10).
        has_x = v.upper().endswith("X")
        digits_part = v[:-1] if has_x else v
        if not digits_part.isdigit():
            raise ValueError(
                f"Invalid ISBN format: '{v}'. "
                "ISBN must contain only digits (and optionally a trailing 'X' for ISBN-10)."
            )

        length = len(v)
        if length not in (10, 13):
            raise ValueError(
                f"Invalid ISBN length: {length} characters. "
                "ISBN must be 10 or 13 digits."
            )

        if length == 10 and not MetadataNormalizer.validate_isbn10(v):
            raise ValueError(
                f"Invalid ISBN-10 checksum: '{v}'. "
                "The checksum digit does not match."
            )

        if length == 13 and not MetadataNormalizer.validate_isbn13(v):
            raise ValueError(
                f"Invalid ISBN-13 checksum: '{v}'. "
                "The checksum digit does not match."
            )

        return v


class MetadataCandidateResponse(BaseModel):
    """Single candidate book metadata in response."""

    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    isbn_13: Optional[str] = None
    isbn_10: Optional[str] = None
    page_count: Optional[int] = None
    categories: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    cover_url: Optional[str] = None
    source: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MetadataSearchResponse(BaseModel):
    """Response schema for metadata search endpoint."""

    status: str = "success"
    query: MetadataSearchRequest
    candidates: List[MetadataCandidateResponse] = Field(default_factory=list)
    selected_candidate: Optional[MetadataCandidateResponse] = None


class MetadataErrorResponse(BaseModel):
    """Error response for metadata search."""

    status: str = "error"
    error: str
    detail: Optional[str] = None