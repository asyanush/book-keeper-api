"""Metadata retrieval package."""

from app.books.metadata.schemas import (
    BookMetadata,
    MetadataSearchRequest,
    MetadataSearchResponse,
    MetadataCandidateResponse,
    MetadataErrorResponse,
)
from app.books.metadata.normalizer import MetadataNormalizer
from app.books.metadata.matcher import CandidateMatcher
from app.books.metadata.cache import MetadataCache
from app.books.metadata.service import BookMetadataRetriever, retrieve_book_metadata

__all__ = [
    "BookMetadata",
    "MetadataSearchRequest",
    "MetadataSearchResponse",
    "MetadataCandidateResponse",
    "MetadataErrorResponse",
    "MetadataNormalizer",
    "CandidateMatcher",
    "MetadataCache",
    "BookMetadataRetriever",
    "retrieve_book_metadata",
]