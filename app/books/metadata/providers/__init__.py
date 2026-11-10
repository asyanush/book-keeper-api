"""Metadata providers package."""

from app.books.metadata.providers.base import MetadataProvider, ProviderResult
from app.books.metadata.providers.google_books import GoogleBooksProvider
from app.books.metadata.providers.open_library import OpenLibraryProvider
from app.books.metadata.providers.scraper import MetadataScraper

__all__ = [
    "MetadataProvider",
    "ProviderResult",
    "GoogleBooksProvider",
    "OpenLibraryProvider",
    "MetadataScraper",
]