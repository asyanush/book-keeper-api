"""Base interface for metadata providers."""

from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    """Result from a provider search operation."""

    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    raw_response: Optional[dict] = None


class MetadataProvider(ABC):
    """Abstract base class for book metadata providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'google_books', 'open_library')."""
        pass

    @abstractmethod
    async def search_by_isbn(self, isbn: str) -> ProviderResult:
        """
        Search for a book by ISBN.

        Args:
            isbn: ISBN-10 or ISBN-13 (normalized, without hyphens/spaces)

        Returns:
            ProviderResult with success flag, data, and error if any.
        """
        pass

    @abstractmethod
    async def search_by_title_author(
        self, title: str, author: Optional[str] = None
    ) -> ProviderResult:
        """
        Search for books by title and optional author.

        Args:
            title: Book title
            author: Author name (optional)

        Returns:
            ProviderResult with success flag, data, and error if any.
        """
        pass

    @abstractmethod
    async def get_book_details(self, provider_id: str) -> ProviderResult:
        """
        Get detailed book information by provider-specific ID.

        Args:
            provider_id: Provider-specific identifier (e.g., Google Books volume ID,
                         Open Library edition ID)

        Returns:
            ProviderResult with success flag, data, and error if any.
        """
        pass

    def _handle_error(self, error: Exception, operation: str) -> ProviderResult:
        """Handle and log provider errors consistently."""
        logger.warning(
            f"{self.name} provider error during {operation}: {type(error).__name__}: {error}"
        )
        return ProviderResult(success=False, error=str(error))