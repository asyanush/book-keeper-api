"""Pytest fixtures for the deterministic cover-parser test suite.

The cover parser is pure (no I/O, no network, no database), so the fixtures
here are intentionally lightweight.  The :func:`normalizer` fixture mirrors the
one defined for the metadata tests (``app/books/metadata/tests/conftest.py``)
because pytest scopes ``conftest.py`` fixtures to their own directory subtree.

All ISBN constants used by the suite are checksum-valid so the parser's
``MetadataNormalizer.normalize_isbn`` call accepts them.
"""

from __future__ import annotations

import pytest

from app.books.metadata.normalizer import MetadataNormalizer

# ---------------------------------------------------------------------------
# ISBN constants
# ---------------------------------------------------------------------------

#: Valid ISBN-13 for *Range* by David Epstein (the canonical cover example).
VALID_ISBN_13 = "9780735211292"

#: Valid ISBN-10 for the same book.
VALID_ISBN_10 = "0735211299"

#: Valid ISBN-13 for *Clean Code* by Robert C. Martin.
VALID_ISBN_13_CLEAN_CODE = "9780132350884"

#: Same digits as :data:`VALID_ISBN_13` but with a broken checksum.
INVALID_ISBN_13 = "9780735211291"

#: Valid ISBN-10 whose check digit is ``X``.
VALID_ISBN_10_WITH_X = "097522980X"


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> MetadataNormalizer:
    """A stateless ``MetadataNormalizer`` instance.

    Equivalent to the ``normalizer`` fixture in the metadata test suite and
    used to cross-check the ISBN values returned by the cover parser.
    """
    return MetadataNormalizer()


# ---------------------------------------------------------------------------
# Scalar ISBN fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_isbn_13() -> str:
    """A valid, digit-only ISBN-13."""
    return VALID_ISBN_13


@pytest.fixture
def valid_isbn_10() -> str:
    """A valid, digit-only ISBN-10."""
    return VALID_ISBN_10


@pytest.fixture
def invalid_isbn_13() -> str:
    """An ISBN-13 with an invalid checksum."""
    return INVALID_ISBN_13


# ---------------------------------------------------------------------------
# Cover-text fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def range_cover_text() -> str:
    """The canonical cover from the feature request.

    ``RANGE`` / ``WHY GENERALISTS TRIUMPH`` / ``DAVID EPSTEIN``
    """
    return "RANGE\nWHY GENERALISTS TRIUMPH\nDAVID EPSTEIN"


@pytest.fixture
def clean_code_cover_text() -> str:
    """A clean cover with a descriptive (mixed-case) subtitle."""
    return (
        "CLEAN CODE\n"
        "A Handbook of Agile Software Craftsmanship\n"
        "Robert C. Martin"
    )
