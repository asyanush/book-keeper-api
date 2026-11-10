"""Unit tests for the metadata normaliser.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 15):

* Normalize complete Google Books response.
* Normalize complete Open Library response.
* Handle missing optional fields.
* Normalize Unicode characters.
* Remove duplicate authors.
* Remove duplicate categories.
* Normalize whitespace.
* Validate ISBN-10.
* Validate ISBN-13.
* Reject invalid ISBN checksums.
* Normalize ISBNs containing hyphens and spaces.
* Preserve publication date precision.
* Preserve provider source attribution.
* Do not invent missing metadata.

The normaliser is pure (no I/O), so these tests are deterministic and use the
reusable fixtures declared in ``conftest.py``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.books.metadata.normalizer import MetadataNormalizer
from app.books.metadata.schemas import BookMetadata

# ---------------------------------------------------------------------------
# ISBN examples from the roadmap (Part 2, Section 15)
# ---------------------------------------------------------------------------

ISBN_13_HYPHENATED = "978-0-13-595705-9"
ISBN_13_DIGITS = "9780135957059"
ISBN_10_HYPHENATED = "0-321-12521-5"
ISBN_10_DIGITS = "0321125215"


# ===========================================================================
# 1. ISBN-10 validation
# ===========================================================================


class TestValidateIsbn10:
    """``MetadataNormalizer.validate_isbn10`` checksum validation."""

    def test_valid_isbn10_with_hyphens(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.validate_isbn10(ISBN_10_HYPHENATED) is True

    def test_valid_isbn10_without_separators(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.validate_isbn10(ISBN_10_DIGITS) is True

    def test_valid_isbn10_with_spaces(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.validate_isbn10("0 321 12521 5") is True

    def test_valid_isbn10_fixture(self, normalizer: MetadataNormalizer, valid_isbn_10: str) -> None:
        assert normalizer.validate_isbn10(valid_isbn_10) is True

    def test_rejects_isbn10_with_bad_checksum(
        self, normalizer: MetadataNormalizer
    ) -> None:
        # Same digits as a valid ISBN-10 but with the checksum digit changed.
        assert normalizer.validate_isbn10("0321125214") is False

    def test_rejects_isbn10_with_all_letters(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.validate_isbn10("abcdefghij") is False

    @pytest.mark.parametrize("isbn", ["", "123", "123456789", "12345678901"])
    def test_rejects_isbn10_with_wrong_length(
        self, normalizer: MetadataNormalizer, isbn: str
    ) -> None:
        assert normalizer.validate_isbn10(isbn) is False

    def test_rejects_invalid_isbn_fixture(
        self, normalizer: MetadataNormalizer, invalid_isbn: str
    ) -> None:
        # ``invalid_isbn`` is an invalid ISBN-13, so it is not a valid ISBN-10 either.
        assert normalizer.validate_isbn10(invalid_isbn) is False

    def test_rejects_all_invalid_isbn_cases(
        self, normalizer: MetadataNormalizer, invalid_isbn_cases: list[dict[str, str]]
    ) -> None:
        for case in invalid_isbn_cases:
            assert normalizer.validate_isbn10(case["isbn"]) is False, (
                f"Expected ISBN-10 validation to reject {case['isbn']!r} "
                f"({case['reason']})"
            )


# ===========================================================================
# 2. ISBN-13 validation
# ===========================================================================


class TestValidateIsbn13:
    """``MetadataNormalizer.validate_isbn13`` checksum validation."""

    def test_valid_isbn13_with_hyphens(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.validate_isbn13(ISBN_13_HYPHENATED) is True

    def test_valid_isbn13_without_separators(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.validate_isbn13(ISBN_13_DIGITS) is True

    def test_valid_isbn13_with_spaces(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.validate_isbn13("978 0 13 595705 9") is True

    def test_valid_isbn13_fixture(
        self, normalizer: MetadataNormalizer, valid_isbn: str
    ) -> None:
        assert normalizer.validate_isbn13(valid_isbn) is True

    def test_rejects_isbn13_with_bad_checksum(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.validate_isbn13("9780135957058") is False

    def test_rejects_hyphenated_isbn13_with_bad_checksum(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.validate_isbn13("978-0-13-595705-8") is False

    @pytest.mark.parametrize("isbn", ["", "123", "978013595705", "97801359570590"])
    def test_rejects_isbn13_with_wrong_length(
        self, normalizer: MetadataNormalizer, isbn: str
    ) -> None:
        assert normalizer.validate_isbn13(isbn) is False

    def test_rejects_invalid_isbn_fixture(
        self, normalizer: MetadataNormalizer, invalid_isbn: str
    ) -> None:
        assert normalizer.validate_isbn13(invalid_isbn) is False

    def test_rejects_all_invalid_isbn_cases(
        self, normalizer: MetadataNormalizer, invalid_isbn_cases: list[dict[str, str]]
    ) -> None:
        for case in invalid_isbn_cases:
            assert normalizer.validate_isbn13(case["isbn"]) is False, (
                f"Expected ISBN-13 validation to reject {case['isbn']!r} "
                f"({case['reason']})"
            )


# ===========================================================================
# 3. ISBN normalization
# ===========================================================================


class TestNormalizeIsbn:
    """``MetadataNormalizer.normalize_isbn`` strips separators and validates."""

    def test_normalizes_hyphenated_isbn13(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_isbn(ISBN_13_HYPHENATED) == ISBN_13_DIGITS

    def test_normalizes_digit_only_isbn13(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_isbn(ISBN_13_DIGITS) == ISBN_13_DIGITS

    def test_normalizes_hyphenated_isbn10(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_isbn(ISBN_10_HYPHENATED) == ISBN_10_DIGITS

    def test_normalizes_digit_only_isbn10(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_isbn(ISBN_10_DIGITS) == ISBN_10_DIGITS

    def test_normalizes_spaced_isbn13(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_isbn("978 0 13 595705 9") == ISBN_13_DIGITS

    def test_normalizes_spaced_isbn10(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_isbn("0 321 12521 5") == ISBN_10_DIGITS

    def test_strips_leading_and_trailing_whitespace(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_isbn(f"  {ISBN_13_HYPHENATED}  ") == ISBN_13_DIGITS

    def test_normalizes_fixture_isbns(
        self,
        normalizer: MetadataNormalizer,
        valid_isbn: str,
        valid_isbn_10: str,
    ) -> None:
        assert normalizer.normalize_isbn(valid_isbn) == valid_isbn
        assert normalizer.normalize_isbn(valid_isbn_10) == valid_isbn_10

    def test_accepts_every_valid_isbn_case(
        self, normalizer: MetadataNormalizer, valid_isbn_cases: list[dict[str, str]]
    ) -> None:
        for case in valid_isbn_cases:
            result = normalizer.normalize_isbn(case["isbn"])
            assert result is not None, (
                f"Expected {case['isbn']!r} ({case['format']}) to normalize"
            )
            assert result.isdigit(), f"Normalized ISBN should be digit-only: {result!r}"
            assert len(result) in (10, 13), (
                f"Unexpected ISBN length for {case['isbn']!r}: {result!r}"
            )

    def test_rejects_invalid_isbn_fixture(
        self, normalizer: MetadataNormalizer, invalid_isbn: str
    ) -> None:
        assert normalizer.normalize_isbn(invalid_isbn) is None

    def test_rejects_all_invalid_isbn_cases(
        self, normalizer: MetadataNormalizer, invalid_isbns: list[str]
    ) -> None:
        for isbn in invalid_isbns:
            assert normalizer.normalize_isbn(isbn) is None, (
                f"Expected {isbn!r} to be rejected as invalid"
            )

    @pytest.mark.parametrize(
        "isbn",
        ["", "   ", "-", "  -  ", "97801359570590", "12345"],
    )
    def test_rejects_empty_or_malformed_input(
        self, normalizer: MetadataNormalizer, isbn: str
    ) -> None:
        assert normalizer.normalize_isbn(isbn) is None


# ===========================================================================
# 4. Whitespace and Unicode normalization (NFKC)
# ===========================================================================


class TestNormalizeWhitespace:
    """``MetadataNormalizer.normalize_whitespace`` collapses and normalises."""

    def test_collapses_internal_whitespace(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_whitespace("Clean     Code") == "Clean Code"

    def test_collapses_tabs_and_newlines(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_whitespace("Clean\t\r\n  Code") == "Clean Code"

    def test_strips_leading_and_trailing_whitespace(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace("   Clean Code   ") == "Clean Code"

    def test_collapses_non_breaking_spaces(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace("Clean\u00a0\u00a0Code") == "Clean Code"

    def test_empty_string_returns_empty_string(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace("") == ""

    def test_whitespace_only_returns_empty_string(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace(" \t\n ") == ""

    def test_nfkc_normalizes_fullwidth_characters(
        self, normalizer: MetadataNormalizer
    ) -> None:
        # Full-width Latin letters and the ideographic space (U+3000).
        assert normalizer.normalize_whitespace("Ｃｌｅａｎ\u3000Ｃｏｄｅ") == "Clean Code"

    def test_nfkc_composes_combining_characters(
        self, normalizer: MetadataNormalizer
    ) -> None:
        # "e" + combining acute accent normalises to the precomposed "é".
        assert normalizer.normalize_whitespace("Cafe\u0301") == "Café"

    def test_nfkc_expands_ligatures(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_whitespace("The \ufb01nal cut") == "The final cut"

    def test_nfkc_converts_circled_digits(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace("Volume \u2460") == "Volume 1"

    def test_unicode_letters_are_preserved(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_whitespace("  José  Saramago ") == "José Saramago"


# ===========================================================================
# 5. Author normalization
# ===========================================================================


class TestNormalizeAuthors:
    """``MetadataNormalizer.normalize_authors`` de-duplicates and trims."""

    def test_removes_duplicate_authors_preserving_order(
        self, normalizer: MetadataNormalizer
    ) -> None:
        authors = [
            "Robert C. Martin",
            "Martin Fowler",
            "Robert C. Martin",
            "Kent Beck",
            "Martin Fowler",
        ]

        assert normalizer.normalize_authors(authors) == [
            "Robert C. Martin",
            "Martin Fowler",
            "Kent Beck",
        ]

    def test_removes_consecutive_duplicates(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_authors(
            ["Robert C. Martin", "Robert C. Martin", "Robert C. Martin"]
        ) == ["Robert C. Martin"]

    def test_trims_whitespace_from_author_names(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_authors(
            ["  Robert C. Martin  ", "\tMartin Fowler\n"]
        ) == ["Robert C. Martin", "Martin Fowler"]

    def test_drops_empty_author_names(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_authors(
            ["", "   ", "Robert C. Martin", "\t"]
        ) == ["Robert C. Martin"]

    def test_empty_list_returns_empty_list(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_authors([]) == []

    def test_preserves_unicode_author_names(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_authors(["José Saramago", "José Saramago"]) == [
            "José Saramago"
        ]

    def test_preserves_distinct_names(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_authors(
            ["Robert C. Martin", "Robert Cecil Martin"]
        ) == ["Robert C. Martin", "Robert Cecil Martin"]


# ===========================================================================
# 6. Category normalization
# ===========================================================================


class TestNormalizeCategories:
    """``MetadataNormalizer.normalize_categories`` de-duplicates categories."""

    def test_removes_duplicate_categories_preserving_order(
        self, normalizer: MetadataNormalizer
    ) -> None:
        categories = [
            "Computers",
            "Software Engineering",
            "Computers",
            "Programming",
            "Software Engineering",
        ]

        assert normalizer.normalize_categories(categories) == [
            "Computers",
            "Software Engineering",
            "Programming",
        ]

    def test_deduplicates_case_insensitively(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_categories(
            ["Computers", "computers", "COMPUTERS"]
        ) == ["Computers"]

    def test_keeps_first_occurrence_case(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_categories(
            ["software engineering", "Software Engineering"]
        ) == ["software engineering"]

    def test_trims_whitespace_from_categories(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_categories(
            ["  Computers  ", "\tProgramming\n"]
        ) == ["Computers", "Programming"]

    def test_drops_empty_categories(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_categories(
            ["", "  ", "Computers", "\t"]
        ) == ["Computers"]

    def test_empty_list_returns_empty_list(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_categories([]) == []

    def test_preserves_distinct_categories(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_categories(
            ["Software Engineering", "Software Architecture"]
        ) == ["Software Engineering", "Software Architecture"]


# ===========================================================================
# 7. Date normalization
# ===========================================================================


class TestNormalizeDate:
    """``MetadataNormalizer.normalize_date`` preserves original precision."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2008", "2008"),
            ("2008-08", "2008-08"),
            ("2008-08-01", "2008-08-01"),
        ],
    )
    def test_preserves_date_precision(
        self, normalizer: MetadataNormalizer, raw: str, expected: str
    ) -> None:
        assert normalizer.normalize_date(raw) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  2008  ", "2008"),
            ("2008-08-01 ", "2008-08-01"),
            ("2008-08-01T00:00:00Z", "2008-08-01"),
            (" 2008-08-01T12:34:56 ", "2008-08-01"),
        ],
    )
    def test_normalizes_date_whitespace_and_time_component(
        self, normalizer: MetadataNormalizer, raw: str, expected: str
    ) -> None:
        assert normalizer.normalize_date(raw) == expected

    def test_empty_string_returns_none(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_date("") is None

    def test_whitespace_only_returns_none(
        self, normalizer: MetadataNormalizer
    ) -> None:
        assert normalizer.normalize_date("   \t ") is None

    def test_none_returns_none(self, normalizer: MetadataNormalizer) -> None:
        assert normalizer.normalize_date(None) is None  # type: ignore[arg-type]

    def test_does_not_invent_missing_date_parts(
        self, normalizer: MetadataNormalizer
    ) -> None:
        # A year-only value must not be expanded into a full date.
        result = normalizer.normalize_date("2008")
        assert result == "2008"
        assert result is not None and len(result) == 4

    @pytest.mark.parametrize(
        "raw",
        ["Jan 2023", "August 1, 2008", "circa 1990", "Spring 2010"],
    )
    def test_non_iso_dates_are_preserved_verbatim(
        self, normalizer: MetadataNormalizer, raw: str
    ) -> None:
        assert normalizer.normalize_date(raw) == raw


# ===========================================================================
# 8. Full candidate normalization - Google Books
# ===========================================================================


class TestNormalizeGoogleBooksCandidate:
    """``normalize_candidate`` on a complete Google Books candidate."""

    def test_returns_book_metadata_instance(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)
        assert isinstance(metadata, BookMetadata)

    def test_normalizes_all_fields(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)

        assert metadata.title == "Clean Code"
        assert metadata.subtitle == "A Handbook of Agile Software Craftsmanship"
        assert metadata.authors == ["Robert C. Martin"]
        assert metadata.publisher == "Prentice Hall"
        assert metadata.published_date == "2008-08-01"
        assert metadata.description is not None
        assert metadata.description.startswith("Even bad code can function")
        assert metadata.page_count == 464
        assert metadata.categories == ["Computers", "Software Engineering"]
        assert metadata.language == "en"
        assert metadata.cover_url is not None
        assert metadata.cover_url.startswith("https://books.google.com/")

    def test_normalizes_isbns(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
        valid_isbn: str,
        valid_isbn_10: str,
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)

        assert metadata.isbn_13 == valid_isbn
        assert metadata.isbn_10 == valid_isbn_10

    def test_preserves_source_attribution(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)

        assert metadata.source == "google_books"
        assert metadata.source_id == "zyTCAlFPjgYC"
        assert metadata.confidence == 0.0

    def test_does_not_mutate_input_candidate(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        original = copy.deepcopy(google_books_candidate)

        normalizer.normalize_candidate(google_books_candidate)

        assert google_books_candidate == original


# ===========================================================================
# 9. Full candidate normalization - Open Library
# ===========================================================================


class TestNormalizeOpenLibraryCandidate:
    """``normalize_candidate`` on a complete Open Library candidate."""

    def test_returns_book_metadata_instance(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)
        assert isinstance(metadata, BookMetadata)

    def test_normalizes_all_fields(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)

        assert metadata.title == "Clean Code"
        assert metadata.subtitle == "A Handbook of Agile Software Craftsmanship"
        assert metadata.authors == ["Robert C. Martin"]
        assert metadata.publisher == "Prentice Hall"
        assert metadata.published_date == "2008"
        assert metadata.description is not None
        assert metadata.description.startswith("Even bad code can function")
        assert metadata.page_count == 464
        assert metadata.categories == [
            "Computer programming",
            "Software engineering",
            "Agile software development",
        ]
        # The provider's language code is preserved verbatim (not mapped to ISO 639-1).
        assert metadata.language == "eng"
        assert metadata.cover_url == "https://covers.openlibrary.org/b/id/1234567-L.jpg"

    def test_normalizes_isbns(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)

        assert metadata.isbn_10 == "0132350882"
        assert metadata.isbn_13 == "9780132350884"

    def test_preserves_source_attribution(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)

        assert metadata.source == "open_library"
        assert metadata.source_id == "OL123456M"

    def test_preserves_year_only_publication_date(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)

        # Open Library supplies only a year; the normaliser must not invent
        # a month or day.
        assert metadata.published_date == "2008"


# ===========================================================================
# 10. Missing optional fields / no fabricated metadata
# ===========================================================================


class TestMissingFields:
    """Missing metadata stays missing; nothing is invented."""

    def test_candidate_with_only_source_leaves_optional_fields_empty(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate({"source": "google_books"})

        assert metadata.title is None
        assert metadata.subtitle is None
        assert metadata.authors == []
        assert metadata.publisher is None
        assert metadata.published_date is None
        assert metadata.description is None
        assert metadata.isbn_10 is None
        assert metadata.isbn_13 is None
        assert metadata.page_count is None
        assert metadata.categories == []
        assert metadata.language is None
        assert metadata.cover_url is None
        assert metadata.source == "google_books"
        assert metadata.source_id is None

    def test_empty_candidate_does_not_invent_metadata(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate({})

        assert metadata.title is None
        assert metadata.authors == []
        assert metadata.categories == []
        assert metadata.isbn_10 is None
        assert metadata.isbn_13 is None
        assert metadata.page_count is None
        assert metadata.published_date is None
        # No provider supplied, so the explicit "unknown" fallback is used.
        assert metadata.source == "unknown"

    def test_missing_isbns_are_not_fabricated(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate.pop("isbn_10", None)
        candidate.pop("isbn_13", None)

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.title == "Clean Code"
        assert metadata.isbn_10 is None
        assert metadata.isbn_13 is None

    def test_missing_authors_and_categories_become_empty_lists(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate.pop("authors", None)
        candidate.pop("categories", None)

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.authors == []
        assert metadata.categories == []

    def test_missing_optional_scalar_fields_become_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        for field in (
            "subtitle",
            "publisher",
            "published_date",
            "description",
            "page_count",
            "language",
            "cover_url",
            "source_id",
        ):
            candidate.pop(field, None)

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.subtitle is None
        assert metadata.publisher is None
        assert metadata.published_date is None
        assert metadata.description is None
        assert metadata.page_count is None
        assert metadata.language is None
        assert metadata.cover_url is None
        assert metadata.source_id is None
        # Retained fields are still present.
        assert metadata.title == "Clean Code"
        assert metadata.authors == ["Robert C. Martin"]

    def test_non_list_authors_and_categories_default_to_empty(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {
                "title": "Clean Code",
                "authors": "Robert C. Martin",  # wrong type
                "categories": "Computers",  # wrong type
                "source": "google_books",
            }
        )

        assert metadata.authors == []
        assert metadata.categories == []

    def test_empty_title_is_not_replaced_with_placeholder(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {"title": "   ", "source": "google_books"}
        )

        assert metadata.title is None


# ===========================================================================
# 11. Unicode, duplicates and whitespace in a full candidate
# ===========================================================================


class TestCandidateTextNormalization:
    """Text/list normalisation applied by ``normalize_candidate``."""

    def test_normalizes_unicode_title_authors_and_publisher(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {
                "title": "  Ｃｌｅａｎ\u3000Ｃｏｄｅ  ",
                "subtitle": "A Handbook\t of Agile  Software Craftsmanship",
                "authors": ["  José Saramago  "],
                "publisher": "  Penguin   Books ",
                "source": "google_books",
            }
        )

        assert metadata.title == "Clean Code"
        assert metadata.subtitle == "A Handbook of Agile Software Craftsmanship"
        assert metadata.authors == ["José Saramago"]
        assert metadata.publisher == "Penguin Books"

    def test_normalizes_whitespace_in_description_and_language(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {
                "title": "Clean Code",
                "description": "Line one.\n\n  Line two.",
                "language": "  en  ",
                "source": "google_books",
            }
        )

        assert metadata.description == "Line one. Line two."
        assert metadata.language == "en"

    def test_removes_duplicate_authors_from_candidate(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["authors"] = [
            "Robert C. Martin",
            "Robert C. Martin",
            "Martin Fowler",
            " Robert C. Martin ",
        ]

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.authors == ["Robert C. Martin", "Martin Fowler"]

    def test_removes_duplicate_categories_from_candidate(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["categories"] = [
            "Computers",
            "computers",
            "Software Engineering",
            "SOFTWARE ENGINEERING",
            "Programming",
        ]

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.categories == [
            "Computers",
            "Software Engineering",
            "Programming",
        ]

    def test_duplicate_removal_preserves_order(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["authors"] = ["Zoe", "Adam", "Zoe", "Mia", "Adam"]
        candidate["categories"] = ["Zeta", "Alpha", "Zeta", "Mu", "Alpha"]

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.authors == ["Zoe", "Adam", "Mia"]
        assert metadata.categories == ["Zeta", "Alpha", "Mu"]


# ===========================================================================
# 12. ISBN normalization in a full candidate
# ===========================================================================


class TestCandidateIsbnNormalization:
    """ISBNs inside a candidate are stripped and checksum-validated."""

    def test_normalizes_hyphenated_isbns_in_candidate(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["isbn_13"] = ISBN_13_HYPHENATED
        candidate["isbn_10"] = ISBN_10_HYPHENATED

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.isbn_13 == ISBN_13_DIGITS
        assert metadata.isbn_10 == ISBN_10_DIGITS

    def test_normalizes_spaced_isbns_in_candidate(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["isbn_13"] = "978 0 13 595705 9"
        candidate["isbn_10"] = "0 321 12521 5"

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.isbn_13 == ISBN_13_DIGITS
        assert metadata.isbn_10 == ISBN_10_DIGITS

    def test_invalid_isbn13_checksum_becomes_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["isbn_13"] = "9780135957058"

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.isbn_13 is None
        # Other metadata survives even though the ISBN was rejected.
        assert metadata.title == "Clean Code"
        assert metadata.isbn_10 == "0132350882"

    def test_invalid_isbn10_checksum_becomes_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["isbn_10"] = "0321125214"

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.isbn_10 is None
        assert metadata.isbn_13 == "9780132350884"

    def test_malformed_isbns_become_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["isbn_10"] = "not-an-isbn"
        candidate["isbn_13"] = "123"

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.isbn_10 is None
        assert metadata.isbn_13 is None


# ===========================================================================
# 13. Publication date precision in a full candidate
# ===========================================================================


class TestCandidateDatePrecision:
    """``normalize_candidate`` preserves whatever precision was supplied."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2008", "2008"),
            ("2008-08", "2008-08"),
            ("2008-08-01", "2008-08-01"),
        ],
    )
    def test_preserves_publication_date_precision(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
        raw: str,
        expected: str,
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["published_date"] = raw

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.published_date == expected

    def test_missing_publication_date_stays_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate.pop("published_date", None)

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.published_date is None


# ===========================================================================
# 14. Source attribution provenance
# ===========================================================================


class TestSourceAttribution:
    """Provider provenance is always preserved, never overwritten."""

    def test_preserves_google_books_source(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(google_books_candidate)

        assert metadata.source == "google_books"
        assert metadata.source_id == "zyTCAlFPjgYC"

    def test_preserves_open_library_source(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        metadata = normalizer.normalize_candidate(open_library_candidate)

        assert metadata.source == "open_library"
        assert metadata.source_id == "OL123456M"

    def test_preserves_arbitrary_source_value(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {"title": "Some Book", "source": "custom_provider", "source_id": "abc-123"}
        )

        assert metadata.source == "custom_provider"
        assert metadata.source_id == "abc-123"

    def test_source_survives_when_isbn_is_rejected(
        self,
        normalizer: MetadataNormalizer,
        open_library_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(open_library_candidate)
        candidate["isbn_13"] = "9780135957058"  # invalid checksum

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.source == "open_library"
        assert metadata.source_id == "OL123456M"

    def test_source_survives_when_optional_fields_missing(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {"title": "Anonymous", "source": "google_books"}
        )

        assert metadata.source == "google_books"


# ===========================================================================
# 15. Scalar coercion and confidence handling
# ===========================================================================


class TestCandidateScalarCoercion:
    """Page count and confidence are coerced defensively."""

    def test_integral_float_page_count_is_coerced_to_int(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["page_count"] = 464.0

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.page_count == 464
        assert isinstance(metadata.page_count, int)

    @pytest.mark.parametrize("page_count", [0, -1, "464", None, [], {}])
    def test_invalid_page_counts_become_none(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
        page_count: Any,
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["page_count"] = page_count

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.page_count is None

    def test_preserves_numeric_confidence(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        candidate["confidence"] = 0.92

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.confidence == pytest.approx(0.92)

    def test_invalid_confidence_defaults_to_zero(
        self, normalizer: MetadataNormalizer
    ) -> None:
        metadata = normalizer.normalize_candidate(
            {"title": "Clean Code", "source": "google_books", "confidence": "high"}
        )

        assert metadata.confidence == 0.0

    def test_preserves_cover_url_verbatim(
        self,
        normalizer: MetadataNormalizer,
        google_books_candidate: dict[str, Any],
    ) -> None:
        candidate = copy.deepcopy(google_books_candidate)
        url = "https://example.test/cover.jpg?size=large&lang=en"
        candidate["cover_url"] = url

        metadata = normalizer.normalize_candidate(candidate)

        assert metadata.cover_url == url


# ===========================================================================
# 16. Idempotence
# ===========================================================================


class TestNormalizationIdempotence:
    """Normalizing already-normalized data must not change it."""

    @pytest.mark.parametrize("fixture_name", ["google_books_candidate", "open_library_candidate"])
    def test_normalizing_twice_is_stable(
        self,
        request: pytest.FixtureRequest,
        normalizer: MetadataNormalizer,
        fixture_name: str,
    ) -> None:
        candidate = request.getfixturevalue(fixture_name)

        first = normalizer.normalize_candidate(candidate)
        second = normalizer.normalize_candidate(first.model_dump())

        assert second == first
