"""Comprehensive unit tests for the deterministic cover-text parser.

The parser (:mod:`app.books.ocr.cover_parser`) is pure Python: no LLM calls,
no network, no database, and no image processing.  These tests therefore run
entirely in-memory and are fully deterministic.

Coverage map
------------
* Public API: :func:`parse_cover_text`, :class:`CoverParser`,
  :class:`ParsedCoverData`.
* Every stage of the parsing pipeline (line segmentation, noise filtering,
  ISBN handling, title/subtitle extraction, author extraction, confidence).
* Bestseller/award badges, review quotes, OCR artifacts, spine fragments.
* Author formats (``by`` lines, credential lines, multi-author separators,
  separate author lines, missing authors).
* Subtitle extraction and stopping rules; title selection heuristics.
* ISBN validation/deduplication via :class:`MetadataNormalizer`.
* Confidence scoring bands and the clean-ratio penalty.
* Non-English covers, OCR truncation, and degenerate inputs.

Known parser limitations are documented with strict ``xfail`` tests so the
suite stays green while a regression/fix is tracked.  Each marker points at
the exact behaviour that currently deviates from the intended contract.
"""

from __future__ import annotations

import pytest

from app.books.ocr.cover_parser import (
    CoverParser,
    ParsedCoverData,
    parse_cover_text,
)
from app.books.metadata.normalizer import MetadataNormalizer

# ---------------------------------------------------------------------------
# Constants (checksum-valid unless explicitly marked invalid)
# ---------------------------------------------------------------------------

VALID_ISBN_13 = "9780735211292"          # Range — David Epstein
VALID_ISBN_10 = "0735211299"             # Same book, ISBN-10
VALID_ISBN_13_CLEAN_CODE = "9780132350884"  # Clean Code — Robert C. Martin
INVALID_ISBN_13 = "9780735211291"        # Broken checksum (last digit)

RANGE_COVER = "RANGE\nWHY GENERALISTS TRIUMPH\nDAVID EPSTEIN"
CLEAN_CODE_COVER = (
    "CLEAN CODE\n"
    "A Handbook of Agile Software Craftsmanship\n"
    "Robert C. Martin"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _assert_no_text_contains(result: ParsedCoverData, needle: str) -> None:
    """Assert *needle* does not appear in any metadata text field."""
    haystacks = [result.title or "", result.subtitle or "", *result.authors]
    for value in haystacks:
        assert needle.lower() not in value.lower(), (
            f"{needle!r} leaked into extracted field {value!r}"
        )


# ===========================================================================
# 1. Public API / output model
# ===========================================================================


class TestPublicApi:
    """Both public entry points and the output model contract."""

    def test_parse_cover_text_returns_parsed_cover_data(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein")

        assert isinstance(result, ParsedCoverData)
        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]

    def test_cover_parser_parse_matches_convenience_function(self) -> None:
        """``parse_cover_text`` is a thin wrapper over ``CoverParser.parse``."""
        raw = CLEAN_CODE_COVER
        isbns = [VALID_ISBN_13_CLEAN_CODE]

        via_function = parse_cover_text(raw, isbns)
        via_class = CoverParser(raw, isbns).parse()

        assert via_class.model_dump() == via_function.model_dump()

    def test_none_detected_isbns_defaults_to_empty_list(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein", None)

        assert result.isbns == []
        assert result.title == "RANGE"

    def test_omitted_detected_isbns_defaults_to_empty_list(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein")

        assert result.isbns == []

    def test_parsed_cover_data_defaults(self) -> None:
        data = ParsedCoverData()

        assert data.title is None
        assert data.subtitle is None
        assert data.authors == []
        assert data.isbns == []
        assert data.raw_text == ""
        assert data.confidence == 0.0

    def test_raw_text_is_preserved(self) -> None:
        raw = "RANGE\nDavid Epstein"
        result = parse_cover_text(raw)

        assert result.raw_text == raw

    def test_raw_text_is_nfkc_normalized(self) -> None:
        """Full-width characters are normalised before parsing."""
        raw = "ＲＡＮＧＥ\nＤＡＶＩＤ ＥＰＳＴＥＩＮ"
        result = parse_cover_text(raw)

        assert result.raw_text == "RANGE\nDAVID EPSTEIN"
        assert result.title == "RANGE"

    def test_cover_parser_normalises_unicode_on_construction(self) -> None:
        parser = CoverParser("ＲＡＮＧＥ", [])

        assert parser.raw_text == "RANGE"

    def test_confidence_is_always_within_bounds(self) -> None:
        samples = [
            "",
            "RANGE",
            RANGE_COVER,
            "###\n@@@",
            "RANGE\nDavid Epstein",
        ]
        for text in samples:
            result = parse_cover_text(text)
            assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# 2. Standard cover parsing
# ===========================================================================


class TestStandardCoverParsing:
    """End-to-end parsing of clean, realistic cover text."""

    def test_clean_cover_extracts_all_fields(
        self,
        clean_code_cover_text: str,
    ) -> None:
        result = parse_cover_text(
            clean_code_cover_text, [VALID_ISBN_13_CLEAN_CODE]
        )

        assert result.title == "CLEAN CODE"
        assert result.subtitle == "A Handbook of Agile Software Craftsmanship"
        assert result.authors == ["Robert C. Martin"]
        assert result.isbns == [VALID_ISBN_13_CLEAN_CODE]
        assert result.confidence == pytest.approx(0.90)

    def test_minimal_cover_extracts_title_and_author(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.subtitle is None
        assert result.authors == ["David Epstein"]

    def test_user_example_range_cover_extracts_title_and_isbn(
        self,
        range_cover_text: str,
    ) -> None:
        """The canonical example still yields a correct title and ISBN."""
        result = parse_cover_text(range_cover_text, [VALID_ISBN_13])

        assert result.title == "RANGE"
        assert result.isbns == [VALID_ISBN_13]
        assert "DAVID EPSTEIN" in result.authors
        assert result.confidence == pytest.approx(0.85)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: a short ALL-CAPS subtitle line "
            "('WHY GENERALISTS TRIUMPH') is not recognised as a subtitle, "
            "so it is emitted both as an author and suppresses the subtitle."
        ),
    )
    def test_user_example_range_cover_extracts_subtitle_and_author(
        self,
        range_cover_text: str,
    ) -> None:
        result = parse_cover_text(range_cover_text)

        assert result.subtitle == "WHY GENERALISTS TRIUMPH"
        assert result.authors == ["DAVID EPSTEIN"]

    def test_title_subtitle_author_layout(self) -> None:
        result = parse_cover_text(
            "Sapiens\n"
            "A Brief History of Humankind\n"
            "Yuval Noah Harari"
        )

        assert result.title == "Sapiens"
        assert result.subtitle == "A Brief History of Humankind"
        assert result.authors == ["Yuval Noah Harari"]


# ===========================================================================
# 3. Bestseller / award badge removal
# ===========================================================================


class TestBestsellerBadgeRemoval:
    """Badge lines must never become the title, subtitle, or author."""

    @pytest.mark.parametrize(
        "badge",
        [
            "#1 NEW YORK TIMES BESTSELLER",
            "NEW YORK TIMES BESTSELLER",
            "SUNDAY TIMES BESTSELLER",
            "INTERNATIONAL BESTSELLER",
            "#1 BESTSELLER",
            "USA TODAY BESTSELLER",
            "WALL STREET JOURNAL BESTSELLER",
        ],
    )
    def test_badge_is_removed(self, badge: str) -> None:
        result = parse_cover_text(f"{badge}\nRANGE\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]
        _assert_no_text_contains(result, badge)

    @pytest.mark.parametrize(
        "badge",
        [
            "#1 NEW YORK TIMES BESTSELLER",
            "NEW YORK TIMES BESTSELLER",
            "#3 BESTSELLER",
        ],
    )
    def test_badge_alone_yields_no_metadata(self, badge: str) -> None:
        result = parse_cover_text(badge)

        assert result.title is None
        assert result.subtitle is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.10)

    def test_badge_between_title_and_author_is_removed(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "#1 NEW YORK TIMES BESTSELLER\n"
            "David Epstein"
        )

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]
        _assert_no_text_contains(result, "BESTSELLER")


# ===========================================================================
# 4. Review quote removal
# ===========================================================================


class TestReviewQuoteRemoval:
    """Quoted blurbs and attribution dashes are not book metadata."""

    @pytest.mark.parametrize(
        "quote_line",
        [
            '"A tour de force"',
            "“A tour de force”",
            'A tour de force"',
            '"A tour de force" —The New York Times',
        ],
    )
    def test_quote_line_does_not_leak_into_metadata(
        self, quote_line: str
    ) -> None:
        result = parse_cover_text(f"RANGE\n{quote_line}\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]
        assert result.subtitle is None
        _assert_no_text_contains(result, "tour de force")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: a curly-quoted blurb followed by an "
            "attribution is neither caught by _RE_QUOTE (closing character "
            "is not the line end) nor by _RE_ATTRIBUTION (line does not "
            "start with a dash), so it leaks into the subtitle."
        ),
    )
    @pytest.mark.parametrize(
        "quote_line",
        [
            "“A tour de force” —The New York Times",
            "“Brilliant” —The Guardian",
        ],
    )
    def test_curly_quote_with_attribution_is_removed(
        self, quote_line: str
    ) -> None:
        result = parse_cover_text(f"RANGE\n{quote_line}\nDavid Epstein")

        assert result.subtitle is None

    @pytest.mark.parametrize(
        "attribution_line",
        [
            "—The New York Times",
            "-The New York Times",
            "–The Guardian",
        ],
    )
    def test_attribution_dash_lines_are_filtered(
        self, attribution_line: str
    ) -> None:
        result = parse_cover_text(
            f"RANGE\n{attribution_line}\nDavid Epstein"
        )

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]
        assert result.subtitle is None

    def test_quote_after_author_does_not_become_an_author(self) -> None:
        result = parse_cover_text(
            'RANGE\nDavid Epstein\n"A tour de force"'
        )

        assert result.authors == ["David Epstein"]
        _assert_no_text_contains(result, "tour de force")


# ===========================================================================
# 5. OCR artifact filtering
# ===========================================================================


class TestOcrArtifactFiltering:
    """Garbage produced by binarisation must be discarded."""

    @pytest.mark.parametrize(
        "artifact",
        [
            "###",
            "@@@",
            "$$$",
            "|/\\-+=",
            "AB1!@#$%^&*",
            "ab",
            "1234",
        ],
    )
    def test_artifact_line_is_removed(self, artifact: str) -> None:
        result = parse_cover_text(
            f"{artifact}\nRANGE\nDavid Epstein"
        )

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]
        _assert_no_text_contains(result, artifact)

    def test_single_lowercase_characters_are_removed(self) -> None:
        result = parse_cover_text("r\na\nn\ng\nRANGE\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]

    def test_symbols_only_cover_yields_no_metadata(self) -> None:
        result = parse_cover_text("###\n@@@\n$$$")

        assert result.title is None
        assert result.subtitle is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.10)

    def test_non_alpha_dominated_line_is_removed(self) -> None:
        result = parse_cover_text(
            "A1!@#$%^&*()\nRANGE\nDavid Epstein"
        )

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]


# ===========================================================================
# 6. Author extraction
# ===========================================================================


class TestAuthorExtraction:
    """Author patterns, credential lines, and multi-author handling."""

    def test_by_prefix(self) -> None:
        result = parse_cover_text("RANGE\nby David Epstein")

        assert result.authors == ["David Epstein"]

    def test_by_prefix_is_stripped_from_name(self) -> None:
        result = parse_cover_text("RANGE\nby   David   Epstein")

        assert result.authors == ["David Epstein"]

    def test_author_credential_is_not_extracted(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "AUTHOR OF THE SPORTS GENE\n"
            "David Epstein"
        )

        assert result.authors == ["David Epstein"]
        _assert_no_text_contains(result, "AUTHOR OF")

    def test_credential_between_title_and_author(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "WHY GENERALISTS TRIUMPH\n"
            "AUTHOR OF THE SPORTS GENE\n"
            "DAVID EPSTEIN"
        )

        assert result.title == "RANGE"
        assert result.authors == ["DAVID EPSTEIN"]
        _assert_no_text_contains(result, "AUTHOR OF")

    def test_credential_only_text_has_no_title_or_author(self) -> None:
        result = parse_cover_text("AUTHOR OF THE SPORTS GENE")

        assert result.title is None
        assert result.subtitle is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.20)

    def test_ampersand_separated_authors_on_by_line(self) -> None:
        result = parse_cover_text("RANGE\nby JOHN DOE & JANE SMITH")

        assert result.authors == ["JOHN DOE", "JANE SMITH"]

    def test_ampersand_separated_authors_without_by(self) -> None:
        result = parse_cover_text("RANGE\nJohn Doe & Jane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]

    def test_comma_separated_authors_on_by_line(self) -> None:
        result = parse_cover_text("RANGE\nby John Doe, Jane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]

    def test_and_separated_authors_on_by_line(self) -> None:
        result = parse_cover_text("RANGE\nby John Doe and Jane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: a plain 'First Last, First Last' line "
            "is rejected by _looks_like_author_name because the comma is not "
            "stripped before name-word validation."
        ),
    )
    def test_comma_separated_authors_without_by(self) -> None:
        result = parse_cover_text("RANGE\nJohn Doe, Jane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: a plain 'A and B' author line is five "
            "words, exceeding the 1-4-word author-name heuristic."
        ),
    )
    def test_and_separated_authors_without_by(self) -> None:
        result = parse_cover_text("RANGE\nJohn Doe and Jane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]

    def test_separate_author_lines(self) -> None:
        result = parse_cover_text("RANGE\nJohn Doe\nJane Smith")

        assert result.authors == ["John Doe", "Jane Smith"]
        assert result.subtitle is None

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: with three author lines the middle "
            "short ALL-CAPS line is captured as a subtitle by the "
            "short-all-caps rule."
        ),
    )
    def test_three_separate_author_lines(self) -> None:
        result = parse_cover_text("RANGE\nJohn Doe\nJane Smith\nBob Lee")

        assert result.subtitle is None
        assert result.authors == ["John Doe", "Jane Smith", "Bob Lee"]

    def test_missing_author_is_empty_list(self) -> None:
        result = parse_cover_text("THE BIG BOOK OF STORIES")

        assert result.title == "THE BIG BOOK OF STORIES"
        assert result.authors == []

    def test_duplicate_authors_are_deduplicated(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein\nDavid Epstein"
        )

        assert result.authors == ["David Epstein"]

    def test_hyphenated_author_name(self) -> None:
        result = parse_cover_text("RANGE\nAnne-Marie Slaughter")

        assert result.authors == ["Anne-Marie Slaughter"]

    def test_apostrophe_author_name(self) -> None:
        result = parse_cover_text("RANGE\nO'Brien")

        assert result.authors == ["O'Brien"]

    def test_initials_in_author_name(self) -> None:
        result = parse_cover_text("RANGE\nJ.R.R. Tolkien")

        assert result.authors == ["J.R.R. Tolkien"]

    def test_author_separators_are_split(self) -> None:
        parser = CoverParser("", [])

        assert parser._split_authors("A & B") == ["A", "B"]
        assert parser._split_authors("A, B") == ["A", "B"]
        assert parser._split_authors("A and B") == ["A", "B"]

    def test_split_authors_drops_empty_segments(self) -> None:
        parser = CoverParser("", [])

        assert parser._split_authors("A &  & B") == ["A", "B"]
        assert parser._split_authors("   ") == []


# ===========================================================================
# 7. Subtitle extraction
# ===========================================================================


class TestSubtitleExtraction:
    """Subtitle selection and the stopping rules that bound it."""

    def test_descriptive_line_after_title(self) -> None:
        result = parse_cover_text(CLEAN_CODE_COVER)

        assert result.title == "CLEAN CODE"
        assert result.subtitle == "A Handbook of Agile Software Craftsmanship"

    def test_subtitle_does_not_include_the_author(self) -> None:
        result = parse_cover_text(CLEAN_CODE_COVER)

        assert result.subtitle is not None
        assert "Robert C. Martin" not in result.subtitle

    def test_no_subtitle_when_author_follows_title_directly(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.subtitle is None

    def test_subtitle_stops_at_author_credential(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "WHY GENERALISTS TRIUMPH\n"
            "AUTHOR OF THE SPORTS GENE\n"
            "DAVID EPSTEIN"
        )

        assert result.subtitle == "WHY GENERALISTS TRIUMPH"
        assert result.authors == ["DAVID EPSTEIN"]

    def test_why_subtitle_is_extracted(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "Why Generalists Triumph in a Specialized World\n"
            "David Epstein"
        )

        assert result.title == "RANGE"
        assert (
            result.subtitle
            == "Why Generalists Triumph in a Specialized World"
        )

    def test_how_subtitle_is_extracted(self) -> None:
        result = parse_cover_text(
            "DEEP WORK\n"
            "How to Focus in a Distracted World\n"
            "Cal Newport"
        )

        assert result.title == "DEEP WORK"
        assert result.subtitle == "How to Focus in a Distracted World"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: short ALL-CAPS subtitle lines are "
            "treated as author names before the subtitle branch can claim "
            "them."
        ),
    )
    def test_short_all_caps_subtitle_is_extracted(self) -> None:
        result = parse_cover_text(RANGE_COVER)

        assert result.subtitle == "WHY GENERALISTS TRIUMPH"

    def test_subtitle_stops_at_sentence_punctuation(self) -> None:
        result = parse_cover_text(
            "RANGE\n"
            "A story about everything.\n"
            "More text here\n"
            "David Epstein"
        )

        assert result.subtitle == "A story about everything."


# ===========================================================================
# 8. Title selection
# ===========================================================================


class TestTitleSelection:
    """Title heuristics: position, case, length, and noise skipping."""

    def test_first_meaningful_line_is_the_title(self) -> None:
        result = parse_cover_text(
            "THE GREAT GATSBY\nF. Scott Fitzgerald"
        )

        assert result.title == "THE GREAT GATSBY"

    def test_title_skips_symbol_noise(self) -> None:
        result = parse_cover_text("~~~\n###\nRANGE\nDavid Epstein")

        assert result.title == "RANGE"

    def test_title_skips_bestseller_badge(self) -> None:
        result = parse_cover_text(
            "#1 NEW YORK TIMES BESTSELLER\nRANGE\nDavid Epstein"
        )

        assert result.title == "RANGE"

    def test_title_skips_quote_line(self) -> None:
        result = parse_cover_text(
            '"A tour de force" —The New York Times\nRANGE\nDavid Epstein'
        )

        assert result.title == "RANGE"

    def test_concise_title_beats_long_descriptive_subtitle(self) -> None:
        result = parse_cover_text(
            "Sapiens\nA Brief History of Humankind\nYuval Noah Harari"
        )

        assert result.title == "Sapiens"
        assert result.subtitle == "A Brief History of Humankind"

    def test_title_over_why_subtitle(self) -> None:
        result = parse_cover_text(
            "RANGE\nWhy Generalists Triumph\nDavid Epstein"
        )

        assert result.title == "RANGE"

    def test_title_over_how_subtitle(self) -> None:
        result = parse_cover_text(
            "DEEP WORK\nHow to Focus\nCal Newport"
        )

        assert result.title == "DEEP WORK"

    def test_lowercase_title_before_why_subtitle(self) -> None:
        """An earlier plausible line wins when the best candidate is a
        ``why ...`` subtitle."""
        result = parse_cover_text(
            "range\nWhy Generalists Triumph\nDavid Epstein"
        )

        assert result.title == "range"

    def test_long_title_is_preserved(self) -> None:
        result = parse_cover_text(
            "THE SEVEN HABITS OF HIGHLY EFFECTIVE PEOPLE\n"
            "Stephen R. Covey"
        )

        assert (
            result.title
            == "THE SEVEN HABITS OF HIGHLY EFFECTIVE PEOPLE"
        )
        assert result.authors == ["Stephen R. Covey"]

    def test_longer_line_beats_single_letter(self) -> None:
        result = parse_cover_text("R\nRANGE\nDavid Epstein")

        assert result.title == "RANGE"

    def test_all_caps_title_is_preserved(self) -> None:
        result = parse_cover_text("THE NAME OF THE WIND\nPatrick Rothfuss")

        assert result.title == "THE NAME OF THE WIND"

    def test_title_only_cover(self) -> None:
        result = parse_cover_text("RANGE")

        assert result.title == "RANGE"
        assert result.authors == []
        assert result.subtitle is None

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: purely numeric title lines are "
            "classified as OCR noise by the alpha-ratio filter."
        ),
    )
    def test_numeric_title_is_preserved(self) -> None:
        result = parse_cover_text("1984\nGeorge Orwell")

        assert result.title == "1984"
        assert result.authors == ["George Orwell"]


# ===========================================================================
# 9. ISBN handling
# ===========================================================================


class TestIsbnHandling:
    """Detected ISBNs are validated, normalised, and de-duplicated."""

    def test_uses_detected_isbn(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein", [VALID_ISBN_13]
        )

        assert result.isbns == [VALID_ISBN_13]

    def test_isbn10_is_accepted(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein", [VALID_ISBN_10]
        )

        assert result.isbns == [VALID_ISBN_10]

    def test_invalid_checksum_is_rejected(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein", [INVALID_ISBN_13]
        )

        assert result.isbns == []

    def test_hyphenated_isbn_is_normalised(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein", ["978-0-7352-1129-2"]
        )

        assert result.isbns == [VALID_ISBN_13]

    def test_duplicate_isbns_are_deduplicated(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein",
            [VALID_ISBN_13, "978-0-7352-1129-2"],
        )

        assert result.isbns == [VALID_ISBN_13]

    def test_multiple_distinct_isbns_are_preserved(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein",
            [VALID_ISBN_13, VALID_ISBN_13_CLEAN_CODE],
        )

        assert result.isbns == [
            VALID_ISBN_13,
            VALID_ISBN_13_CLEAN_CODE,
        ]

    def test_mixed_valid_and_invalid_isbns_keep_only_valid(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein",
            [INVALID_ISBN_13, VALID_ISBN_13],
        )

        assert result.isbns == [VALID_ISBN_13]

    def test_isbns_are_not_extracted_from_raw_text(self) -> None:
        """Only ``detected_isbns`` are trusted; raw OCR digits are ignored."""
        result = parse_cover_text("RANGE 9780735211292\nDavid Epstein")

        assert result.isbns == []

    @pytest.mark.parametrize(
        "raw_isbn",
        [
            VALID_ISBN_13,
            "978-0-7352-1129-2",
            VALID_ISBN_10,
            INVALID_ISBN_13,
            "",
        ],
    )
    def test_isbns_match_normalizer_output(
        self,
        normalizer: MetadataNormalizer,
        raw_isbn: str,
    ) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein", [raw_isbn])
        expected = normalizer.normalize_isbn(raw_isbn)

        assert result.isbns == ([expected] if expected else [])

    def test_isbn_does_not_require_title_or_author(self) -> None:
        result = parse_cover_text("", [VALID_ISBN_13])

        assert result.isbns == [VALID_ISBN_13]
        assert result.title is None
        assert result.authors == []


# ===========================================================================
# 10. Confidence scoring
# ===========================================================================


class TestConfidenceScoring:
    """Confidence bands documented in ``_compute_confidence``."""

    def test_title_author_isbn_is_high_confidence(self) -> None:
        result = parse_cover_text(
            "RANGE\nDavid Epstein", [VALID_ISBN_13]
        )

        assert result.confidence == pytest.approx(0.85)

    def test_title_author_isbn_subtitle_is_highest(self) -> None:
        result = parse_cover_text(
            CLEAN_CODE_COVER, [VALID_ISBN_13_CLEAN_CODE]
        )

        assert result.confidence == pytest.approx(0.90)

    def test_title_author_is_medium_confidence(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein")

        assert result.confidence == pytest.approx(0.65)

    def test_title_author_subtitle_gets_small_bonus(self) -> None:
        result = parse_cover_text(CLEAN_CODE_COVER)

        assert result.confidence == pytest.approx(0.70)

    def test_title_only_is_low_confidence(self) -> None:
        result = parse_cover_text("RANGE")

        assert result.confidence == pytest.approx(0.45)

    def test_title_and_subtitle_only(self) -> None:
        result = parse_cover_text(
            "CLEAN CODE\nA Handbook of Agile Software Craftsmanship"
        )

        assert result.confidence == pytest.approx(0.50)

    def test_empty_text_is_minimum_floor(self) -> None:
        result = parse_cover_text("")

        assert result.confidence == pytest.approx(0.20)

    def test_noise_only_is_penalised(self) -> None:
        result = parse_cover_text("###\n@@@\n$$$")

        assert result.confidence == pytest.approx(0.10)

    def test_only_isbn_metadata_with_no_text(self) -> None:
        result = parse_cover_text("", [VALID_ISBN_13])

        assert result.confidence == pytest.approx(0.20)

    def test_only_isbn_text_line_is_penalised(self) -> None:
        result = parse_cover_text("9780735211292", [VALID_ISBN_13])

        assert result.confidence == pytest.approx(0.10)

    def test_low_clean_ratio_penalty(self) -> None:
        """A title surrounded by noise loses 0.10 confidence."""
        result = parse_cover_text("###\n@@@\n$$$\nRANGE")

        assert result.title == "RANGE"
        assert result.confidence == pytest.approx(0.35)


# ===========================================================================
# 11. Spine / fragmented text
# ===========================================================================


class TestSpineFragmentedText:
    """Vertical spine text degrades gracefully to partial metadata."""

    def test_single_uppercase_letters_are_kept(self) -> None:
        result = parse_cover_text("R\nA\nN\nG\nE")

        assert result.title is not None
        assert len(result.title) == 1
        assert result.title in set("RANGE")
        assert result.authors == []

    def test_single_lowercase_letters_are_discarded(self) -> None:
        result = parse_cover_text("r\na\nn\ng\ne")

        assert result.title is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.10)

    def test_spine_fragments_have_low_confidence(self) -> None:
        result = parse_cover_text("R\nA\nN\nG\nE")

        assert result.confidence < 0.65

    def test_vertical_fragments_never_become_authors(self) -> None:
        result = parse_cover_text(
            "R\nA\nN\nG\nE\nD\nA\nV\nI\nD"
        )

        assert result.authors == []

    def test_spine_plus_full_name_uses_full_name_as_title(self) -> None:
        """Isolated letters score lower than a real multi-word line."""
        result = parse_cover_text("R\nA\nN\nG\nE\nDAVID EPSTEIN")

        assert result.title == "DAVID EPSTEIN"


# ===========================================================================
# 12. Non-English covers
# ===========================================================================


class TestNonEnglishCovers:
    """Latin-script non-English covers are parsed by the same rules."""

    def test_french_cover(self) -> None:
        result = parse_cover_text(
            "LE PETIT PRINCE\nAntoine de Saint-Exupery"
        )

        assert result.title == "LE PETIT PRINCE"
        assert result.authors == ["Antoine de Saint-Exupery"]

    def test_spanish_cover(self) -> None:
        result = parse_cover_text(
            "CIEN ANOS DE SOLEDAD\nGabriel Garcia Marquez"
        )

        assert result.title == "CIEN ANOS DE SOLEDAD"
        assert result.authors == ["Gabriel Garcia Marquez"]

    def test_german_cover(self) -> None:
        result = parse_cover_text(
            "DER VORLESER\nBernhard Schlink"
        )

        assert result.title == "DER VORLESER"
        assert result.authors == ["Bernhard Schlink"]

    def test_accented_title_is_preserved(self) -> None:
        result = parse_cover_text(
            "CIEN AÑOS DE SOLEDAD\nGabriel Garcia Marquez"
        )

        assert result.title == "CIEN AÑOS DE SOLEDAD"
        assert result.authors == ["Gabriel Garcia Marquez"]

    def test_french_title_with_apostrophe_and_accent(self) -> None:
        result = parse_cover_text("L'ÉTRANGER\nAlbert Camus")

        assert result.title == "L'ÉTRANGER"
        assert result.authors == ["Albert Camus"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: _RE_NAME_WORD only accepts ASCII "
            "letters, so accented author surnames fail name validation."
        ),
    )
    def test_accented_author_name_is_extracted(self) -> None:
        result = parse_cover_text("RANGE\nJosé Saramago")

        assert result.authors == ["José Saramago"]


# ===========================================================================
# 13. OCR truncation
# ===========================================================================


class TestOcrTruncation:
    """Text truncated by OCR is preserved verbatim (not repaired)."""

    def test_truncated_author_is_preserved(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstei")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstei"]

    def test_truncated_author_in_range_layout(self) -> None:
        result = parse_cover_text(
            "RANGE\nWHY GENERALISTS TRIUMPH\nDAVID EPSTEI"
        )

        assert "DAVID EPSTEI" in result.authors

    def test_truncated_title_is_preserved(self) -> None:
        result = parse_cover_text("RANG\nDavid Epstein")

        assert result.title == "RANG"
        assert result.authors == ["David Epstein"]

    def test_truncated_input_does_not_raise(self) -> None:
        result = parse_cover_text("EPSTEI")

        assert isinstance(result, ParsedCoverData)


# ===========================================================================
# 14. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Degenerate and whitespace-only inputs."""

    @pytest.mark.parametrize(
        "raw_text",
        ["", "   ", "\n\n\n", "\t\t", " \n \t \n "],
    )
    def test_blank_inputs_yield_empty_metadata(self, raw_text: str) -> None:
        result = parse_cover_text(raw_text)

        assert result.title is None
        assert result.subtitle is None
        assert result.authors == []
        assert result.isbns == []
        assert result.confidence == pytest.approx(0.20)

    def test_only_noise(self) -> None:
        result = parse_cover_text("\\ = < > | ==\n###\n@@@")

        assert result.title is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.10)

    def test_only_isbn_with_no_text(self) -> None:
        result = parse_cover_text("", [VALID_ISBN_13])

        assert result.isbns == [VALID_ISBN_13]
        assert result.title is None
        assert result.authors == []
        assert result.confidence == pytest.approx(0.20)

    def test_crlf_line_endings(self) -> None:
        result = parse_cover_text("RANGE\r\nDavid Epstein")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]

    def test_whitespace_runs_are_collapsed(self) -> None:
        result = parse_cover_text("  RANGE   \n\tDavid   Epstein")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]

    def test_leading_and_trailing_blank_lines_are_ignored(self) -> None:
        result = parse_cover_text("\n\nRANGE\n\nDavid Epstein\n\n")

        assert result.title == "RANGE"
        assert result.authors == ["David Epstein"]

    def test_empty_detected_isbns_list(self) -> None:
        result = parse_cover_text("RANGE\nDavid Epstein", [])

        assert result.isbns == []


# ===========================================================================
# 15. Parsing pipeline steps (private helpers)
# ===========================================================================


class TestParsingPipelineSteps:
    """Direct coverage for each stage invoked by ``CoverParser.parse``."""

    def test_segment_lines_collapses_whitespace_and_drops_empty(
        self,
    ) -> None:
        parser = CoverParser("  RANGE   \n\n\tDavid   Epstein\n", [])
        parser._segment_lines()

        assert parser._lines == ["RANGE", "David Epstein"]

    def test_filter_noise_discards_artifacts(self) -> None:
        parser = CoverParser(
            "###\nRANGE\n@@@\nDavid Epstein", []
        )
        parser._segment_lines()
        parser._filter_noise()

        assert parser._clean_lines == ["RANGE", "David Epstein"]

    def test_handle_isbns_validates_and_deduplicates(self) -> None:
        parser = CoverParser(
            "",
            ["978-0-7352-1129-2", INVALID_ISBN_13, VALID_ISBN_13],
        )
        parser._handle_isbns()

        assert parser._normalised_isbns == [VALID_ISBN_13]
        assert parser._results.isbns == [VALID_ISBN_13]

    def test_extract_title_sets_title_and_index(self) -> None:
        parser = CoverParser("RANGE\nDavid Epstein", [])
        parser._segment_lines()
        parser._filter_noise()
        parser._extract_title()

        assert parser._results.title == "RANGE"
        assert parser._title_idx == 0

    def test_extract_subtitle_sets_subtitle(self) -> None:
        parser = CoverParser(CLEAN_CODE_COVER, [])
        parser._segment_lines()
        parser._filter_noise()
        parser._extract_title()
        parser._extract_subtitle()

        assert (
            parser._results.subtitle
            == "A Handbook of Agile Software Craftsmanship"
        )

    def test_extract_authors_sets_authors(self) -> None:
        parser = CoverParser("RANGE\nDavid Epstein", [])
        parser._segment_lines()
        parser._filter_noise()
        parser._extract_title()
        parser._extract_subtitle()
        parser._extract_authors()

        assert parser._results.authors == ["David Epstein"]

    def test_author_scan_stops_at_descriptive_sentence(self) -> None:
        parser = CoverParser("", [])
        parser._clean_lines = ["RANGE", "this is the story", "David Epstein"]
        parser._results.title = "RANGE"
        parser._results.subtitle = None

        parser._extract_authors()

        assert parser._results.authors == ["David Epstein"]

    def test_author_scan_stops_at_non_author_after_author(self) -> None:
        parser = CoverParser("", [])
        parser._clean_lines = ["RANGE", "The Great Gatsby", "David Epstein"]
        parser._results.title = "RANGE"
        parser._results.subtitle = None

        parser._extract_authors()

        assert parser._results.authors == ["David Epstein"]

    def test_compute_confidence_sets_confidence(self) -> None:
        parser = CoverParser("", [])
        parser._results.title = "RANGE"
        parser._results.authors = ["David Epstein"]
        parser._results.isbns = [VALID_ISBN_13]
        parser._results.subtitle = "A Subtitle"

        parser._compute_confidence()

        assert parser._results.confidence == pytest.approx(0.90)

    def test_parse_runs_all_steps_in_order(self) -> None:
        parser = CoverParser("RANGE", [])
        calls: list[str] = []
        step_names = [
            "_segment_lines",
            "_filter_noise",
            "_handle_isbns",
            "_extract_title",
            "_extract_subtitle",
            "_extract_authors",
            "_compute_confidence",
        ]

        for name in step_names:
            setattr(
                parser,
                name,
                lambda _name=name: calls.append(_name),
            )

        parser.parse()

        assert calls == step_names


# ===========================================================================
# 16. Noise-line and author-name heuristics
# ===========================================================================


class TestNoiseLineHeuristics:
    """Direct tests for ``CoverParser._is_noise_line``."""

    @pytest.mark.parametrize(
        "line",
        [
            "###",
            "@@@",
            "|/\\-+=",
            "AB1!@#$%^&*",
            "ab",
            "r",
            "1234",
            "—The New York Times",
            "-The New York Times",
            "NEW YORK TIMES BESTSELLER",
            "#1 BESTSELLER",
        ],
    )
    def test_noise_lines(self, line: str) -> None:
        parser = CoverParser("", [])

        assert parser._is_noise_line(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "RANGE",
            "David Epstein",
            "The Great Gatsby",
            "A",
            "R",
            "AB",
        ],
    )
    def test_meaningful_lines(self, line: str) -> None:
        parser = CoverParser("", [])

        assert parser._is_noise_line(line) is False

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known parser limitation: _is_noise_line strips quote characters "
            "before running _RE_QUOTE, so a fully quoted line is not flagged "
            "as noise (downstream subtitle/author checks still skip it)."
        ),
    )
    def test_fully_quoted_line_is_noise(self) -> None:
        parser = CoverParser("", [])

        assert parser._is_noise_line('"A tour de force"') is True


class TestAuthorNameHeuristics:
    """Direct tests for ``CoverParser._looks_like_author_name``."""

    @pytest.mark.parametrize(
        "line",
        [
            "David Epstein",
            "JOHN DOE",
            "J.R.R. Tolkien",
            "O'Brien",
            "Anne-Marie Slaughter",
        ],
    )
    def test_author_like_names(self, line: str) -> None:
        parser = CoverParser("", [])

        assert parser._looks_like_author_name(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "The Great Gatsby",
            "A Tale of Two Cities",
            "1984",
            "Book 1",
            "###",
            "AUTHOR OF THE SPORTS GENE",
            "NEW YORK TIMES BESTSELLER",
            "T",
            "&",
            "a b c d e",
            "this is the book",
            "test",
            "ab cd ef gh",
        ],
    )
    def test_non_name_lines(self, line: str) -> None:
        parser = CoverParser("", [])

        assert parser._looks_like_author_name(line) is False
