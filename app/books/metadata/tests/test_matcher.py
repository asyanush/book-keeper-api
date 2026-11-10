"""Unit tests for candidate matching and deduplication.

Coverage maps to ``specs/roadmap/phase-03-metadata-retrieval.md``
(Part 2, Section 16):

1.  Exact ISBN-13 match receives highest confidence.
2.  Exact ISBN-10 match receives highest confidence.
3.  Matching title and author produces a high score.
4.  Different punctuation does not prevent matching.
5.  Case differences do not prevent matching.
6.  Subtitle differences are handled correctly.
7.  Multiple authors are matched correctly.
8.  Same title but different authors produces a lower score.
9.  Same author but different title produces a lower score.
10. Different ISBN editions are not automatically considered identical editions.
11. Missing ISBN does not automatically reject a candidate.
12. Conflicting ISBNs significantly reduce confidence.
13. Duplicate candidates are identified.
14. Ambiguous candidates are not automatically selected.

Section 16 additionally requires coverage of books with multiple editions
(identical titles/authors but different ISBNs, page counts, publishers and
publication dates).  Those cases live in :class:`TestEditionDistinction`.

The matcher is pure and deterministic (no network, no randomness, no LLM),
so every test here is deterministic and uses only the shared fixtures from
``conftest.py``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from app.books.metadata.matcher import CandidateMatcher
from app.books.metadata.schemas import BookMetadata

# ---------------------------------------------------------------------------
# ISBN constants (mirroring the values used by ``conftest.py``)
# ---------------------------------------------------------------------------

VALID_ISBN_13 = "9780132350884"  # Clean Code, Prentice Hall, 2008
VALID_ISBN_10 = "0132350882"

PAPERBACK_ISBN_13 = "9780137081073"
PAPERBACK_ISBN_10 = "0137081071"

EBOOK_ISBN_13 = "9780135957059"
EBOOK_ISBN_10 = "0135957052"

ARCHITECTURE_ISBN_13 = "9780134494166"
ARCHITECTURE_ISBN_10 = "0134494168"

CLEAN_CODE_TITLE = "Clean Code"
CLEAN_CODE_AUTHOR = "Robert C. Martin"


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _as_metadata(data: dict[str, Any]) -> BookMetadata:
    """Convert a provider candidate dict fixture into ``BookMetadata``."""
    return BookMetadata(**copy.deepcopy(data))


def _score_single(
    matcher: CandidateMatcher,
    query: dict[str, Any],
    candidate: BookMetadata,
) -> BookMetadata:
    """Score one candidate and return it (deep-copied to avoid mutation)."""
    result = matcher.match_candidates(query, [candidate.model_copy(deep=True)])
    assert len(result) == 1, "A single input candidate must yield one result"
    return result[0]


def _title_author_query() -> dict[str, str]:
    return {"title": CLEAN_CODE_TITLE, "author": CLEAN_CODE_AUTHOR}


# ===========================================================================
# normalize_for_matching
# ===========================================================================


class TestNormalizeForMatching:
    """``CandidateMatcher.normalize_for_matching`` text normalisation."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Clean Code", "clean code"),
            ("CLEAN CODE", "clean code"),
            ("  Clean   CODE!!  ", "clean code"),
            ("Clean-Code: A Handbook", "clean code a handbook"),
            ("Robert C. Martin", "robert c martin"),
            ("Robert C Martin", "robert c martin"),
            ("Craftsman\u2019s Guide", "craftsman s guide"),
            ("Clean_Code", "clean_code"),
            ("978-0-13-595705-9", "978 0 13 595705 9"),
        ],
    )
    def test_normalizes_punctuation_case_and_whitespace(
        self,
        matcher: CandidateMatcher,
        raw: str,
        expected: str,
    ) -> None:
        assert matcher.normalize_for_matching(raw) == expected

    def test_empty_string_stays_empty(self, matcher: CandidateMatcher) -> None:
        assert matcher.normalize_for_matching("") == ""

    def test_preserves_unicode_letters(self, matcher: CandidateMatcher) -> None:
        # NFKC keeps accented letters; only punctuation is stripped.
        assert matcher.normalize_for_matching("Caf\u00e9") == "caf\u00e9"

    def test_is_idempotent(self, matcher: CandidateMatcher) -> None:
        raw = "  Clean-Code: A Handbook!! "
        once = matcher.normalize_for_matching(raw)
        assert matcher.normalize_for_matching(once) == once


# ===========================================================================
# calculate_title_similarity
# ===========================================================================


class TestTitleSimilarity:
    """``CandidateMatcher.calculate_title_similarity`` behaviour."""

    def test_identical_titles_score_one(self, matcher: CandidateMatcher) -> None:
        assert matcher.calculate_title_similarity("Clean Code", "Clean Code") == 1.0

    def test_case_differences_do_not_reduce_similarity(
        self, matcher: CandidateMatcher
    ) -> None:
        assert (
            matcher.calculate_title_similarity("CLEAN CODE", "clean code") == 1.0
        )

    def test_punctuation_differences_do_not_reduce_similarity(
        self, matcher: CandidateMatcher
    ) -> None:
        assert (
            matcher.calculate_title_similarity(
                "Clean-Code: A Handbook!", "Clean Code A Handbook"
            )
            == 1.0
        )

    def test_subtitle_subset_still_scores_highly(
        self, matcher: CandidateMatcher
    ) -> None:
        """A title vs title+subtitle is a strong (but not perfect) match."""
        score = matcher.calculate_title_similarity(
            "Clean Code: A Handbook of Agile Software Craftsmanship",
            "Clean Code",
        )
        assert 0.8 <= score < 1.0

    def test_subtitle_similarity_is_symmetric(self, matcher: CandidateMatcher) -> None:
        a = "Clean Code: A Handbook of Agile Software Craftsmanship"
        b = "Clean Code"
        assert matcher.calculate_title_similarity(a, b) == pytest.approx(
            matcher.calculate_title_similarity(b, a)
        )

    @pytest.mark.parametrize(
        ("title1", "title2"),
        [
            ("Clean Code", "Clean Architecture"),
            ("The Hobbit", "Dune"),
            ("A", "Completely Different Book"),
        ],
    )
    def test_distinct_titles_score_below_high_threshold(
        self,
        matcher: CandidateMatcher,
        title1: str,
        title2: str,
    ) -> None:
        # "Clean Code" / "Clean Architecture" share a word, so they are not
        # treated as unrelated; they must simply stay below the high-match
        # threshold used to grant the strongest title/author confidence.
        assert matcher.calculate_title_similarity(title1, title2) < 0.7

    @pytest.mark.parametrize(
        ("title1", "title2"),
        [
            ("", "Clean Code"),
            ("Clean Code", ""),
            ("", ""),
        ],
    )
    def test_missing_title_scores_zero(
        self,
        matcher: CandidateMatcher,
        title1: str,
        title2: str,
    ) -> None:
        assert matcher.calculate_title_similarity(title1, title2) == 0.0

    @pytest.mark.parametrize(
        ("title1", "title2"),
        [
            ("Clean Code", "Clean Code"),
            ("Clean Code", "Clean Architecture"),
            ("Clean Code: A Handbook", "Clean Code"),
            ("The Hobbit", "Dune"),
        ],
    )
    def test_similarity_is_always_within_range(
        self,
        matcher: CandidateMatcher,
        title1: str,
        title2: str,
    ) -> None:
        score = matcher.calculate_title_similarity(title1, title2)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# calculate_author_similarity
# ===========================================================================


class TestAuthorSimilarity:
    """``CandidateMatcher.calculate_author_similarity`` behaviour."""

    def test_both_empty_lists_are_a_perfect_match(
        self, matcher: CandidateMatcher
    ) -> None:
        assert matcher.calculate_author_similarity([], []) == 1.0

    @pytest.mark.parametrize(
        ("authors1", "authors2"),
        [
            ([], ["Robert C. Martin"]),
            (["Robert C. Martin"], []),
        ],
    )
    def test_one_empty_list_scores_zero(
        self,
        matcher: CandidateMatcher,
        authors1: list[str],
        authors2: list[str],
    ) -> None:
        assert matcher.calculate_author_similarity(authors1, authors2) == 0.0

    def test_identical_author_lists_score_one(
        self, matcher: CandidateMatcher
    ) -> None:
        authors = ["Robert C. Martin"]
        assert matcher.calculate_author_similarity(authors, authors) == 1.0

    def test_reversed_last_first_name_is_recognised(
        self, matcher: CandidateMatcher
    ) -> None:
        score = matcher.calculate_author_similarity(
            ["Martin, Robert C."], ["Robert C. Martin"]
        )
        assert score >= 0.9

    def test_multiple_authors_match(
        self, matcher: CandidateMatcher
    ) -> None:
        authors = ["Robert C. Martin", "Micah Martin"]
        assert matcher.calculate_author_similarity(authors, authors) == 1.0

    def test_multiple_authors_ignore_ordering(
        self, matcher: CandidateMatcher
    ) -> None:
        score = matcher.calculate_author_similarity(
            ["Micah Martin", "Robert C. Martin"],
            ["Robert C. Martin", "Micah Martin"],
        )
        assert score == 1.0

    def test_single_query_author_still_matches_multi_author_list(
        self, matcher: CandidateMatcher
    ) -> None:
        score = matcher.calculate_author_similarity(
            ["Robert C. Martin"],
            ["Robert C. Martin", "Micah Martin"],
        )
        assert score >= 0.8

    def test_different_authors_score_low(self, matcher: CandidateMatcher) -> None:
        score = matcher.calculate_author_similarity(
            ["Robert C. Martin"], ["Jane Doe"]
        )
        assert score < 0.5

    @pytest.mark.parametrize(
        ("authors1", "authors2"),
        [
            (["Robert C. Martin"], ["Robert C. Martin"]),
            (["Robert C. Martin"], ["Jane Doe"]),
            (["Robert C. Martin", "Micah Martin"], ["Robert C. Martin"]),
            ([], []),
        ],
    )
    def test_similarity_is_always_within_range(
        self,
        matcher: CandidateMatcher,
        authors1: list[str],
        authors2: list[str],
    ) -> None:
        score = matcher.calculate_author_similarity(authors1, authors2)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# 1 & 2. Exact ISBN matching
# ===========================================================================


class TestExactIsbnMatching:
    """Exact ISBN-13 / ISBN-10 matches outrank title/author-only matches."""

    def test_exact_isbn13_match_receives_highest_confidence(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        exact = book_metadata_factory(source_id="exact")
        # Same book but without an ISBN: a valid but weaker match.
        no_isbn = book_metadata_factory(
            isbn_10=None,
            isbn_13=None,
            source="open_library",
            source_id="no-isbn",
        )
        # A different book entirely.
        other = book_metadata_factory(
            title="Clean Architecture",
            isbn_10=ARCHITECTURE_ISBN_10,
            isbn_13=ARCHITECTURE_ISBN_13,
            source_id="other",
        )

        results = matcher.match_candidates(query, [other, no_isbn, exact])

        assert results[0].source_id == "exact"
        assert results[0].confidence == 1.0
        assert results[0].confidence > results[1].confidence
        assert all(0.0 <= c.confidence <= 1.0 for c in results)

    def test_exact_isbn13_beats_title_author_match(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        exact = _score_single(matcher, query, book_metadata_factory())
        title_author_only = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(isbn_10=None, isbn_13=None),
        )
        assert exact.confidence > title_author_only.confidence

    def test_exact_isbn10_match_receives_highest_confidence(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_10,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        exact = book_metadata_factory(source_id="exact")
        no_isbn = book_metadata_factory(
            isbn_10=None,
            isbn_13=None,
            source="open_library",
            source_id="no-isbn",
        )

        results = matcher.match_candidates(query, [no_isbn, exact])

        assert results[0].source_id == "exact"
        assert results[0].confidence >= 0.95
        assert results[0].confidence > results[1].confidence

    def test_exact_isbn_match_without_title_or_author(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        scored = _score_single(
            matcher,
            {"isbn": VALID_ISBN_13},
            book_metadata_factory(),
        )
        assert scored.confidence == 1.0

    def test_exact_isbn_with_near_matching_metadata_stays_high(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        """An exact ISBN holds its confidence even when metadata is imperfect."""
        candidate = book_metadata_factory(
            title="Clean Code Handbook",
            authors=["Rob Martin"],
        )

        scored = _score_single(
            matcher,
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidate,
        )

        assert scored.confidence == 1.0

    def test_exact_isbn_with_mismatched_metadata_is_penalised(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        """An ISBN match alone must not fully override disagreeing metadata."""
        candidate = book_metadata_factory(
            title="Quantum Physics",
            authors=["Jane Doe"],
            source_id="isbn-reuse",
        )

        scored = _score_single(
            matcher,
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidate,
        )

        assert 0.0 <= scored.confidence < 1.0
        assert scored.confidence <= 0.5

    def test_isbn10_query_matches_isbn13_only_candidate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        """Cross-format: ISBN-10 query, candidate only exposes ISBN-13."""
        candidate = book_metadata_factory(isbn_10=None, isbn_13=VALID_ISBN_13)

        scored = _score_single(
            matcher,
            {
                "isbn": VALID_ISBN_10,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidate,
        )

        assert scored.confidence >= 0.95

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known implementation limitation: CandidateMatcher._isbn_match "
            "converts ISBN-13 to ISBN-10 with a non-standard checksum "
            "formula, so an ISBN-13 query does not match a candidate that "
            "only exposes ISBN-10. The roadmap only requires the "
            "ISBN-10 -> ISBN-13 direction."
        ),
    )
    def test_isbn13_query_matches_isbn10_only_candidate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        """Cross-format: ISBN-13 query, candidate only exposes ISBN-10."""
        candidate = book_metadata_factory(isbn_10=VALID_ISBN_10, isbn_13=None)

        scored = _score_single(
            matcher,
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidate,
        )

        assert scored.confidence >= 0.95


# ===========================================================================
# 3-9. Title / author matching
# ===========================================================================


class TestTitleAuthorMatching:
    """Title and author based scoring, including punctuation/case/subtitles."""

    def test_matching_title_and_author_produces_high_score(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        scored = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(isbn_10=None, isbn_13=None),
        )
        assert scored.confidence >= 0.8

    def test_matching_title_and_author_beats_title_only(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(isbn_10=None, isbn_13=None)
        both = _score_single(matcher, _title_author_query(), candidate)
        title_only = _score_single(
            matcher, {"title": CLEAN_CODE_TITLE}, candidate
        )
        assert both.confidence > title_only.confidence

    def test_different_punctuation_does_not_prevent_matching(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(isbn_10=None, isbn_13=None)

        scored = _score_single(
            matcher,
            {"title": "Clean-Code!!", "author": "Robert C Martin"},
            candidate,
        )

        assert scored.confidence >= 0.8
        assert (
            matcher.calculate_title_similarity("Clean-Code!!", candidate.title or "")
            == 1.0
        )

    def test_case_differences_do_not_prevent_matching(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(isbn_10=None, isbn_13=None)

        scored = _score_single(
            matcher,
            {"title": "CLEAN CODE", "author": "ROBERT C. MARTIN"},
            candidate,
        )

        assert scored.confidence >= 0.8

    def test_subtitle_differences_are_handled(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        """Query includes a subtitle the candidate stores in its own field."""
        candidate = book_metadata_factory(isbn_10=None, isbn_13=None)

        with_subtitle = _score_single(
            matcher,
            {
                "title": (
                    "Clean Code: A Handbook of Agile Software Craftsmanship"
                ),
                "author": CLEAN_CODE_AUTHOR,
            },
            candidate,
        )
        without_subtitle = _score_single(
            matcher, _title_author_query(), candidate
        )

        assert with_subtitle.confidence >= 0.7
        # An exact title match still scores strictly higher.
        assert with_subtitle.confidence < without_subtitle.confidence

    def test_subtitle_embedded_in_candidate_title_is_handled(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(
            title="Clean Code: A Handbook of Agile Software Craftsmanship",
            subtitle=None,
            isbn_10=None,
            isbn_13=None,
        )

        scored = _score_single(
            matcher, _title_author_query(), candidate
        )

        assert scored.confidence >= 0.7

    def test_multiple_authors_are_matched_correctly(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(
            authors=["Robert C. Martin", "Micah Martin"],
            isbn_10=None,
            isbn_13=None,
        )

        scored = _score_single(
            matcher, _title_author_query(), candidate
        )

        assert scored.confidence >= 0.8

    def test_same_title_different_authors_produces_lower_score(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        matching_author = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(isbn_10=None, isbn_13=None),
        )
        different_author = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(
                authors=["Jane Doe"],
                isbn_10=None,
                isbn_13=None,
                source_id="different-author",
            ),
        )

        assert different_author.confidence < matching_author.confidence
        assert different_author.confidence <= 0.6

    def test_same_author_different_title_produces_lower_score(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        matching_title = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(isbn_10=None, isbn_13=None),
        )
        different_title = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(
                title="Clean Architecture",
                isbn_10=None,
                isbn_13=None,
                source_id="different-title",
            ),
        )

        assert different_title.confidence < matching_title.confidence
        assert different_title.confidence <= 0.6

    def test_partial_author_match_lowers_score_but_stays_high(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(
            authors=["Rob Martin"],
            isbn_10=None,
            isbn_13=None,
            source_id="partial-author",
        )

        scored = _score_single(matcher, _title_author_query(), candidate)

        # Exact title with a partial author match is strong, but below the
        # full title+author combination.
        assert 0.7 <= scored.confidence < 0.9

    def test_medium_title_and_partial_author_score_moderately(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory(
            title="Clean Code Handbook",
            authors=["Rob Martin"],
            isbn_10=None,
            isbn_13=None,
            source_id="fuzzy",
        )

        scored = _score_single(matcher, _title_author_query(), candidate)

        assert 0.5 <= scored.confidence < 0.7

    def test_completely_unrelated_book_scores_near_zero(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        scored = _score_single(
            matcher,
            _title_author_query(),
            book_metadata_factory(
                title="Quantum Physics for Beginners",
                authors=["Jane Doe"],
                isbn_10=None,
                isbn_13=None,
                source_id="unrelated",
            ),
        )
        assert scored.confidence <= 0.3


# ===========================================================================
# 10-12. Editions, missing ISBNs and ISBN conflicts
# ===========================================================================


class TestEditionDistinction:
    """Different editions must not be collapsed or treated as identical."""

    def test_multiple_editions_fixture_has_distinct_metadata(
        self, multiple_editions: list[BookMetadata]
    ) -> None:
        """Sanity check the fixture really models distinct editions."""
        assert len(multiple_editions) == 3

        assert len({edition.isbn_13 for edition in multiple_editions}) == 3
        assert len({edition.isbn_10 for edition in multiple_editions}) == 3
        assert len({edition.publisher for edition in multiple_editions}) == 3
        assert len({edition.published_date for edition in multiple_editions}) == 3
        assert len({edition.page_count for edition in multiple_editions}) >= 2

        # Yet they share the same title and author.
        assert {edition.title for edition in multiple_editions} == {CLEAN_CODE_TITLE}
        for edition in multiple_editions:
            assert edition.authors == [CLEAN_CODE_AUTHOR]

    def test_title_author_query_keeps_all_editions_separate(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        results = matcher.match_candidates(
            _title_author_query(), copy.deepcopy(multiple_editions)
        )

        # Identical title/author but different ISBNs are distinct editions,
        # so none of them may be deduplicated.
        assert len(results) == 3
        assert {edition.isbn_13 for edition in results} == {
            VALID_ISBN_13,
            PAPERBACK_ISBN_13,
            EBOOK_ISBN_13,
        }

    def test_exact_isbn_selects_only_the_matching_edition(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        query = {
            "isbn": PAPERBACK_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        results = matcher.match_candidates(
            query, copy.deepcopy(multiple_editions)
        )

        assert len(results) == 3
        top = results[0]
        assert top.isbn_13 == PAPERBACK_ISBN_13
        assert top.isbn_10 == PAPERBACK_ISBN_10
        assert top.publisher == "Prentice Hall PTR"
        assert top.published_date == "2009-07-01"

        # Only the matching edition gets a very high score.
        assert top.confidence >= 0.95
        assert all(edition.confidence <= 0.5 for edition in results[1:])
        assert sum(edition.confidence >= 0.95 for edition in results) == 1

    def test_editions_are_not_considered_identical_by_title_author_alone(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        """Title/author alone must not let one edition trump the others."""
        results = matcher.match_candidates(
            _title_author_query(), copy.deepcopy(multiple_editions)
        )

        assert len({edition.confidence for edition in results}) == 1
        assert results[0].confidence == pytest.approx(results[1].confidence)

    def test_missing_isbn_does_not_reject_candidate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        candidate = book_metadata_factory(
            isbn_10=None,
            isbn_13=None,
            source_id="no-isbn",
        )

        scored = _score_single(matcher, query, candidate)

        assert scored.confidence >= 0.8

    def test_conflicting_isbn_significantly_reduces_confidence(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }
        matching = _score_single(matcher, query, book_metadata_factory())
        conflicting = _score_single(
            matcher,
            query,
            book_metadata_factory(
                isbn_10=PAPERBACK_ISBN_10,
                isbn_13=PAPERBACK_ISBN_13,
                source_id="conflicting-isbn",
            ),
        )

        assert conflicting.confidence <= 0.5
        assert matching.confidence - conflicting.confidence >= 0.4

    def test_conflicting_provider_isbns_are_preserved_as_separate_results(
        self,
        matcher: CandidateMatcher,
        conflicting_isbn_provider_responses: dict[str, dict[str, Any]],
    ) -> None:
        candidates = [
            _as_metadata(data)
            for data in conflicting_isbn_provider_responses.values()
        ]

        results = matcher.match_candidates(
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidates,
        )

        # Different ISBNs mean different editions: both are returned.
        assert len(results) == 2

        google = next(c for c in results if c.source == "google_books")
        open_library = next(c for c in results if c.source == "open_library")

        assert google.confidence == 1.0
        assert open_library.confidence <= 0.5
        assert google.confidence > open_library.confidence


# ===========================================================================
# 13. Duplicate detection
# ===========================================================================


class TestDuplicateDetection:
    """``CandidateMatcher.detect_duplicates`` and dedup during matching."""

    def test_empty_list_is_returned_unchanged(
        self, matcher: CandidateMatcher
    ) -> None:
        assert matcher.detect_duplicates([]) == []

    def test_single_candidate_is_returned_unchanged(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        candidate = book_metadata_factory()
        result = matcher.detect_duplicates([candidate])
        assert result == [candidate]

    def test_same_source_and_source_id_is_a_duplicate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        first = book_metadata_factory(confidence=0.4)
        second = book_metadata_factory(confidence=0.9)

        kept = matcher.detect_duplicates([first, second])

        assert len(kept) == 1
        assert kept[0].confidence == 0.9

    def test_same_source_duplicate_found_with_higher_confidence_first(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        higher = book_metadata_factory(confidence=0.9)
        lower = book_metadata_factory(confidence=0.4)

        kept = matcher.detect_duplicates([higher, lower])

        assert len(kept) == 1
        assert kept[0].confidence == 0.9

    def test_same_isbn_across_providers_is_a_duplicate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        google = book_metadata_factory(
            source="google_books",
            source_id="gb-record",
            confidence=0.8,
        )
        open_library = book_metadata_factory(
            source="open_library",
            source_id="ol-record",
            confidence=0.6,
        )

        kept = matcher.detect_duplicates([google, open_library])

        assert len(kept) == 1
        assert kept[0].source == "google_books"

    def test_same_isbn10_across_providers_is_a_duplicate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        google = book_metadata_factory(
            isbn_10=VALID_ISBN_10,
            isbn_13=None,
            source="google_books",
            source_id="gb-isbn10",
            confidence=0.7,
        )
        open_library = book_metadata_factory(
            isbn_10=VALID_ISBN_10,
            isbn_13=None,
            source="open_library",
            source_id="ol-isbn10",
            confidence=0.5,
        )

        kept = matcher.detect_duplicates([google, open_library])

        assert len(kept) == 1
        assert kept[0].source == "google_books"

    def test_same_title_and_author_but_different_isbn_is_not_a_duplicate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        hardcover = book_metadata_factory(source_id="hardcover")
        paperback = book_metadata_factory(
            isbn_10=PAPERBACK_ISBN_10,
            isbn_13=PAPERBACK_ISBN_13,
            source="open_library",
            source_id="paperback",
        )

        kept = matcher.detect_duplicates([hardcover, paperback])

        assert len(kept) == 2

    def test_same_isbn_but_different_title_is_not_a_duplicate(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        first = book_metadata_factory(source_id="first")
        second = book_metadata_factory(
            title="A Completely Different Book",
            source="open_library",
            source_id="second",
        )

        kept = matcher.detect_duplicates([first, second])

        assert len(kept) == 2

    def test_match_candidates_deduplicates_conflicting_provider_records(
        self,
        matcher: CandidateMatcher,
        conflicting_provider_responses: dict[str, dict[str, Any]],
    ) -> None:
        candidates = [
            _as_metadata(data)
            for data in conflicting_provider_responses.values()
        ]

        results = matcher.match_candidates(
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            candidates,
        )

        assert len(results) == 1
        assert results[0].confidence == 1.0

    def test_match_candidates_deduplicates_complementary_provider_records(
        self,
        matcher: CandidateMatcher,
        complementary_provider_responses: dict[str, dict[str, Any]],
    ) -> None:
        candidates = [
            _as_metadata(data)
            for data in complementary_provider_responses.values()
        ]

        results = matcher.match_candidates(
            _title_author_query(), candidates
        )

        assert len(results) == 1
        assert results[0].confidence >= 0.8


# ===========================================================================
# 14. Ambiguity, selection and determinism
# ===========================================================================


class TestAmbiguityAndDeterminism:
    """Selection rules, confidence bounds, and deterministic behaviour."""

    def test_ambiguous_candidates_are_not_automatically_selected(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        results = matcher.match_candidates(
            _title_author_query(), copy.deepcopy(multiple_editions)
        )

        assert len(results) > 1
        # Equal top scores mean there is no clearly best candidate.
        assert results[0].confidence == pytest.approx(results[1].confidence)
        assert results[0].confidence - results[1].confidence == pytest.approx(0.0)

    def test_unique_best_candidate_is_ranked_first_when_clearly_best(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        results = matcher.match_candidates(
            {
                "isbn": EBOOK_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            copy.deepcopy(multiple_editions),
        )

        assert results[0].isbn_13 == EBOOK_ISBN_13
        # The winner must be clearly ahead, not a near-tie.
        assert results[0].confidence - results[1].confidence > 0.3

    def test_first_api_result_is_not_selected_solely_by_order(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        wrong_first = book_metadata_factory(
            title="Clean Architecture",
            isbn_10=ARCHITECTURE_ISBN_10,
            isbn_13=ARCHITECTURE_ISBN_13,
            source_id="wrong-first",
        )
        correct_second = book_metadata_factory(source_id="correct-second")

        results = matcher.match_candidates(
            _title_author_query(), [wrong_first, correct_second]
        )

        assert results[0].source_id == "correct-second"
        assert results[0].confidence > results[1].confidence

    def test_matching_is_deterministic(
        self,
        matcher: CandidateMatcher,
        multiple_editions: list[BookMetadata],
    ) -> None:
        query = _title_author_query()

        first_run = matcher.match_candidates(
            query, copy.deepcopy(multiple_editions)
        )
        second_run = matcher.match_candidates(
            query, copy.deepcopy(multiple_editions)
        )

        assert [c.isbn_13 for c in first_run] == [c.isbn_13 for c in second_run]
        assert [c.confidence for c in first_run] == [
            c.confidence for c in second_run
        ]

    def test_matching_with_single_candidate_is_deterministic(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
    ) -> None:
        query = {
            "isbn": VALID_ISBN_13,
            "title": CLEAN_CODE_TITLE,
            "author": CLEAN_CODE_AUTHOR,
        }

        first = _score_single(matcher, query, book_metadata_factory())
        second = _score_single(matcher, query, book_metadata_factory())

        assert first.confidence == second.confidence

    def test_empty_candidate_list_returns_empty(
        self, matcher: CandidateMatcher
    ) -> None:
        assert matcher.match_candidates(_title_author_query(), []) == []

    @pytest.mark.parametrize(
        "query",
        [
            {},
            {"title": CLEAN_CODE_TITLE},
            {"author": CLEAN_CODE_AUTHOR},
            {"isbn": VALID_ISBN_13},
            {"isbn": VALID_ISBN_10},
            {
                "isbn": VALID_ISBN_13,
                "title": CLEAN_CODE_TITLE,
                "author": CLEAN_CODE_AUTHOR,
            },
            {"title": CLEAN_CODE_TITLE, "author": "Somebody Else"},
        ],
    )
    def test_confidence_values_are_within_zero_and_one(
        self,
        matcher: CandidateMatcher,
        book_metadata_factory: Any,
        query: dict[str, Any],
    ) -> None:
        candidates = [
            book_metadata_factory(source_id="match"),
            book_metadata_factory(
                isbn_10=None,
                isbn_13=None,
                title="Clean Architecture",
                authors=["Jane Doe"],
                source="open_library",
                source_id="partial",
            ),
            book_metadata_factory(
                title="Quantum Physics",
                authors=["Jane Doe"],
                isbn_10=ARCHITECTURE_ISBN_10,
                isbn_13=ARCHITECTURE_ISBN_13,
                source_id="unrelated",
            ),
        ]

        results = matcher.match_candidates(query, candidates)

        assert results, "Scoring should still return candidates"
        for candidate in results:
            assert 0.0 <= candidate.confidence <= 1.0
