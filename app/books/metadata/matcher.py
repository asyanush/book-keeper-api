"""Candidate matching and deduplication module.

Scores and ranks ``BookMetadata`` candidates against a search query
using deterministic rules.  Does *not* depend on any LLM.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.books.metadata.normalizer import MetadataNormalizer
from app.books.metadata.schemas import BookMetadata

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level constants
# ------------------------------------------------------------------

# Confidence ranges (per spec)
CONFIDENCE_EXACT_ISBN13 = 1.0
CONFIDENCE_EXACT_ISBN10 = 0.95
CONFIDENCE_TITLE_AUTHOR_MATCH = 0.9
CONFIDENCE_TITLE_AUTHOR_HIGH = 0.8
CONFIDENCE_TITLE_AUTHOR_FUZZY = 0.7
CONFIDENCE_TITLE_AUTHOR_SIMILAR = 0.6
CONFIDENCE_PARTIAL_MATCH = 0.5
CONFIDENCE_PARTIAL_LOW = 0.4
CONFIDENCE_BARELY = 0.3

# Penalties
ISBN_CONFLICT_PENALTY = 0.5
ISBN_MISSING_PENALTY = 0.2
TITLE_MISMATCH_PENALTY = 0.3
AUTHOR_MISMATCH_PENALTY = 0.3

# Similarity thresholds
TITLE_EXACT_SIMILARITY = 1.0
TITLE_HIGH_SIMILARITY = 0.90
TITLE_MEDIUM_SIMILARITY = 0.70
TITLE_LOW_SIMILARITY = 0.50

AUTHOR_EXACT_SIMILARITY = 1.0
AUTHOR_HIGH_SIMILARITY = 0.85

# Duplicate detection threshold
DUPLICATE_SIMILARITY = 0.95

# Punctuation / noise pattern for matching
_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


# ------------------------------------------------------------------
# CandidateMatcher
# ------------------------------------------------------------------


class CandidateMatcher:
    """Scores, ranks, and deduplicates ``BookMetadata`` candidates.

    Matching is deterministic and follows the priority specified in
    Phase 3 of the roadmap:

    1. Exact valid ISBN-13 match → highest confidence
    2. Exact valid ISBN-10 match → highest confidence
    3. Normalised title + author match → high confidence
    4. Title similarity + author similarity → medium confidence
    5. Partial metadata match → lower confidence
    """

    # ------------------------------------------------------------------
    # Text normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """Normalise text for fuzzy comparison.

        Converts to lowercase, removes punctuation, normalises Unicode
        (NFKC), and collapses whitespace.

        Args:
            text: Arbitrary string.

        Returns:
            Normalised, stripped string ready for comparison.
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = _NORMALIZE_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Title similarity
    # ------------------------------------------------------------------

    @classmethod
    def calculate_title_similarity(cls, title1: str, title2: str) -> float:
        """Calculate normalised string similarity between two titles.

        Uses :class:`difflib.SequenceMatcher` on lowercased,
        punctuation-stripped text.  Additionally accounts for subtitle
        overlap to handle cases where one title includes the subtitle
        and the other does not.

        Args:
            title1: First title (may include subtitle).
            title2: Second title (may include subtitle).

        Returns:
            Similarity score in the range [0.0, 1.0].
        """
        if not title1 or not title2:
            return 0.0

        t1 = cls.normalize_for_matching(title1)
        t2 = cls.normalize_for_matching(title2)

        if t1 == t2:
            return 1.0

        # Compute overall string similarity
        overall = SequenceMatcher(None, t1, t2).ratio()

        # If one string is a subset of the other (e.g. title vs
        # title+subtitle), boost the similarity modestly.
        if t1 in t2 or t2 in t1:
            # The ratio already captures this to some degree; take max
            overall = max(overall, 0.85)

        # Also check word-level overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            # Blend the two scores
            overall = max(overall, jaccard * 0.9 + overall * 0.1)

        return min(overall, 1.0)

    # ------------------------------------------------------------------
    # Author similarity
    # ------------------------------------------------------------------

    @classmethod
    def calculate_author_similarity(
        cls, authors1: List[str], authors2: List[str]
    ) -> float:
        """Calculate similarity between two author lists.

        Compares each author name in *authors1* against every author
        in *authors2* and finds the best pairwise match.  The final
        score is an average of per-author best-match similarities.

        A missing single-author name (e.g. "Last, First" vs "First Last")
        is detected and handled.

        Args:
            authors1: First author list.
            authors2: Second author list.

        Returns:
            Similarity score in [0.0, 1.0].  Returns 1.0 if both
            lists are empty, 0.0 if one is empty and the other is not.
        """
        # Both empty → perfect match (no author information to compare)
        if not authors1 and not authors2:
            return 1.0

        # One empty, the other not → penalty
        if not authors1 or not authors2:
            return 0.0

        scores: List[float] = []

        # For each author in the query, find best match in candidate
        for a1 in authors1:
            n1 = cls.normalize_for_matching(a1)
            best = 0.0
            for a2 in authors2:
                n2 = cls.normalize_for_matching(a2)
                sim = SequenceMatcher(None, n1, n2).ratio()

                # Check for reversed name (e.g. "Martin, Robert" vs "Robert Martin")
                parts1 = [p for p in re.split(r"[,;\s]+", n1) if p]
                parts2 = [p for p in re.split(r"[,;\s]+", n2) if p]
                if parts1 and parts2:
                    s1 = set(parts1)
                    s2 = set(parts2)
                    overlap = len(s1 & s2)
                    # If all parts overlap (or nearly all), it's a reversed-name match
                    if overlap >= min(len(s1), len(s2)):
                        sim = max(sim, 0.95)

                best = max(best, sim)
            scores.append(best)

        # Also check in the reverse direction (each candidate author vs query)
        for a2 in authors2:
            n2 = cls.normalize_for_matching(a2)
            best = 0.0
            for a1 in authors1:
                n1 = cls.normalize_for_matching(a1)
                sim = SequenceMatcher(None, n2, n1).ratio()
                parts1 = [p for p in re.split(r"[,;\s]+", n1) if p]
                parts2 = [p for p in re.split(r"[,;\s]+", n2) if p]
                if parts1 and parts2:
                    s1 = set(parts1)
                    s2 = set(parts2)
                    if len(s1 & s2) >= min(len(s1), len(s2)):
                        sim = max(sim, 0.95)
                best = max(best, sim)
            if best > 0:
                scores.append(best)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # ISBN matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _isbn_match(
        query_isbn: Optional[str],
        candidate_isbn_10: Optional[str],
        candidate_isbn_13: Optional[str],
    ) -> tuple:
        """Determine the kind of ISBN match between query and candidate.

        Returns:
            Tuple of ``(match_level, conflicting)`` where *match_level*
            is one of ``"exact_13"``, ``"exact_10"``, ``"none"`` and
            *conflicting* is ``True`` when both sides have an ISBN but
            they don't match.
        """
        if not query_isbn:
            # No query ISBN to compare
            return ("none", False)

        # First, try exact ISBN match
        if candidate_isbn_13 and query_isbn == candidate_isbn_13:
            return ("exact_13", False)

        if candidate_isbn_10 and query_isbn == candidate_isbn_10:
            return ("exact_10", False)

        # Also try cross-format: query is ISBN-10, candidate has ISBN-13
        # and vice versa.  Convert ISBN-10 to ISBN-13 prefix for
        # cross-format matching.
        normalized_query = query_isbn

        # For ISBN-10 → ISBN-13: prefix is 978 + first 9 digits of ISBN-10
        if len(normalized_query) == 10:
            candidate_13 = candidate_isbn_13
            if candidate_13 and len(candidate_13) == 13:
                expected_13 = "978" + normalized_query[:9]
                # Compute ISBN-13 checksum for the expected
                total = 0
                for i in range(12):
                    weight = 1 if i % 2 == 0 else 3
                    total += weight * int(expected_13[i])
                check = (10 - (total % 10)) % 10
                expected_full = expected_13 + str(check)
                if expected_full == candidate_13:
                    return ("exact_10", False)  # ISBN-10 maps to candidate ISBN-13

        # For ISBN-13 → ISBN-10: strip 978 prefix, compute ISBN-10 checksum
        if len(normalized_query) == 13 and normalized_query.startswith("978"):
            candidate_10 = candidate_isbn_10
            if candidate_10 and len(candidate_10) == 10:
                base_10 = normalized_query[3:12]
                total = 0
                for i in range(9):
                    total += (10 - i) * int(base_10[i])
                check = total % 11
                check_digit = "X" if check == 10 else str(check) if check < 10 else str(11 - check)
                expected_10 = base_10 + str(check_digit)
                if expected_10 == candidate_10:
                    return ("exact_13", False)  # ISBN-13 maps to candidate ISBN-10

        # Both sides have an ISBN but they don't match → conflict
        has_candidate_isbn = bool(candidate_isbn_10 or candidate_isbn_13)
        if has_candidate_isbn:
            return ("none", True)

        return ("none", False)

    # ------------------------------------------------------------------
    # Single-candidate scoring
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        query: Dict[str, Any],
        candidate: BookMetadata,
    ) -> BookMetadata:
        """Score a single candidate against the query.

        Modifies the candidate's ``confidence`` in-place and returns
        the same object for convenience.

        Args:
            query: Search query with optional keys ``isbn``, ``title``, ``author``.
            candidate: A ``BookMetadata`` instance to score.

        Returns:
            The same ``BookMetadata`` instance with ``confidence`` updated.
        """
        query_isbn: Optional[str] = query.get("isbn")
        query_title: Optional[str] = query.get("title")
        query_author: Optional[str] = query.get("author")

        # Build a list of query authors from the single author field
        query_authors: List[str] = []
        if query_author:
            query_authors = [query_author]

        confidence = 0.0

        # --- Phase 1: ISBN matching -----------------------------------------
        isbn_match, isbn_conflict = self._isbn_match(
            query_isbn, candidate.isbn_10, candidate.isbn_13
        )

        if isbn_match == "exact_13":
            confidence = CONFIDENCE_EXACT_ISBN13
        elif isbn_match == "exact_10":
            confidence = CONFIDENCE_EXACT_ISBN10

        # If there's an ISBN conflict, reduce confidence significantly
        if isbn_conflict and confidence > 0:
            confidence = max(0, confidence - ISBN_CONFLICT_PENALTY)
        elif isbn_conflict:
            confidence = max(0, CONFIDENCE_BARELY - ISBN_CONFLICT_PENALTY)

        # If the query has an ISBN but the candidate has none, apply a
        # penalty (but only if we haven't matched by title/author yet).
        if (
            query_isbn
            and not isbn_match.startswith("exact")
            and not candidate.isbn_10
            and not candidate.isbn_13
        ):
            confidence = max(0, confidence - ISBN_MISSING_PENALTY)

        # --- Phase 2: Title + author matching -------------------------------
        # If we already have an exact ISBN match, title/author only refines
        # the score slightly.
        title_sim = 0.0
        author_sim = 0.0

        if query_title and candidate.title:
            title_sim = self.calculate_title_similarity(query_title, candidate.title)

        if query_authors and candidate.authors:
            author_sim = self.calculate_author_similarity(query_authors, candidate.authors)

        # Compute a title+author composite score
        ta_composite = 0.0
        if title_sim > 0 and author_sim > 0:
            ta_composite = title_sim * 0.6 + author_sim * 0.4
        elif title_sim > 0:
            ta_composite = title_sim * 0.7
        elif author_sim > 0:
            ta_composite = author_sim * 0.5

        # If we have an exact ISBN match, the title/author score only
        # nudges the confidence slightly.
        if isbn_match.startswith("exact"):
            if ta_composite >= TITLE_HIGH_SIMILARITY:
                confidence = max(confidence, confidence + 0.02)
            elif ta_composite >= TITLE_MEDIUM_SIMILARITY:
                confidence = max(confidence, confidence)
            elif ta_composite < TITLE_LOW_SIMILARITY and ta_composite > 0:
                # Title/author doesn't match despite ISBN match →
                # could be a reused ISBN; reduce slightly
                confidence = max(confidence - 0.1, 0)
        else:
            # No exact ISBN match – score purely on title/author
            if title_sim >= TITLE_HIGH_SIMILARITY and author_sim >= AUTHOR_HIGH_SIMILARITY:
                confidence = max(confidence, CONFIDENCE_TITLE_AUTHOR_MATCH)
            elif title_sim >= TITLE_HIGH_SIMILARITY and author_sim >= AUTHOR_EXACT_SIMILARITY:
                confidence = max(confidence, CONFIDENCE_TITLE_AUTHOR_MATCH)
            elif title_sim >= TITLE_HIGH_SIMILARITY and author_sim >= 0.6:
                confidence = max(confidence, CONFIDENCE_TITLE_AUTHOR_HIGH)
            elif title_sim >= TITLE_MEDIUM_SIMILARITY and author_sim >= AUTHOR_HIGH_SIMILARITY:
                confidence = max(confidence, CONFIDENCE_TITLE_AUTHOR_FUZZY)
            elif title_sim >= TITLE_MEDIUM_SIMILARITY and author_sim >= 0.5:
                confidence = max(confidence, CONFIDENCE_TITLE_AUTHOR_SIMILAR)
            elif title_sim >= TITLE_MEDIUM_SIMILARITY:
                confidence = max(confidence, CONFIDENCE_PARTIAL_MATCH)
            elif author_sim >= 0.6:
                confidence = max(confidence, CONFIDENCE_PARTIAL_MATCH)
            elif title_sim >= TITLE_LOW_SIMILARITY or author_sim >= 0.4:
                confidence = max(confidence, CONFIDENCE_PARTIAL_LOW)
            else:
                confidence = max(confidence, CONFIDENCE_BARELY)

        # Apply title/author mismatch penalties when query is specific
        if isbn_conflict:
            confidence = max(confidence - ISBN_CONFLICT_PENALTY, 0.0)

        if query_title and candidate.title and title_sim < TITLE_LOW_SIMILARITY:
            confidence = max(confidence - TITLE_MISMATCH_PENALTY, 0.0)

        if query_authors and candidate.authors and author_sim < 0.3:
            confidence = max(confidence - AUTHOR_MISMATCH_PENALTY, 0.0)

        # Clamp to [0, 1]
        candidate.confidence = min(max(confidence, 0.0), 1.0)
        return candidate

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    @classmethod
    def detect_duplicates(cls, candidates: List[BookMetadata]) -> List[BookMetadata]:
        """Identify and merge/flag duplicate candidates.

        Two candidates are considered duplicates of the same edition
        when they have the same source and source_id, or when they
        share an ISBN and have very high title similarity.

        Duplicate entries are removed from the list.  Metadata from
        duplicate candidates is *not* merged – the highest-confidence
        entry is kept and the rest are discarded.

        Args:
            candidates: List of scored ``BookMetadata`` candidates.

        Returns:
            Deduplicated list (may be smaller than input).
        """
        if len(candidates) <= 1:
            return candidates

        kept: List[BookMetadata] = []
        discard_ids: set = set()

        for i, c1 in enumerate(candidates):
            if i in discard_ids:
                continue

            for j, c2 in enumerate(candidates):
                if j <= i or j in discard_ids:
                    continue

                is_dup = False

                # Same source + same source_id → exact same record
                if c1.source == c2.source and c1.source_id and c2.source_id:
                    if c1.source_id == c2.source_id:
                        is_dup = True

                # Same ISBN (either 10 or 13) + very high title similarity
                # → same edition from different providers
                if not is_dup:
                    shared_isbn = False
                    if c1.isbn_13 and c2.isbn_13 and c1.isbn_13 == c2.isbn_13:
                        shared_isbn = True
                    elif c1.isbn_10 and c2.isbn_10 and c1.isbn_10 == c2.isbn_10:
                        shared_isbn = True
                    elif c1.isbn_13 and c2.isbn_10 and c1.isbn_13 == c2.isbn_10:
                        shared_isbn = True
                    elif c1.isbn_10 and c2.isbn_13 and c1.isbn_10 == c2.isbn_13:
                        shared_isbn = True

                    if shared_isbn:
                        t1 = c1.title or ""
                        t2 = c2.title or ""
                        if cls.calculate_title_similarity(t1, t2) >= DUPLICATE_SIMILARITY:
                            is_dup = True

                if is_dup:
                    # Keep the one with higher confidence; discard the other
                    if c1.confidence >= c2.confidence:
                        discard_ids.add(j)
                    else:
                        discard_ids.add(i)
                        break  # c1 is being discarded; skip to next i

        for i, c in enumerate(candidates):
            if i not in discard_ids:
                kept.append(c)

        return kept

    # ------------------------------------------------------------------
    # Main matching entry point
    # ------------------------------------------------------------------

    def match_candidates(
        self, query: Dict[str, Any], candidates: List[BookMetadata]
    ) -> List[BookMetadata]:
        """Score and rank candidates against the search query.

        Steps:

        1. Score each candidate individually.
        2. Sort by descending confidence.
        3. Detect and remove duplicates (keeping higher-confidence entry).
        4. Return the final ranked list.

        Args:
            query: Search query dict, typically with keys ``isbn``,
                   ``title``, and ``author``.
            candidates: List of ``BookMetadata`` instances (usually
                        already normalised by :class:`MetadataNormalizer`).

        Returns:
            Ranked, deduplicated list of ``BookMetadata`` instances
            with updated ``confidence`` values.
        """
        if not candidates:
            return []

        # Step 1: Score each candidate
        scored: List[BookMetadata] = []
        for candidate in candidates:
            scored.append(self._score_candidate(query, candidate))

        # Step 2: Sort by confidence descending, then by title for stability
        scored.sort(
            key=lambda c: (c.confidence, c.title or ""),
            reverse=True,
        )

        # Step 3: Detect and remove duplicates
        result = self.detect_duplicates(scored)

        logger.debug(
            f"Matched {len(candidates)} candidates → "
            f"{len(result)} unique (top confidence: "
            f"{result[0].confidence:.2f}" if result else "no results" + ")"
        )

        return result