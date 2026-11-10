"""Deterministic cover-text parser that extracts structured metadata from raw OCR output.

No LLM calls — everything is rule-based.  Handles typical cover layouts (trade
paperback / hardcover front covers) and partial spine-only text.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from pydantic import BaseModel, Field

from app.books.metadata.normalizer import MetadataNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level compiled regexes
# ---------------------------------------------------------------------------

# Bestseller / award badge lines (e.g. "#1 NEW YORK TIMES BESTSELLER").
_RE_BESTSELLER = re.compile(
    r"(?:#1?\s*)?(?:NEW\s*YORK\s*TIMES|SUNDAY\s*TIMES|WALL\s*STREET|"
    r"INTERNATIONAL|USA\s*TODAY|#\d+|NO\.?\s*\d+)"
    r".*?(?:BESTSELLER|BEST\s*SELLING|BEST-SELLER|AWARD|PRIZE|WINNER|FINALIST)",
    re.IGNORECASE,
)

# Series markers: "Book 1", "Vol. 3", "Book One".
_RE_SERIES = re.compile(
    r"^(?:Book\s+\d+|[Vv]ol\.?\s*\d+|#[Vv]olume\s+\d+|"
    r"Book\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve))$",
)

# Review quotes: lines that start or end with quotation marks.
_RE_QUOTE = re.compile(r'^["""\']|[“”\'"]$')

# Attribution dashes (em-dash, en-dash, hyphen) followed by a letter.
_RE_ATTRIBUTION = re.compile(r"^[—–-]\s*\w")

# Author-credential line: "AUTHOR OF ...", "EDITOR OF ...", etc.
_RE_AUTHOR_CREDENTIAL = re.compile(
    r"^(?:AUTHOR|EDITOR|TRANSLATOR|ILLUSTRATOR|PHOTOGRAPHER|WRITER|COMPILER)\s+OF\s",
    re.IGNORECASE,
)

# "by <Name>" pattern.
_RE_BY_LINE = re.compile(r"^[Bb]y\s+(.+?)$")

# Pure OCR artifact lines: dominated by non-word, non-whitespace chars.
_RE_PURE_ARTIFACT = re.compile(r"^[^\w\s]{3,}$")

# Multiple author splitters.
_RE_AUTHOR_SPLIT = re.compile(r"\s*(?:,\s*|\band\b|&)\s*")

# Collapse multiple whitespace runs.
_RE_WHITESPACE = re.compile(r"\s+")

# A "name word" starts with a letter ad may contain letters, hyphens,
# apostrophes, or periods (e.g "J.R.R.", "O'Brien", "Anne-Marie").
_RE_NAME_WORD = re.compile(r"^[A-Za-z][A-Za-z'.\-]*$")

# Common English funcion words that usualy do not appear in author names.
_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "from", "by", "as", "is", "it", "be", "no", "not",
    "but", "if", "so", "we", "he", "she", "they", "this", "that",
    "all", "are", "was", "were", "been", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "its", "a", "an",
})


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class ParsedCoverData(BaseModel):
    """Structured data extracted from a book cover OCR output."""

    title: str | None = None
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    isbns: list[str] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def parse_cover_text(
    raw_text: str, detected_isbns: list[str] | None = None
) -> ParsedCoverData:
    """Parse raw OCR cover/spine text into structured metadata.

    Parameters
    ----------
    raw_text:
        Raw OCR text from cover/spine image.
    detected_isbns:
        ISBNs already detected by the OCR pipeline (normalised, no hyphens/spaces).

    Returns
    -------
    ParsedCoverData
        Extracted metadata with confidence score.
    """
    parser = CoverParser(raw_text, detected_isbns or [])
    return parser.parse()


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


class CoverParser:
    """Deterministic rule-based cover-text parser.

    Use :func:`parse_cover_text` as the primary public API; this class is
    exposed for cases where incremental / partial parsing is needed.
    """

    def __init__(self, raw_text: str, detected_isbns: list[str]) -> None:
        # Normalise Unicode early (NFKC combining chars -> composed forms).
        self.raw_text = unicodedata.normalize("NFKC", raw_text)
        self.detected_isbns = list(detected_isbns) if detected_isbns else []
        self._normalised_isbns: list[str] = []
        self._lines: list[str] = []
        self._clean_lines: list[str] = []  # after noise removal
        self._title_idx: int | None = None
        self._results: ParsedCoverData = ParsedCoverData(raw_text=self.raw_text)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> ParsedCoverData:
        """Run the full parsing pipeline and return structured metadata."""
        self._segment_lines()
        self._filter_noise()
        self._handle_isbns()
        self._extract_title()
        self._extract_subtitle()
        self._extract_authors()
        self._compute_confidence()
        return self._results

    # ------------------------------------------------------------------
    # Step 1: Line segmentation & basic cleaning
    # ------------------------------------------------------------------

    def _segment_lines(self) -> None:
        """Split text into lines, strip whitespace, drop empty lines."""
        self._lines = [
            _RE_WHITESPACE.sub(" ", line).strip()
            for line in self.raw_text.split("\n")
        ]
        # Remove empty lines.
        self._lines = [ln for ln in self._lines if ln]

    # ------------------------------------------------------------------
    # Step 2: Noise filtering
    # ------------------------------------------------------------------

    def _filter_noise(self) -> None:
        """Check each line and keep only meaningful ones."""
        for line in self._lines:
            if self._is_noise_line(line):
                logger.debug("Noise line discarded: %r", line)
                continue
            self._clean_lines.append(line)

    def _is_noise_line(self, line: str) -> bool:
        """Return True when *line* should be discarded as non-metadata."""

        # --- Single-character artifacts (unless a capital letter - spine text) ---
        if len(line) < 2:
            return not (len(line) == 1 and line.isupper() and line.isalpha())

        if len(line) < 3 and not line.isupper():
            return True

        # --- Pure symbol garbage that OCR produces from binarisation ---
        if _RE_PURE_ARTIFACT.search(line):
            return True

        # --- Lines dominated by non-alphanumeric chars (OCR artifacts) ---
        alpha_count = sum(1 for ch in line if ch.isalpha())
        if len(line) > 3 and alpha_count < len(line) * 0.3:
            return True

        # --- Bestseller / award badge ---
        if _RE_BESTSELLER.search(line):
            return True

        # --- Review quotes ---
        stripped = line.strip("\"" + "'" + "\u201c\u201d\u2018\u2019")
        if _RE_QUOTE.search(stripped):
            return True
        if _RE_ATTRIBUTION.match(line):
            return True

        return False

    # ------------------------------------------------------------------
    # Step 3: ISBN handling
    # ------------------------------------------------------------------

    def _handle_isbns(self) -> None:
        """Validate and store normalised ISBNs."""
        valid: list[str] = []
        for isbn in self.detected_isbns:
            norm = MetadataNormalizer.normalize_isbn(isbn)
            if norm:
                valid.append(norm)
            else:
                logger.debug("ISBN failed checksum validation: %r", isbn)
        # Dedupe preserving order.
        seen: set = set()
        deduped: list[str] = []
        for v in valid:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        self._normalised_isbns = deduped
        self._results.isbns = self._normalised_isbns

    # ------------------------------------------------------------------
    # Step 4: Title extraction
    # ------------------------------------------------------------------

    def _extract_title(self) -> None:
        """Extract the book title from cleaned lines using scoring heuristics."""
        if not self._clean_lines:
            return

        # --- Score each candidate line for "title-ness" ---
        scored: list[tuple[int, float, str]] = []

        for i, line in enumerate(self._clean_lines):
            # Skip lines that are clearly not titles.
            if _RE_AUTHOR_CREDENTIAL.match(line):
                continue

            words = line.split()
            wc = len(words)
            score = 0.0

            # Position bonus: lines near the top score much higher.
            n_lines = max(len(self._clean_lines), 1)
            # First line gets full bonus, second 0.7, rest drop off quickly.
            if i == 0:
                position_bonus = 1.0
            elif i == 1:
                position_bonus = 0.7
            else:
                position_bonus = max(0.0, 1.0 - (i / n_lines) * 0.9)
            score += position_bonus * 0.3

            # ALL CAPS = probable title (common on covers), but moderate weight
            # since author names are also often in all caps.
            if line.isupper():
                score += 0.25
            elif line[0].isupper():
                score += 0.15

            # Prefer 1-4 word lines as titles (most common).
            if 1 <= wc <= 4:
                score += 0.3
            elif 5 <= wc <= 7:
                score += 0.1

            # Length bonus: longer is more informative for short-len candidates.
            line_len = len(line)
            if 3 <= line_len <= 60:
                score += min(line_len / 60.0, 1.0) * 0.1

            # Penalise lines that look like descriptions (mixed case, many words).
            lowercase_words = sum(1 for w in words if w.islower())
            if wc > 3 and lowercase_words > wc * 0.5:
                score -= 0.3

            # Penalise very short single-word lines (single letters, etc.).
            if wc == 1 and line_len < 4:
                score -= 0.3

            scored.append((i, score, line))

        if not scored:
            return

        # Pick the line with the highest score.
        scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        best_idx, best_score, best_line = scored[0]

        # If another candidate scored very close (< 0.05 difference) and appears
        # earlier, prefer the earlier one (title typically comes before author).
        for idx, score, line in scored[1:]:
            if best_score - score < 0.05 and idx < best_idx:
                # The earlier candidate must look plausible as a title.
                if line.isupper() or len(line.split()) <= 5:
                    best_idx, best_score, best_line = idx, score, line
                    break

        # If the best candidate looks like a subtitle (colon, long, descriptive),
        # and there is an earlier ALL CAPS line, prefer the all-caps one.
        looks_like_subtitle = (
            best_score < 0.5
            or (";" in best_line and len(best_line.split()) > 4)
            or best_line.lower().startswith(("why ", "how "))
        )
        if looks_like_subtitle:
            for idx, _, line in scored:
                if (
                    idx < best_idx
                    and ";" not in line
                    and not line.lower().startswith(("why ", "how "))
                ):
                    if line.isupper() or (len(line.split()) <= 4):
                        best_idx, best_line = idx, line
                        break

        self._results.title = best_line
        self._title_idx = best_idx

    # ------------------------------------------------------------------
    # Step 5: Subtitle extraction
    # ------------------------------------------------------------------

    def _extract_subtitle(self) -> None:
        """Extract the subtitle - typically the line(s) immediately after the title."""
        if not self._results.title or self._title_idx is None or not self._clean_lines:
            return

        # Gather up to 3 candidate lines right after title.
        subtitle_parts: list[str] = []
        for i in range(self._title_idx + 1, min(self._title_idx + 4, len(self._clean_lines))):
            line = self._clean_lines[i]

            # Stop if we hit an author-credential line.
            if _RE_AUTHOR_CREDENTIAL.match(line):
                break

            # Stop if we hit a review quote.
            if _RE_QUOTE.search(line) or _RE_ATTRIBUTION.match(line):
                break

            # If short ALL CAPS line (different from title), include it but stop.
            if line.isupper() and len(line.split()) <= 2 and line != self._results.title:
                subtitle_parts.append(line)
                break

            # If this is the last or second-to-last clean line and looks like
            # an author name, skip it (author, not subtitle).
            if i >= len(self._clean_lines) - 2 and self._looks_like_author_name(line):
                break

            # Otherwise include it.
            subtitle_parts.append(line)

            # Stop if the line ends with sentence-ending punctuation.
            if line.rstrip().endswith((".", "!", "?")):
                break

        if subtitle_parts:
            subtitle = " ".join(subtitle_parts)
            subtitle = _RE_BESTSELLER.sub("", subtitle).strip()
            if subtitle and subtitle != self._results.title:
                self._results.subtitle = subtitle

    # ------------------------------------------------------------------
    # Step 6: Author extraction
    # ------------------------------------------------------------------

    def _extract_authors(self) -> None:
        """Extract author names from lines, scanning bottom-up."""
        if not self._clean_lines:
            return

        authors: list[str] = []

        for line in reversed(self._clean_lines):
            # "by <Name>" pattern.
            by_match = _RE_BY_LINE.match(line)
            if by_match:
                name = by_match.group(1).strip()
                if name:
                    authors = self._split_authors(name) + authors
                    continue

            # "AUTHOR OF ..." — skip, it is a credential not a name.
            if _RE_AUTHOR_CREDENTIAL.match(line):
                continue

            # Skip review quotes.
            if _RE_QUOTE.search(line) or _RE_ATTRIBUTION.match(line):
                continue

            # Stop scanning upward if we hit what looks like the title,
            # subtitle, or a descriptive sentence (not an author).
            if line in (self._results.title, self._results.subtitle):
                break

            # Stop if the line looks like part of the subtitle/description
            # (long, sentence-like, contains many function words).
            words_in_line = line.split()
            if (
                len(words_in_line) >= 4
                and sum(1 for w in words_in_line if w.lower() in _FUNCTION_WORDS) >= 2
            ):
                break

            # Check if it looks like an author name.
            if self._looks_like_author_name(line):
                authors = self._split_authors(line) + authors
            else:
                # If we already found authors and hit a non-author line, stop.
                if authors:
                    break

        # Deduplicate preserving order.
        seen: set = set()
        deduped: list[str] = []
        for a in authors:
            stripped = a.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                deduped.append(stripped)
        self._results.authors = deduped

    def _split_authors(self, text: str) -> list[str]:
        """Split a string that may contain multiple authors."""
        parts = _RE_AUTHOR_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _looks_like_author_name(self, line: str) -> bool:
        """Heuristic: does *line* look like a person's name?"""
        raw_words = line.split()
        # Ignore author separators (commas, ampersands) for word counting.
        words = [w for w in raw_words if w not in (",", "&")]
        if not (1 <= len(words) <= 4):
            return False

        # Reject single-character "names" (spine fragments like "T", "H").
        if all(len(w) == 1 and w.isalpha() for w in words):
            return False

        # Reject artifact lines.
        if _RE_PURE_ARTIFACT.match(line):
            return False

        # Reject clearly non-name patterns.
        non_name_patterns = (
            re.compile(r"^\d+$"),
            re.compile(r"^[IVX]+$"),
            re.compile(r"^(?:A|An|The)\s", re.IGNORECASE),
            _RE_BESTSELLER,
            _RE_SERIES,
            _RE_AUTHOR_CREDENTIAL,
        )
        for pat in non_name_patterns:
            if pat.search(line):
                return False

        # Filter out lines of common function words (any case).
        function_count = sum(1 for w in words if w.lower() in _FUNCTION_WORDS)
        if function_count >= len(words) * 0.5:
            return False

        # Reject all-lowercase short names (OCR fragments like "eat", not real names).
        all_lower = all(w.islower() for w in words)
        if all_lower and len(words) == 1 and len(words[0]) <= 4:
            return False

        # Reject lines where most words are very short lowercase fragments
        # (likely OCR trailing garbage).
        short_lower = sum(1 for w in words if w.islower() and len(w) <= 3)
        if len(words) >= 1 and short_lower / len(words) > 0.5:
            return False

        # Each word should be reasonably name-like.
        for w in words:
            if not _RE_NAME_WORD.match(w) and not w.isupper():
                return False

        return True

    # ------------------------------------------------------------------
    # Step 7: Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self) -> None:
        """Compute overall parsing confidence on 0.0—1.0 scale.

        - 0.8+ : title + author + ISBN all found.
        - 0.5-0.7 : title + author found.
        - <0.5 : only partial data.
        """
        has_title = self._results.title is not None
        has_author = bool(self._results.authors)
        has_isbn = bool(self._results.isbns)
        has_subtitle = self._results.subtitle is not None

        if has_title and has_author and has_isbn:
            base = 0.85
        elif has_title and has_author:
            base = 0.65
        elif has_title:
            base = 0.45
        else:
            base = 0.20

        if has_subtitle:
            base += 0.05

        if self._lines:
            clean_ratio = len(self._clean_lines) / max(len(self._lines), 1)
            if clean_ratio < 0.3:
                base -= 0.10

        self._results.confidence = round(max(0.0, min(1.0, base)), 2)