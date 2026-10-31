"""OCR text-extraction engine (Tesseract / Google Vision / Hybrid).

All heavy dependencies (``pytesseract``, ``google.cloud.vision``) are imported
**lazily** inside their respective functions so that this module stays
importable when those optional packages are not installed in the environment.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.books.ocr.image_processing import preprocess_image
from app.core.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISBN regex patterns
# ---------------------------------------------------------------------------
# Match ISBN-13 (978/979 + 10 digits) with optional hyphens/spaces between groups.
_ISBN13_RE = re.compile(r"\b(?:97[89])[-\s]?\d(?:[-\s]?\d){9}\b")
# Match ISBN-10 (9 digits + check digit which may be X) with optional separators.
_ISBN10_RE = re.compile(r"\b(?:\d[-\s]?){9}[\dXx]\b")

# Word-count threshold below which the fallback engine will be invoked.
_WORD_THRESHOLD = 2

# Symbols that virtually never appear in legitimate title/author text but are
# common in binarization/cover-art noise that Tesseract misreads as characters.
_ARTIFACT_SYMBOLS = set(r"\/|=<>[]{}~^_§†¶¤")

# Minimum fraction of tokens that must be artifact-symbol tokens for the
# whole text to be considered noise.
_ARTIFACT_RATIO = 0.12

# Minimum fraction of alphabetic tokens that must look like plausible words
# (contain a vowel, no 3+ consecutive identical characters) for the text to be
# treated as real.
_PLAUSIBLE_RATIO = 0.5

_VOWELS = set("aeiouAEIOU")


# ---------------------------------------------------------------------------
# ISBN parsing
# ---------------------------------------------------------------------------
def _parse_isbns(text: str) -> list[str]:
    """Extract, normalize and deduplicate ISBN-10 / ISBN-13 candidates from *text*.

    Hyphens and spaces are stripped; ordering is preserved.
    ISBN-10 matches that overlap with an ISBN-13 match are discarded to
    avoid capturing the trailing 10 digits of a hyphenated ISBN-13.
    """
    seen: set[str] = set()
    result: list[str] = []

    # Collect spans of ISBN-13 matches so we can reject overlapping ISBN-10s.
    isbn13_spans: list[tuple[int, int]] = []
    isbn13_values: list[str] = []

    for match in _ISBN13_RE.finditer(text):
        raw = match.group()
        normalized = re.sub(r"[-\s]", "", raw)
        if normalized not in seen and len(normalized) == 13 and normalized.isdigit():
            seen.add(normalized)
            isbn13_spans.append((match.start(), match.end()))
            isbn13_values.append(normalized)

    result.extend(isbn13_values)

    for match in _ISBN10_RE.finditer(text):
        # Skip if this match overlaps any ISBN-13 match.
        s, e = match.start(), match.end()
        if any(not (e <= start or s >= end) for start, end in isbn13_spans):
            continue

        raw = match.group()
        normalized = re.sub(r"[-\s]", "", raw).upper()
        if normalized not in seen and len(normalized) == 10 and _is_valid_isbn10_chars(normalized):
            seen.add(normalized)
            result.append(normalized)

    return result


def _is_valid_isbn10_chars(s: str) -> bool:
    """Check that *s* consists only of digits and an optional trailing X/x."""
    if not s:
        return False
    # All but the last character must be digits; last may be digit or X.
    return s[:-1].isdigit() and (s[-1].isdigit() or s[-1] == "X")


def _dedupe(items: list[str]) -> list[str]:
    """Return *items* without duplicates, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# OCR-quality heuristics
# ---------------------------------------------------------------------------
def _has_vowel(token: str) -> bool:
    """Return True if *token* contains at least one vowel."""
    return any(ch in _VOWELS for ch in token)


def _has_triple(token: str) -> bool:
    """Return True if *token* contains 3+ consecutive identical characters."""
    for i in range(len(token) - 2):
        if token[i] == token[i + 1] == token[i + 2]:
            return True
    return False


def _looks_like_noise(raw_text: str) -> bool:
    """Heuristically decide whether OCR output is noise rather than real text.

    Tesseract run on a thresholded/stylized cover frequently produces a large
    number of whitespace-separated tokens that are *not* real words, so the
    word-count fallback threshold alone is not enough to detect failure.  This
    function flags two tell-tale signatures:

    1. Artifact tokens — whole tokens made of backslashes, pipes, equals signs,
       angle brackets, etc. that cover art/binarization turns into "characters".
    2. Implausible words — alphabetic tokens lacking any vowel (``NNNN``) or
       containing 3+ identical characters in a row (``eee``).

    Parameters
    ----------
    raw_text:
        OCR output text.

    Returns
    -------
    bool
        True when the text should be treated as noise and the fallback engine
        should be consulted.
    """
    if not raw_text:
        return True

    tokens = [t for t in raw_text.split() if t]
    if not tokens:
        return True

    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not alpha_tokens:
        return True

    # 1. Structural artifacts across the whole token stream.
    artifact_tokens = [t for t in tokens if any(ch in _ARTIFACT_SYMBOLS for ch in t)]
    if artifact_tokens and len(artifact_tokens) / len(tokens) >= _ARTIFACT_RATIO:
        return True

    # 2. Implausible "words" among the alphabetic tokens.
    plausible = [
        t for t in alpha_tokens if _has_vowel(t) and not _has_triple(t)
    ]
    if len(plausible) / len(alpha_tokens) < _PLAUSIBLE_RATIO:
        return True

    return False


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------
def extract_text_tesseract(image_bytes: bytes) -> str:
    """Run Tesseract OCR on the (preprocessed) image bytes.

    Parameters
    ----------
    image_bytes:
        Preprocessed PNG image bytes.

    Returns
    -------
    str
        Extracted text, stripped of leading/trailing whitespace.
    """
    import io

    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    text: str = pytesseract.image_to_string(image)
    return text.strip()


def extract_text_vision(image_bytes: bytes) -> tuple[str, list[str]]:
    """Run Google Cloud Vision text detection + barcode parsing.

    Parameters
    ----------
    image_bytes:
        Preprocessed PNG image bytes.

    Returns
    -------
    tuple[str, list[str]]
        ``(concatenated_text, barcode_values)``.

    Notes
    -----
    Returns ``("", [])`` instead of raising when the Google credentials are
    unavailable or if the Vision client cannot be constructed; this function
    is designed to serve as a safe fallback engine.
    """
    # Gracefully handle missing GCP credentials / packages.
    if not settings.GOOGLE_APPLICATION_CREDENTIALS:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set — Vision fallback disabled.")
        return "", []

    try:
        from google.cloud import vision
    except ImportError:
        logger.warning("google-cloud-vision is not installed — Vision fallback disabled.")
        return "", []

    try:
        client = vision.ImageAnnotatorClient.from_service_account_file(
            settings.GOOGLE_APPLICATION_CREDENTIALS
        )
    except Exception:
        logger.exception("Failed to construct Vision client — Vision fallback disabled.")
        return "", []

    image = vision.Image(content=image_bytes)

    # --- Text detection + barcode extraction --------------------------------
    texts: list[str] = []
    barcodes: list[str] = []
    try:
        response = client.text_detection(image=image)
        if response.text_annotations:
            # First annotation contains the full block of detected text.
            texts.append(response.text_annotations[0].description)
            # Subsequent annotations are individual words — harvest barcode-like tokens.
            for annotation in response.text_annotations[1:]:
                token = annotation.description.strip()
                # Barcodes are typically all-digit, 8-13 characters long.
                if token.isdigit() and 8 <= len(token) <= 13:
                    barcodes.append(token)
    except Exception:
        logger.exception("Vision text detection failed.")

    full_text = "\n".join(texts).strip()
    return full_text, barcodes


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def extract_book_text(image_bytes: bytes) -> dict:
    """Run the OCR pipeline and return structured results.

    The pipeline:
    1. Preprocess the raw image bytes (grayscale + contrast normalization).
    2. Run the **primary** OCR engine (determined by ``settings.OCR_STRATEGY``).
    3. If the primary output is empty, contains fewer than ``_WORD_THRESHOLD``
       (2) words, or looks like OCR noise, invoke the **fallback** engine.
    4. Parse ISBN-10 / ISBN-13 patterns from the final text.
    5. Return a dict with ``raw_text``, ``detected_isbns``, and
       ``engine_used``.

    Strategy matrix
    ---------------
    ============  =============  ==============
    OCR_STRATEGY  Primary        Fallback
    ============  =============  ==============
    TESSERACT     Tesseract      Vision
    VISION        Vision         Tesseract
    HYBRID        Tesseract      Vision
    ============  =============  ==============

    Returns
    -------
    dict
        ``{"raw_text": str, "detected_isbns": list[str], "engine_used": str}``
    """
    processed = preprocess_image(image_bytes)

    strategy = settings.OCR_STRATEGY.upper()

    # Determine engine order based on strategy.
    primary_is_vision = strategy == "VISION"
    primary_name = "Vision" if primary_is_vision else "Tesseract"
    fallback_name = "Tesseract" if primary_is_vision else "Vision"

    # Barcodes detected by the Vision engine (if used) are merged into the
    # final ISBN list below.
    barcodes: list[str] = []

    # Execute primary engine.
    if primary_is_vision:
        raw_text, barcodes = extract_text_vision(processed)
    else:
        raw_text = extract_text_tesseract(processed)

    used_fallback = False

    word_count = len(raw_text.split()) if raw_text else 0
    noisy = _looks_like_noise(raw_text)

    if word_count < _WORD_THRESHOLD or noisy:
        reason = (
            f"{word_count} words (< {_WORD_THRESHOLD})"
            if word_count < _WORD_THRESHOLD
            else "output looks like OCR noise"
        )
        logger.info(
            "Primary engine '%s' returned %s — invoking fallback '%s'.",
            primary_name,
            reason,
            fallback_name,
        )
        if primary_is_vision:
            fallback_text = extract_text_tesseract(processed)
        else:
            fallback_text, vision_barcodes = extract_text_vision(processed)
            barcodes.extend(vision_barcodes)

        # Accept the fallback result when the primary output was empty or pure
        # noise; otherwise only when it contains more words than the primary.
        # (Noise detection is what makes the first two cases reachable — a
        # noisy primary can easily have *more* tokens than the real fallback.)
        fallback_word_count = len(fallback_text.split()) if fallback_text else 0
        if fallback_text and (
            word_count < _WORD_THRESHOLD
            or noisy
            or fallback_word_count > word_count
        ):
            raw_text = fallback_text
            used_fallback = True

    # Parse ISBNs from the final text and from any barcodes the Vision engine
    # surfaced; barcode values are filtered through the same ISBN patterns.
    detected_isbns = _dedupe(
        _parse_isbns(raw_text) + _parse_isbns(" ".join(barcodes))
    )

    engine_desc = (
        f"{primary_name} (Fallback: {fallback_name})"
        if used_fallback
        else f"{primary_name} (Fallback: None)"
    )

    return {
        "raw_text": raw_text,
        "detected_isbns": detected_isbns,
        "engine_used": engine_desc,
    }