"""Tests for the OCR preprocessing pipeline and service layer.

All heavy dependencies (``cv2``, ``numpy``, ``pytesseract``, ``google.cloud.vision``)
are imported lazily inside their respective functions.  This test file mocks
those packages at the ``sys.modules`` level or via ``monkeypatch.setattr``
on the consuming module so that the tests never require the real packages.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from app.books.exceptions import ImageProcessingError
from app.books.ocr.image_processing import preprocess_image
from app.books.ocr.ocr_service import (
    _looks_like_noise,
    _parse_isbns,
    extract_book_text,
)
from app.books.tests.fixtures import SAMPLE_PNG_BYTES
from app.core.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cv2() -> MagicMock:
    """Return a fake ``cv2`` module with pipeline attributes already configured.

    The returned mock has all the constants and functions needed by
    :func:`~app.books.ocr.image_processing.preprocess_image`.  Callers can
    further customise individual attributes (e.g. ``imdecode.return_value``).
    """
    fake_cv2 = MagicMock()
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.COLOR_BGR2GRAY = 6

    # Pipeline steps return chained mocks by default.
    fake_cv2.imdecode.return_value = MagicMock()       # img
    fake_cv2.cvtColor.return_value = MagicMock()        # gray
    clahe = MagicMock()
    clahe.apply.return_value = MagicMock()              # enhanced
    fake_cv2.createCLAHE.return_value = clahe
    buf = MagicMock()
    buf.tobytes.return_value = b"processed_png_bytes"
    fake_cv2.imencode.return_value = (True, buf)
    return fake_cv2


def _make_fake_numpy() -> MagicMock:
    """Return a fake ``numpy`` module for use by ``preprocess_image``."""
    fake_np = MagicMock()
    fake_np.uint8 = int  # A callable that accepts a single argument.
    fake_np.frombuffer.return_value = b"fake_arr"
    return fake_np


# ===================================================================
# 1.  Image-preprocessing pipeline
# ===================================================================

class TestPreprocessingPipeline:
    """Tests for :func:`~app.books.ocr.image_processing.preprocess_image`."""

    def _install_fakes(
        self, monkeypatch: pytest.MonkeyPatch,
        fake_cv2: MagicMock | None = None,
        fake_np: MagicMock | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Inject fake ``cv2`` / ``numpy`` into ``sys.modules``.

        Because :func:`preprocess_image` *lazily imports* these packages
        inside the function body, we must insert our fakes **before** the
        first call so that the ``import cv2`` / ``import numpy as np``
        statements resolve to them.
        """
        if fake_cv2 is None:
            fake_cv2 = _make_fake_cv2()
        if fake_np is None:
            fake_np = _make_fake_numpy()

        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        monkeypatch.setitem(sys.modules, "numpy", fake_np)
        return fake_cv2, fake_np

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_successful_preprocessing_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All pipeline steps are invoked and the final encoded bytes are returned."""
        fake_cv2, fake_np = self._install_fakes(monkeypatch)

        result = preprocess_image(SAMPLE_PNG_BYTES)

        # Final return value comes from buf.tobytes().
        assert result == b"processed_png_bytes"

        # --- Verify each pipeline step was called with plausible arguments ---

        # 1. np.frombuffer(image_bytes, np.uint8)
        fake_np.frombuffer.assert_called_once()
        call_args, _ = fake_np.frombuffer.call_args
        assert call_args[0] == SAMPLE_PNG_BYTES

        # 2. cv2.imdecode(arr, cv2.IMREAD_COLOR)
        fake_cv2.imdecode.assert_called_once_with(b"fake_arr", 1)

        # 3. cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fake_cv2.cvtColor.assert_called_once()
        assert fake_cv2.cvtColor.call_args[0][1] == 6  # COLOR_BGR2GRAY

        # 4. cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        fake_cv2.createCLAHE.assert_called_once()
        assert fake_cv2.createCLAHE.call_args[1] == {
            "clipLimit": 2.0,
            "tileGridSize": (8, 8),
        }

        # 5. clahe.apply(gray)
        clahe = fake_cv2.createCLAHE.return_value
        clahe.apply.assert_called_once()
        assert clahe.apply.call_args[0][0] is fake_cv2.cvtColor.return_value

        # 6. cv2.imencode(".png", enhanced)
        fake_cv2.imencode.assert_called_once()
        assert fake_cv2.imencode.call_args[0][0] == ".png"
        assert fake_cv2.imencode.call_args[0][1] is clahe.apply.return_value

        # 7. buf.tobytes()
        buf = fake_cv2.imencode.return_value[1]
        buf.tobytes.assert_called_once()

    # ------------------------------------------------------------------
    # Corrupted-image path (imdecode returns None)
    # ------------------------------------------------------------------

    def test_corrupted_image_raises_image_processing_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``ImageProcessingError`` (a ``DomainException`` with status 400)
        is raised when ``cv2.imdecode`` returns ``None``."""
        fake_cv2, _ = self._install_fakes(monkeypatch)
        fake_cv2.imdecode.return_value = None  # Simulate decode failure

        with pytest.raises(ImageProcessingError) as exc_info:
            preprocess_image(b"corrupted_bytes")

        assert exc_info.type.status_code == 400
        assert "Could not decode" in str(exc_info.value)

    # ------------------------------------------------------------------
    # Encoding-failure path (imencode returns ok=False)
    # ------------------------------------------------------------------

    def test_encoding_failure_raises_image_processing_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``ImageProcessingError`` when ``cv2.imencode`` returns ``(False, _)``."""
        fake_cv2, _ = self._install_fakes(monkeypatch)
        fake_cv2.imencode.return_value = (False, MagicMock())

        with pytest.raises(ImageProcessingError) as exc_info:
            preprocess_image(SAMPLE_PNG_BYTES)

        assert exc_info.type.status_code == 400
        assert "Could not encode" in str(exc_info.value)


# ===================================================================
# 1b. OCR noise detection
# ===================================================================

class TestLooksLikeNoise:
    """Tests for :func:`~app.books.ocr.ocr_service._looks_like_noise`."""

    @pytest.mark.parametrize(
        "raw_text",
        [
            # The exact symptom reported on a stylized cover photo.
            (
                "owen eee\n\n; Art EAN AANA SHARAN NUNES\n"
                "Se ; ARARARIRA \\ SENN BANS =<\nSe eae ee A c =\n\nFon\n\n\\"
            ),
            # Standalone artifact symbols only.
            "\\ = < > | ==",
            # Alphabetic tokens without vowels / with runs of the same char.
            "NNNN WWW eee ttt",
        ],
    )
    def test_noise_is_detected(self, raw_text: str) -> None:
        assert _looks_like_noise(raw_text) is True

    @pytest.mark.parametrize(
        "raw_text",
        [
            "The Name of the Wind Patrick Rothfuss",
            "Atomic Habits James Clear",
            "Sufficient text here",
            # Real text may contain a normal punctuation symbol.
            "Clean Code: A Handbook of Agile Software Craftsmanship",
        ],
    )
    def test_real_text_is_not_noise(self, raw_text: str) -> None:
        assert _looks_like_noise(raw_text) is False

    def test_empty_text_is_noise(self) -> None:
        assert _looks_like_noise("") is True
        assert _looks_like_noise("   \n  ") is True


# ===================================================================
# 2.  Tesseract → Vision fallback
# ===================================================================

class TestTesseractFallbackToVision:
    """When Tesseract returns empty, the orchestrator should invoke Vision."""

    def test_tesseract_empty_falls_back_to_vision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_vision_client: MagicMock,
    ) -> None:
        """Tesseract returns ``""`` → the real Vision engine runs and its client is called."""
        # ── Ensure TESSERACT is the primary engine ──────────────────
        monkeypatch.setattr(settings, "OCR_STRATEGY", "TESSERACT")

        # ── Configure the mocked Vision client to return known text ─
        vision_client = (
            mock_vision_client.ImageAnnotatorClient.from_service_account_file.return_value
        )
        vision_client.text_detection.return_value.text_annotations[0].description = (
            "Vision identified text"
        )

        # ── Mock preprocess_image so no real cv2/numpy is needed ──
        mock_preprocess = MagicMock(return_value=b"processed")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.preprocess_image", mock_preprocess
        )

        # ── Mock Tesseract (returns empty) ─────────────────────────
        mock_tesseract = MagicMock(return_value="")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_tesseract", mock_tesseract
        )

        result = extract_book_text(b"raw_input_bytes")

        # Fallback text should be used.
        assert result["raw_text"] == "Vision identified text"
        assert result["engine_used"] == "Tesseract (Fallback: Vision)"
        assert result["detected_isbns"] == []  # No ISBNs in the text

        # Tesseract got the preprocessed bytes, and the Vision client was hit.
        mock_tesseract.assert_called_once_with(b"processed")
        vision_client.text_detection.assert_called_once()

    def test_tesseract_gibberish_falls_back_to_vision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_vision_client: MagicMock,
    ) -> None:
        """Multi-word garbage (e.g. threshold noise on a stylized cover) must
        still trigger the Vision fallback instead of being returned as-is."""
        monkeypatch.setattr(settings, "OCR_STRATEGY", "TESSERACT")

        vision_client = (
            mock_vision_client.ImageAnnotatorClient.from_service_account_file.return_value
        )
        vision_client.text_detection.return_value.text_annotations[0].description = (
            "The Name of the Wind"
        )

        mock_preprocess = MagicMock(return_value=b"processed")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.preprocess_image", mock_preprocess
        )

        garbage = "owen eee ; Art EAN AANA SHARAN NUNES \\ SENN BANS =<"
        mock_tesseract = MagicMock(return_value=garbage)
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_tesseract", mock_tesseract
        )

        result = extract_book_text(b"raw_input_bytes")

        assert result["raw_text"] == "The Name of the Wind"
        assert result["engine_used"] == "Tesseract (Fallback: Vision)"
        vision_client.text_detection.assert_called_once()

    def test_tesseract_sufficient_words_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tesseract returns ≥2 words → Vision is NOT called."""
        monkeypatch.setattr(settings, "OCR_STRATEGY", "TESSERACT")

        mock_preprocess = MagicMock(return_value=b"processed")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.preprocess_image", mock_preprocess
        )

        mock_tesseract = MagicMock(return_value="Sufficient text here")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_tesseract", mock_tesseract
        )

        mock_vision = MagicMock()
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_vision", mock_vision
        )

        result = extract_book_text(b"raw_input_bytes")

        assert result["raw_text"] == "Sufficient text here"
        assert result["engine_used"] == "Tesseract (Fallback: None)"

        # Tesseract was called; Vision was never called.
        mock_tesseract.assert_called_once()
        mock_vision.assert_not_called()


# ===================================================================
# 3.  ISBN regex parsing
# ===================================================================

class TestIsbnParsing:
    """ISBN extraction, normalisation and deduplication.

    Tests exercise the ``_parse_isbns`` helper directly **and** the
    ``extract_book_text`` orchestrator (with fully mocked engines).
    """

    # ------------------------------------------------------------------
    # Direct unit tests for _parse_isbns
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("raw_text", "expected"),
        [
            # ISBN-13 with hyphens → normalised (all digits).
            ("978-0-13-595705-9", ["9780135957059"]),
            # ISBN-13 without separators → unchanged.
            ("9780135957059", ["9780135957059"]),
            # ISBN-10 embedded in text.
            ("ISBN 0-321-12521-5", ["0321125215"]),
            # ISBN-13 with spaces.
            ("ISBN 978 0 13 595705 9", ["9780135957059"]),
            # Multiple ISBNs → deduplicated.
            (
                "First: 978-0-13-595705-9  Second: 9780135957059",
                ["9780135957059"],
            ),
            # ISBN-13 and ISBN-10 that do NOT overlap → both returned.
            (
                "ISBN-13: 978-0-13-595705-9  ISBN-10: 0-321-12521-5",
                ["9780135957059", "0321125215"],
            ),
            # ISBN-10 with trailing X (lowercase x).
            ("0-306-40615-X", ["030640615X"]),
            ("0-306-40615-x", ["030640615X"]),
            # Nothing ISBN-like → empty list.
            ("This is just plain text without any identifiers.", []),
        ],
    )
    def test_parse_isbns_various_formats(
        self, raw_text: str, expected: list[str],
    ) -> None:
        assert _parse_isbns(raw_text) == expected

    def test_isbn10_overlapping_isbn13_is_discarded(self) -> None:
        """The trailing 10 digits of a hyphenated ISBN-13 should NOT also
        be returned as a separate ISBN-10 match."""
        # "0-13-595705-9" overlaps with the full ISBN-13 match "978-0-13-595705-9".
        result = _parse_isbns("978-0-13-595705-9")
        assert result == ["9780135957059"]

    # ------------------------------------------------------------------
    # Integration test through extract_book_text
    # ------------------------------------------------------------------

    def test_extract_book_text_returns_parsed_isbns(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``extract_book_text`` returns the parsed ISBNs from the OCR output."""
        monkeypatch.setattr(settings, "OCR_STRATEGY", "TESSERACT")

        # Mock preprocessing.
        mock_preprocess = MagicMock(return_value=b"processed")
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.preprocess_image", mock_preprocess
        )

        # Tesseract returns text containing ISBNs.
        mock_tesseract = MagicMock(
            return_value="Book title 978-0-13-595705-9 and 0-321-12521-5"
        )
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_tesseract", mock_tesseract
        )

        # Vision should NOT be called (sufficient words), but mock it anyway.
        monkeypatch.setattr(
            "app.books.ocr.ocr_service.extract_text_vision", MagicMock()
        )

        result = extract_book_text(b"raw_input_bytes")

        assert result["raw_text"] == "Book title 978-0-13-595705-9 and 0-321-12521-5"
        assert result["detected_isbns"] == ["9780135957059", "0321125215"]
        assert result["engine_used"] == "Tesseract (Fallback: None)"