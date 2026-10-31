"""Reusable test data and factories for the books test suite.

All heavy dependencies (``cv2``, ``numpy``, ``pytesseract``, ``google.cloud.vision``,
``PIL``) are avoided at import time so this module can be imported even when
those optional packages are not installed.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Sample image bytes  (no Pillow / OpenCV needed)
# ---------------------------------------------------------------------------
# A minimal 1×1 red pixel PNG, base64-encoded.
_1X1_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)

SAMPLE_PNG_BYTES: bytes = base64.b64decode(_1X1_PNG_B64)
"""Pre-encoded 1×1 red PNG suitable as a stand-in for any image upload or
OCR test that needs realistic image bytes without pulling in Pillow."""


def sample_image_bytes() -> bytes:
    """Return a small in-memory PNG byte string (1×1 red pixel).

    Convenience helper for tests that prefer a callable over a module-level
    constant.
    """
    return SAMPLE_PNG_BYTES


# ---------------------------------------------------------------------------
# Google Cloud Vision mock factory
# ---------------------------------------------------------------------------
# The Vision API is imported lazily inside ``extract_text_vision`` (see
# ``app/books/ocr/ocr_service.py``).  This factory creates a fake
# ``google.cloud.vision`` module that can be installed into ``sys.modules``
# so that the lazy import resolves without hitting a real GCP endpoint.
#
# Usage in a test::
#
#     from app.books.tests.fixtures import make_vision_mock
#
#     def test_foo(monkeypatch):
#         import sys
#         fake_vision = make_vision_mock(return_text="some text")
#         monkeypatch.setitem(sys.modules, "google.cloud.vision", fake_vision)
#         ...


def make_vision_mock(
    *,
    return_text: str = "",
    return_barcodes: Sequence[str] | None = None,
) -> MagicMock:
    """Build a fake ``google.cloud.vision`` module for offline testing.

    The returned ``MagicMock`` mimics the parts of the ``google.cloud.vision``
    API that ``extract_text_vision`` interacts with:

    * ``vision.Image`` — stores ``content`` as a keyword argument
    * ``vision.ImageAnnotatorClient`` — its
      ``from_service_account_file(path)`` classmethod returns the client, whose
      ``text_detection(image=…)`` method returns a response with a
      ``text_annotations`` list mirroring what a real Vision API call
      returns.

    Parameters
    ----------
    return_text:
        The ``description`` of the first (full-text) annotation.  Defaults to
        ``""``.
    return_barcodes:
        A sequence of barcode-like strings.  Each string becomes a subsequent
        annotation (``response.text_annotations[1:]``) whose ``description``
        is set to the string.  Defaults to an empty list.

    Returns
    -------
    MagicMock
        A fake ``google.cloud.vision`` module.  The client instance can be
        retrieved as
        ``fake_vision.ImageAnnotatorClient.from_service_account_file.return_value``
        for assertions.
    """
    if return_barcodes is None:
        return_barcodes = []

    # --- Build annotation list as the Vision API would expose it ---------
    annotations: list[MagicMock] = []

    # First annotation = full block of detected text.
    first = MagicMock(spec=[])  # no __getattr__ surprises
    first.description = return_text
    annotations.append(first)

    # Subsequent annotations = individual word / token detections.
    for barcode in return_barcodes:
        ann = MagicMock(spec=[])
        ann.description = barcode
        annotations.append(ann)

    # --- Response object -------------------------------------------------
    response = MagicMock(spec=["text_annotations"])
    response.text_annotations = annotations

    # --- Client ----------------------------------------------------------
    client = MagicMock(spec=["text_detection"])
    client.text_detection.return_value = response

    # --- Fake module -----------------------------------------------------
    # ``extract_text_vision`` constructs the client via
    # ``ImageAnnotatorClient.from_service_account_file(...)``.
    fake_vision = MagicMock(spec=["Image", "ImageAnnotatorClient"])
    fake_vision.Image = MagicMock(spec=[])
    fake_vision.ImageAnnotatorClient = MagicMock()
    fake_vision.ImageAnnotatorClient.from_service_account_file.return_value = client

    return fake_vision


# ---------------------------------------------------------------------------
# Default fake annotation attributes — exported for documentation convenience
# ---------------------------------------------------------------------------
__all__ = [
    "SAMPLE_PNG_BYTES",
    "make_vision_mock",
    "sample_image_bytes",
]