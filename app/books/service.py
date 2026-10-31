"""Business-logic layer for the Books domain."""

from app.books.ocr.ocr_service import extract_book_text
from app.books.schemas import OCRResultResponse


def extract_cover_text(image_bytes: bytes) -> OCRResultResponse:
    """Run the OCR pipeline on cover image bytes and return a structured result.

    Parameters
    ----------
    image_bytes:
        Raw image file bytes (JPEG, PNG, or WebP).

    Returns
    -------
    OCRResultResponse
        Parsed OCR output with raw text, detected ISBNs, and engine info.
    """
    result = extract_book_text(image_bytes)
    return OCRResultResponse(
        raw_text=result["raw_text"],
        detected_isbns=result["detected_isbns"],
        engine_used=result["engine_used"],
    )