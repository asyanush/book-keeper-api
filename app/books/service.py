"""Business-logic layer for the Books domain."""

from app.books.ocr.cover_parser import parse_cover_text
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


async def analyze_cover(
    image_bytes: bytes,
    *,
    search_metadata: bool = True,
) -> dict:
    """Run the full cover analysis pipeline: OCR → parse → metadata search.

    Parameters
    ----------
    image_bytes:
        Raw image file bytes (JPEG, PNG, or WebP).
    search_metadata:
        When ``True`` (the default) the pipeline follows cover parsing
        with a metadata search when enough information is available
        (ISBN, title, or author).  Set to ``False`` to skip the
        external provider round-trip and return only the OCR+parsed
        result.

    Returns
    -------
    dict
        Keys: ``raw_text``, ``detected_isbns``, ``engine_used``,
        ``parsed_cover``, ``metadata_candidates``,
        ``selected_candidate``, ``metadata_status``.
    """
    # ------------------------------------------------------------------
    # Step 1: OCR
    # ------------------------------------------------------------------
    ocr_result = extract_book_text(image_bytes)
    raw_text = ocr_result["raw_text"]
    detected_isbns = ocr_result["detected_isbns"]
    engine_used = ocr_result["engine_used"]

    # ------------------------------------------------------------------
    # Step 2: Parse cover text (graceful failure)
    # ------------------------------------------------------------------
    try:
        parsed_cover = parse_cover_text(raw_text, detected_isbns)
    except Exception:
        parsed_cover = None

    # ------------------------------------------------------------------
    # Step 3: Metadata search (optional, graceful failure)
    # ------------------------------------------------------------------
    metadata_candidates: list = []
    selected_candidate = None
    metadata_status = "skipped"

    if search_metadata and parsed_cover:
        has_data = bool(
            parsed_cover.isbns
            or parsed_cover.title
            or parsed_cover.authors
        )
        if has_data:
            try:
                from app.books.metadata.service import retrieve_book_metadata

                isbn = parsed_cover.isbns[0] if parsed_cover.isbns else None
                author = parsed_cover.authors[0] if parsed_cover.authors else None

                metadata_result = await retrieve_book_metadata(
                    isbn=isbn,
                    title=parsed_cover.title,
                    author=author,
                )
                metadata_candidates = metadata_result.get("candidates", [])
                selected_candidate = metadata_result.get("selected_candidate")
                metadata_status = "success"
            except Exception:
                metadata_status = "failed"

    return {
        "raw_text": raw_text,
        "detected_isbns": detected_isbns,
        "engine_used": engine_used,
        "parsed_cover": parsed_cover,
        "metadata_candidates": metadata_candidates,
        "selected_candidate": selected_candidate,
        "metadata_status": metadata_status,
    }