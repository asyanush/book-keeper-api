"""Books routes — cover upload, OCR text extraction, cover analysis, metadata search, and related endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from fastapi.exceptions import HTTPException
from starlette import status

from app.auth.dependencies import CurrentAuthenticatedUser
from app.books.exceptions import (
    FileTooLargeError,
    InvalidFileTypeError,
    MetadataServiceUnavailable,
)
from app.books.metadata.schemas import MetadataSearchRequest, MetadataSearchResponse
from app.books.metadata.service import retrieve_book_metadata
from app.books.ocr.cover_parser import parse_cover_text
from app.books.ocr.ocr_service import extract_book_text
from app.books.schemas import CoverAnalysisResponse, CoverToMetadataResponse
from app.books.service import analyze_cover as analyze_cover_service
from app.core.config import settings
from app.core.utils import openapi_responses

router = APIRouter(prefix="/books", tags=["Books"])

# Allowed MIME types for cover image uploads.
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/upload-cover",
    response_model=CoverAnalysisResponse,
    responses=openapi_responses(InvalidFileTypeError, FileTooLargeError),
    status_code=status.HTTP_200_OK,
    operation_id="upload_book_cover",
)
async def upload_cover(
    authenticated_user: CurrentAuthenticatedUser,
    file: Annotated[UploadFile, File()],
) -> CoverAnalysisResponse:
    """Upload a book cover image, extract text via OCR, and parse structured metadata.

    Accepts JPEG, PNG, or WebP images.  The authenticated user's identity
    is validated before processing.  Returns the raw OCR text together with
    any detected ISBNs, the OCR engine that was used, and parsed cover
    metadata (title, subtitle, authors) extracted from the OCR output.

    The ``parsed_cover`` field is ``null`` when the parser cannot extract
    structured data from the cover text.
    """
    # --- Content-type validation -------------------------------------------
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise InvalidFileTypeError()

    # --- Size enforcement --------------------------------------------------
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError()

    # --- OCR + cover parsing (backward-compat; same core data) ------------
    ocr_result = extract_book_text(content)
    parsed_cover = None
    try:
        parsed_cover = parse_cover_text(
            ocr_result["raw_text"], ocr_result["detected_isbns"]
        )
    except Exception:
        pass

    return CoverAnalysisResponse(
        raw_text=ocr_result["raw_text"],
        detected_isbns=ocr_result["detected_isbns"],
        engine_used=ocr_result["engine_used"],
        parsed_cover=parsed_cover,
    )


@router.post(
    "/analyze-cover",
    response_model=CoverToMetadataResponse,
    responses=openapi_responses(InvalidFileTypeError, FileTooLargeError, MetadataServiceUnavailable),
    status_code=status.HTTP_200_OK,
    operation_id="analyze_book_cover",
)
async def analyze_cover(
    authenticated_user: CurrentAuthenticatedUser,
    file: Annotated[UploadFile, File()],
) -> CoverToMetadataResponse:
    """Run the full cover-to-metadata pipeline in a single request.

    Accepts JPEG, PNG, or WebP images.  The pipeline runs:

    1. OCR text extraction
    2. Deterministic cover-text parsing (title, subtitle, authors)
    3. External metadata search using extracted data

    The ``metadata_candidates`` list is empty when no external search
    was performed or results were found.  ``metadata_status`` indicates
    whether the search succeeded, was skipped, or failed.

    **Status  codes:**
    - **200** – Successful processing (including empty results).
    - **400** – Invalid file type.
    - **401** – Missing or invalid authentication token.
    - **413** – File too large.
    - **503** – All external providers unavailable with no cached results.
    """
    # --- Content-type validation -------------------------------------------
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise InvalidFileTypeError()

    # --- Size enforcement --------------------------------------------------
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError()

    # --- Full pipeline: OCR → parse → metadata -----------------------------
    result = await analyze_cover_service(content)

    return CoverToMetadataResponse(
        raw_text=result["raw_text"],
        detected_isbns=result["detected_isbns"],
        engine_used=result["engine_used"],
        parsed_cover=result["parsed_cover"],
        metadata_candidates=result["metadata_candidates"],
        selected_candidate=result["selected_candidate"],
        metadata_status=result["metadata_status"],
    )


@router.post(
    "/metadata/search",
    response_model=MetadataSearchResponse,
    responses=openapi_responses(MetadataServiceUnavailable),
    status_code=status.HTTP_200_OK,
    operation_id="search_book_metadata",
)
async def search_metadata(
    payload: MetadataSearchRequest,
    authenticated_user: CurrentAuthenticatedUser,
) -> MetadataSearchResponse:
    """Search for book metadata across external provider APIs.

    Accepts ISBN, title, and/or author.  At least one search parameter
    must be supplied.  Returns ranked candidate matches together with
    an optional automatically-selected best candidate when confidence
    is unambiguous.

    **Status codes:**

    - **200** – Successful search (including no results).
    - **400** – No search parameters provided.
    - **401** – Missing or invalid authentication token.
    - **422** – Invalid ISBN format / checksum.
    - **503** – All external providers unavailable with no cached results.
    """
    # --- At least one meaningful search parameter is required --------------
    if not any((payload.isbn, payload.title, payload.author)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "At least one search parameter is required. "
                "Provide an ISBN, title, or author."
            ),
        )

    # --- Run the metadata retrieval pipeline -------------------------------
    try:
        result = await retrieve_book_metadata(
            isbn=payload.isbn,
            title=payload.title,
            author=payload.author,
        )
    except MetadataServiceUnavailable:
        # Re-raise so the domain exception handler converts it to a 503.
        raise

    return MetadataSearchResponse(**result)