"""Books routes — cover upload, OCR text extraction, and related endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from starlette import status

from app.auth.dependencies import CurrentAuthenticatedUser
from app.books.exceptions import FileTooLargeError, InvalidFileTypeError
from app.books.schemas import OCRResultResponse
from app.books.service import extract_cover_text
from app.core.config import settings
from app.core.utils import openapi_responses

router = APIRouter(prefix="/books", tags=["Books"])

# Allowed MIME types for cover image uploads.
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/upload-cover",
    response_model=OCRResultResponse,
    responses=openapi_responses(InvalidFileTypeError, FileTooLargeError),
    status_code=status.HTTP_200_OK,
    operation_id="upload_book_cover",
)
async def upload_cover(
    authenticated_user: CurrentAuthenticatedUser,
    file: Annotated[UploadFile, File()],
) -> OCRResultResponse:
    """Upload a book cover image and extract text via OCR.

    Accepts JPEG, PNG, or WebP images.  The authenticated user's identity
    is validated before processing.  Returns the raw OCR text together with
    any detected ISBNs and the OCR engine that was used.
    """
    # --- Content-type validation -------------------------------------------
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise InvalidFileTypeError()

    # --- Size enforcement --------------------------------------------------
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError()

    # --- OCR processing ----------------------------------------------------
    return extract_cover_text(content)