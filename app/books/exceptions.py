from fastapi import status

from app.core.exceptions import DomainException


class ImageProcessingError(DomainException):
    """Raised when image decoding or processing fails."""

    status_code = status.HTTP_400_BAD_REQUEST
    message = "Could not process the provided image."


class InvalidFileTypeError(DomainException):
    """Raised when the uploaded file is not an accepted image type."""

    status_code = status.HTTP_400_BAD_REQUEST
    message = "Invalid file type. Only JPEG, PNG, and WebP images are accepted."


class FileTooLargeError(DomainException):
    """Raised when the uploaded file exceeds the maximum allowed size."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    message = "File too large."


class MetadataServiceUnavailable(DomainException):
    """Raised when all external metadata providers fail and no cached results exist."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    message = (
        "All external book metadata providers are currently unavailable "
        "and no cached results were found. Please try again later."
    )