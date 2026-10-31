from typing import ClassVar

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class DomainException(Exception):
    """Base class for domain-specific exceptions."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    message: str = "An unexpected error occurred."
    headers: ClassVar[dict[str, str] | None] = None

    def __init__(self, message: str | None = None):
        if message is not None:
            self.message = message
        super().__init__(self.message)


async def domain_exception_handler(request: Request, exc: DomainException):
    """Map domain errors to their declared status code and public message."""
    # Keep internal exception details out of the API response.
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message},
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Flatten Pydantic validation errors into the API error format."""
    # A single message is easier for clients to display consistently.
    messages = [err["msg"] for err in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"message": ", ".join(messages)},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Normalize framework HTTP errors to the same API error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
        headers=exc.headers,
    )
