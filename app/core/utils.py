from fastapi import status

from app.core.exceptions import DomainException
from app.core.schemas import ErrorResponse


def openapi_responses(*exceptions: type[DomainException]) -> dict:
    """Generate OpenAPI responses for the given exceptions."""
    responses = {}
    for exc in exceptions:
        responses[exc.status_code] = {
            "model": ErrorResponse,
            "description": exc.message,
        }
    # Add a default response for validation errors (422) to ensure consistent error format.
    responses[status.HTTP_422_UNPROCESSABLE_CONTENT] = {
        "model": ErrorResponse,
        "description": "Validation error",
    }    
    return responses