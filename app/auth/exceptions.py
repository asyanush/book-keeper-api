from typing import ClassVar

from fastapi import status

from app.core.exceptions import DomainException


class DuplicateUsernameError(DomainException):
    """Raised when registration attempts to reuse an existing username."""

    status_code = status.HTTP_409_CONFLICT
    message = "Username already exists."


class InvalidTokenError(DomainException):
    """Raised when authentication credentials cannot be validated."""

    status_code = status.HTTP_401_UNAUTHORIZED
    message = "Could not validate credentials"
    headers: ClassVar[dict[str, str] | None] = {"WWW-Authenticate": "Bearer"}


class InvalidCredentialsError(DomainException):
    """Raised when login attempts with incorrect credentials."""

    status_code = status.HTTP_401_UNAUTHORIZED
    message = "Incorrect username or password"