"""FastAPI authentication dependencies."""

from typing import Annotated

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.exceptions import InvalidTokenError, NotAuthenticatedError
from app.auth.repository import AuthRepository
from app.auth.service import AuthService
from app.core.security import decode_access_token
from app.db.database import Database, get_db

# Dependency to extract Bearer auth credentials from the Authorization header.
bearer_scheme = HTTPBearer(auto_error=False)

AuthenticatedCredentials = Annotated[
    HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
]


def get_auth_repository(
    db: Annotated[Database, Depends(get_db)],
) -> AuthRepository:
    """Build an AuthRepository backed by the shared Database instance."""
    return AuthRepository(db)


def get_auth_service(
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
) -> AuthService:
    """Build an AuthService with its repository dependency."""
    return AuthService(repository)


AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]


def get_current_user(
    token: AuthenticatedCredentials, service: AuthServiceDependency
) -> dict:
    """Validate a Bearer JWT and return the authenticated public user."""
    if token is None:
        raise NotAuthenticatedError()

    if service.access_token_is_revoked(token.credentials):
        raise InvalidTokenError()

    try:
        username = decode_access_token(token.credentials)
    except jwt.PyJWTError:
        raise InvalidTokenError()

    user = service.find_user(username)
    if not user:
        raise InvalidTokenError()

    return user


CurrentUser = Annotated[dict, Depends(get_current_user)]


class AuthenticatedUser:
    """Bundle the validated user and token for authenticated routes."""

    def __init__(
        self,
        token: AuthenticatedCredentials,
        user: CurrentUser,
    ):
        if token is None:
            raise NotAuthenticatedError()
        self.token = token.credentials
        self.user = user

CurrentAuthenticatedUser = Annotated[AuthenticatedUser, Depends()]