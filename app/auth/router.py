"""Authentication routes (register, login, me, etc.)."""

import jwt
from fastapi import APIRouter
from starlette import status

from app.auth.dependencies import (
    AuthServiceDependency,
    CurrentAuthenticatedUser,
)
from app.auth.exceptions import (
    DuplicateUsernameError,
    InvalidCredentialsError,
    InvalidTokenError,
    NotAuthenticatedError,
)
from app.auth.schemas import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UsernameResponse,
    UserResponse,
)
from app.core.schemas import MessageResponse
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
)
from app.core.utils import openapi_responses

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
    responses=openapi_responses(DuplicateUsernameError),
)
def register(
    payload: UserCreate,
    service: AuthServiceDependency,
):
    """Register a new user account.

    Hashes the password, inserts into the database, and returns the
    created user.  Returns 409 if the username is already taken.
    """

    hashed = hash_password(payload.password)

    return service.register_user(payload.username, hashed)


@router.post(
    "/login",
    response_model=TokenResponse,
    responses=openapi_responses(InvalidCredentialsError),
)
def login(payload: LoginRequest, service: AuthServiceDependency):
    """Authenticate user and return a Bearer JWT access token.

    Accepts JSON body with ``username`` and ``password``.
    Returns 401 on invalid credentials.
    """
    user = service.authenticate_user(payload.username, payload.password)
    if not user:
        raise InvalidCredentialsError()
    access_token = create_access_token(subject=user["username"])
    refresh_token = create_refresh_token(subject=user["username"])
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    responses=openapi_responses(InvalidTokenError),
)
def refresh(payload: RefreshTokenRequest, service: AuthServiceDependency):
    """Rotate a refresh token and issue a new access-token pair."""
    if service.access_token_is_revoked(payload.refresh_token):
        raise InvalidTokenError()

    try:
        username = decode_refresh_token(payload.refresh_token)
    except jwt.PyJWTError:
        raise InvalidTokenError()

    if not service.find_user(username):
        raise InvalidTokenError()

    service.revoke_access_token(payload.refresh_token)
    return TokenResponse(
        access_token=create_access_token(subject=username),
        refresh_token=create_refresh_token(subject=username),
        token_type="bearer",
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    responses=openapi_responses(NotAuthenticatedError, InvalidTokenError),
)
def logout(
    authenticated_user: CurrentAuthenticatedUser,
    service: AuthServiceDependency,
):
    """Revoke the authenticated user's access token.

    The client should discard the token after receiving the acknowledgment.
    """
    service.revoke_access_token(authenticated_user.token)
    return MessageResponse(message="Successfully logged out")


@router.get(
    "/me",
    response_model=UsernameResponse,
    responses=openapi_responses(NotAuthenticatedError, InvalidTokenError),
)
def read_me(authenticated_user: CurrentAuthenticatedUser):
    """Return the currently authenticated user's username.

    Requires a valid Bearer token.  Returns 401 if the token is missing
    or invalid.
    """
    return {"username": authenticated_user.user["username"]}
