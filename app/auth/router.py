"""Authentication routes (register, login, me, etc.)."""

from typing import Annotated

from fastapi import APIRouter, Depends
from starlette import status

from app.auth.dependencies import AuthenticatedUser, get_auth_service
from app.auth.exceptions import (
    DuplicateUsernameError,
    InvalidCredentialsError,
)
from app.auth.schemas import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UsernameResponse,
    UserResponse,
)
from app.auth.service import AuthService
from app.core.schemas import MessageResponse
from app.core.security import create_access_token, hash_password
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
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    """Register a new user account.

    Hashes the password, inserts into the database, and returns the
    created user.  Returns 409 if the username is already taken.
    """
    if service.find_user(payload.username):
        raise DuplicateUsernameError()

    hashed = hash_password(payload.password)
    try:
        user = service.register_user(payload.username, hashed)
    except DuplicateUsernameError:
        raise DuplicateUsernameError()
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    responses=openapi_responses(InvalidCredentialsError),
)
def login(
    payload: LoginRequest, service: Annotated[AuthService, Depends(get_auth_service)]
):
    """Authenticate user and return a Bearer JWT access token.

    Accepts JSON body with ``username`` and ``password``.
    Returns 401 on invalid credentials.
    """
    user = service.authenticate_user(payload.username, payload.password)
    if not user:
        raise InvalidCredentialsError()
    access_token = create_access_token(subject=user["username"])
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.post("/logout", response_model=MessageResponse)
def logout(
    authenticated_user: Annotated[AuthenticatedUser, Depends()],
    service: Annotated[AuthService, Depends(get_auth_service)],
):
    """Revoke the authenticated user's access token.

    The client should discard the token after receiving the acknowledgment.
    """
    service.revoke_access_token(authenticated_user.token)
    return MessageResponse(message="Successfully logged out")


@router.get("/me", response_model=UsernameResponse)
def read_me(authenticated_user: Annotated[AuthenticatedUser, Depends()]):
    """Return the currently authenticated user's username.

    Requires a valid Bearer token.  Returns 401 if the token is missing
    or invalid.
    """
    return {"username": authenticated_user.user["username"]}
