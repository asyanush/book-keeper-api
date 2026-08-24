"""Business logic for authentication (login, user verification)."""

import sqlite3

from app.auth.exceptions import DuplicateUsernameError
from app.auth.repository import AuthRepository
from app.core.security import hash_token, verify_password


class AuthService:
    def __init__(self, repository: AuthRepository):
        self.repository = repository

    def find_user(self, username: str) -> dict | None:
        """Return the public user fields for the given username, if it exists."""
        user = self.repository.get_user_by_username(username)
        if user is None:
            return None
        return {
            "id": user["id"],
            "username": user["username"],
            "created_at": user["created_at"],
        }


    def register_user(self, username: str, hashed_password: str) -> dict:
        """Create a user with an already-hashed password."""
        try:
            return self.repository.create_user(username, hashed_password)
        except sqlite3.IntegrityError as exc:
            raise DuplicateUsernameError from exc


    def revoke_access_token(self, token: str) -> None:
        """Revoke an access token without storing the token itself."""
        self.repository.revoke_token(hash_token(token))


    def access_token_is_revoked(self, token: str) -> bool:
        """Return whether an access token has been revoked."""
        return self.repository.is_token_revoked(hash_token(token))


    def authenticate_user(self, username: str, password: str) -> dict | None:
        """Verify credentials and return the user dict or None if invalid."""
        user = self.repository.get_user_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user["hashed_password"]):
            return None
        return user
