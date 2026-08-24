"""User and token repository — SQLite data access for the auth module."""

from datetime import datetime, timezone

from app.db.database import Database


class AuthRepository:
    """Repository for user and token data access."""

    def __init__(self, db: Database):
        self.db = db

    def get_user_by_username(self, username: str) -> dict | None:
        """Return a user dict (id, username, hashed_password, created_at) or None."""
        return self.db.fetch_one(
            "SELECT id, username, hashed_password, created_at FROM users WHERE username = ?",
            (username,),
        )

    def create_user(self, username: str, hashed_password: str) -> dict:
        """Insert a new user and return a dict with id, username, and created_at."""
        created_at = datetime.now(timezone.utc).isoformat()
        user_id = self.db.execute(
            "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
            (username, hashed_password, created_at),
        )
        return {"id": user_id, "username": username, "created_at": created_at}

    def revoke_token(self, token_hash: str) -> None:
        """Persist a token hash so the token can no longer be authenticated."""
        revoked_at = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT OR IGNORE INTO revoked_tokens (token_hash, revoked_at) VALUES (?, ?)",
            (token_hash, revoked_at),
        )

    def is_token_revoked(self, token_hash: str) -> bool:
        """Return whether a token hash has been revoked."""
        return self.db.exists(
            "SELECT 1 FROM revoked_tokens WHERE token_hash = ?",
            (token_hash,),
        )