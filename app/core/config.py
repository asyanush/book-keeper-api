"""Application settings loaded from environment / .env file."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine environment input
RAW_ENV = os.getenv("ENVIRONMENT", "dev").lower()

# Absolute path to the directory containing .env files (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Map potential input aliases to standardized file extensions
ENV_MAP = {
    "development": "dev",
    "dev": "dev",
    "production": "prod",
    "prod": "prod",
}

ENV_SUFFIX = ENV_MAP.get(RAW_ENV, "dev")

# Point directly to project_root/.env.dev
ENV_FILE_PATH = BASE_DIR / f".env.{ENV_SUFFIX}"

class Settings(BaseSettings):
    API_NAME: str = "Book Keeper API"
    API_VERSION: str = "1.0"
    JWT_SECRET_KEY: str = Field(default=...)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_PATH: Path = BASE_DIR / "data" / "book_keeper.db"
    APP_TAGLINE: str = "Book Keeper - A simple book management application."

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()