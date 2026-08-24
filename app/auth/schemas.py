"""Pydantic v2 request/response schemas for the auth module."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    """Request body for POST /register."""

    username: str = Field(min_length=3, max_length=50, examples=["jane_doe"])
    password: str = Field(min_length=8, max_length=128, examples=["secret123"])


class LoginRequest(BaseModel):
    """Request body for POST /login."""

    username: str = Field(min_length=3, max_length=50, examples=["jane_doe"])
    password: str = Field(min_length=8, max_length=128, examples=["secret123"])


class TokenResponse(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"


class UserResponse(BaseModel):
    """Public representation of a user."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    created_at: datetime


class UsernameResponse(BaseModel):
    username: str
