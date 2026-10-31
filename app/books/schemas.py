"""Pydantic schemas for the Books domain."""

from pydantic import BaseModel


class OCRResultResponse(BaseModel):
    """Response model for the OCR text-extraction endpoint."""

    raw_text: str
    detected_isbns: list[str]
    engine_used: str