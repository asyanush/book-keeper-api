"""Pydantic schemas for the Books domain."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.books.metadata.schemas import MetadataCandidateResponse
from app.books.ocr.cover_parser import ParsedCoverData


class OCRResultResponse(BaseModel):
    """Response model for the OCR text-extraction endpoint."""

    raw_text: str
    detected_isbns: list[str]
    engine_used: str


class CoverAnalysisResponse(OCRResultResponse):
    """Extended OCR result with parsed cover metadata."""

    parsed_cover: ParsedCoverData | None = None


class CoverToMetadataResponse(CoverAnalysisResponse):
    """Full pipeline result: OCR + cover parsing + metadata search."""

    metadata_candidates: list[MetadataCandidateResponse] = Field(default_factory=list)
    selected_candidate: MetadataCandidateResponse | None = None
    metadata_status: str = "success"  # or "skipped", "failed"