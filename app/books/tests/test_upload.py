"""Tests for the books cover-upload endpoint."""
from __future__ import annotations

import pytest
from starlette import status
from starlette.testclient import TestClient

BOOKS_URL = "/v1/books/upload-cover"


class TestUploadCover:
    """POST /v1/books/upload-cover"""

    def test_upload_cover_unauthenticated_returns_401(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """No Authorization header → 401."""
        resp = client.post(
            BOOKS_URL,
            files={"file": ("cover.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == status.HTTP_401_UNAUTHORIZED

    def test_upload_invalid_file_type_returns_400(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Non-image content-type → 400."""
        resp = client.post(
            BOOKS_URL,
            files={"file": ("book.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")},
            headers=auth_headers,
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_file_exceeds_max_size_returns_413(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """File > 10 MB → 413."""
        oversized = b"0" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            BOOKS_URL,
            files={"file": ("large.png", oversized, "image/png")},
            headers=auth_headers,
        )
        assert resp.status_code == status.HTTP_413_CONTENT_TOO_LARGE

    def test_upload_valid_image_returns_200(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        sample_image_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Valid image with mocked OCR → 200 + expected response keys,
        including parsed_cover."""
        mock_result = {
            "raw_text": "Mocked OCR text with ISBN 9780135957059",
            "detected_isbns": ["9780135957059"],
            "engine_used": "Tesseract (Fallback: None)",
        }

        monkeypatch.setattr(
            "app.books.router.extract_book_text",
            lambda _image_bytes: mock_result,
        )

        resp = client.post(
            BOOKS_URL,
            files={"file": ("cover.png", sample_image_bytes, "image/png")},
            headers=auth_headers,
        )

        assert resp.status_code == status.HTTP_200_OK
        body = resp.json()
        assert body["raw_text"] == mock_result["raw_text"]
        assert body["detected_isbns"] == mock_result["detected_isbns"]
        assert body["engine_used"] == mock_result["engine_used"]
        assert "parsed_cover" in body  # new backward-compat field