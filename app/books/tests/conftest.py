"""Feature-level pytest fixtures for the ``app/books/tests/`` package.

Fixtures defined here are discovered automatically by pytest and are available
to all test modules under ``app/books/tests/`` without any explicit import.

These fixtures are **additive** — they do not modify or override any existing
helpers or fixtures in ``test_upload.py`` or ``test_ocr_service.py``.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette import status
from starlette.testclient import TestClient

from app.books.tests.fixtures import SAMPLE_PNG_BYTES, make_vision_mock
from app.core.config import settings

# ---------------------------------------------------------------------------
# Sample image
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return small in-memory PNG bytes (1×1 red pixel).

    Useful as a stand-in for image upload or OCR tests that need realistic
    image data without requiring Pillow or other image-processing libraries.
    """
    return SAMPLE_PNG_BYTES


# ---------------------------------------------------------------------------
# Google Cloud Vision mock (lazy-import-safe)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vision_client(
    monkeypatch: pytest.MonkeyPatch,
    request: Any,
) -> Generator[MagicMock, None, None]:
    """Install a fake ``google.cloud.vision`` module into ``sys.modules``.

    The fixture patches ``sys.modules["google.cloud.vision"]`` **before** any
    lazy import in ``app.books.ocr.ocr_service.extract_text_vision`` can
    resolve it, so the tests never hit a live GCP endpoint.

    By default, the fake returns empty text and no barcodes.  Customise the
    return value through fixture parametrisation or by accessing the yielded
    fake module::

        # In a test body:
        fake_vision = mock_vision_client  # already-patched module
        client = fake_vision.ImageAnnotatorClient.from_service_account_file.return_value
        client.text_detection.return_value.text_annotations[0].description = (
            "Found ISBN 9780135957059"
        )

    Or set a marker::

        @pytest.mark.parametrize("mock_vision_client", [
            {"return_text": "Hello", "return_barcodes": ["9780135957059"]},
        ], indirect=True)

    Yields
    ------
    MagicMock
        The fake ``google.cloud.vision`` module object.  The client instance
        can be retrieved as
        ``mock_vision_client.ImageAnnotatorClient.from_service_account_file.return_value``.
    """
    # Allow parametrisation via ``indirect=True`` markers.
    if hasattr(request, "param"):
        kwargs = request.param if isinstance(request.param, dict) else {}
    else:
        kwargs = {}

    fake_vision = make_vision_mock(
        return_text=kwargs.get("return_text", ""),
        return_barcodes=kwargs.get("return_barcodes", None),
    )

    # ``extract_text_vision`` bails out unless a credential path is set, so
    # point the setting at a dummy path to keep the mocked path exercised.
    monkeypatch.setattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", "/fake/creds.json")

    # The lazy import in extract_text_vision does:
    #     from google.cloud import vision
    # Python's import machinery needs ``google`` and ``google.cloud`` to exist
    # as proper (namespace) packages in ``sys.modules`` before it can resolve
    # ``google.cloud.vision``.  We create them here as lightweight modules.
    _install_fake_google_namespace(monkeypatch, fake_vision)

    yield fake_vision


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _install_fake_google_namespace(
    monkeypatch: pytest.MonkeyPatch,
    fake_vision: MagicMock,
) -> None:
    """Insert stub ``google`` / ``google.cloud`` namespace packages into
    ``sys.modules`` so that ``from google.cloud import vision`` resolves.

    Must be called **before** any code triggers a lazy import of
    ``google.cloud.vision``.
    """
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # namespace-package marker
    google_pkg.__package__ = "google"

    google_cloud = types.ModuleType("google.cloud")
    google_cloud.__path__ = []
    google_cloud.__package__ = "google.cloud"
    google_cloud.vision = fake_vision  # So ``google.cloud.vision`` is the mock

    google_pkg.cloud = google_cloud

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.vision", fake_vision)


# ---------------------------------------------------------------------------
# Authenticated request headers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Register + login a temporary user and return ``Authorization`` headers.

    Each invocation creates a unique user so tests can safely call this
    fixture without cross-test interference.

    Returns
    -------
    dict[str, str]
        ``{"Authorization": "Bearer <JWT-token>"}``
    """
    import uuid

    username = f"testuser-{uuid.uuid4().hex[:8]}"
    password = "secret123"

    # Register
    resp = client.post(
        "/v1/auth/register",
        json={"username": username, "password": password},
    )
    assert resp.status_code == status.HTTP_201_CREATED, (
        f"User registration failed: {resp.text}"
    )

    # Login
    resp = client.post(
        "/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert resp.status_code == status.HTTP_200_OK, (
        f"Login failed: {resp.text}"
    )
    token: str = resp.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def authenticated_client(
    client: TestClient,
    auth_headers: dict[str, str],
) -> TestClient:
    """A ``TestClient`` whose default headers include the JWT bearer token.

    Mirrors the fixture of the same name defined for the metadata test suite
    (``app/books/metadata/tests/conftest.py``) so endpoint tests living in
    ``app/books/tests/`` can share the same authentication convention.
    """
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def unauthenticated_client(client: TestClient) -> TestClient:
    """A ``TestClient`` with no ``Authorization`` header.

    .. note::
       Do not request this fixture together with ``authenticated_client`` in
       the same test, since both wrap the same per-test ``client`` instance.
    """
    client.headers.pop("Authorization", None)
    return client
