import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.database import get_db
from app.main import app


@pytest.fixture(autouse=True)
def fast_bcrypt(monkeypatch):
    monkeypatch.setattr(settings, "BCRYPT_ROUNDS", 4)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_PATH", tmp_path / "test.db")
    get_db.cache_clear()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        get_db.cache_clear()
