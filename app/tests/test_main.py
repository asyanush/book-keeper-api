"""Tests for application-level endpoints."""

from app.core.config import settings


def test_root_returns_metadata(client):
    resp = client.get("/")

    assert resp.status_code == 200
    assert resp.json() == {
        "name": settings.API_NAME,
        "version": settings.API_VERSION,
        "tagline": settings.APP_TAGLINE,
        "status": "healthy",
    }


def test_health_returns_status_and_version(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "name": settings.API_NAME,
        "status": "healthy",
        "version": settings.API_VERSION,
    }
