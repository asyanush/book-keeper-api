"""Tests for the auth module."""
from datetime import datetime

from starlette import status

AUTH_URL_PREFIX = "/v1/auth"


def test_register_creates_user(client):
    resp = client.post(
        f"{AUTH_URL_PREFIX}/register",
        json={"username": "alice", "password": "secret123"},
    )
    assert resp.status_code == status.HTTP_201_CREATED
    body = resp.json()
    assert body["username"] == "alice"
    assert "id" in body
    assert datetime.fromisoformat(body["created_at"])


def test_register_rejects_empty_password(client):
    resp = client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "alice", "password": ""}
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_register_rejects_short_password(client):
    resp = client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "alice", "password": "a"}
    )
    assert resp.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_register_rejects_duplicate_username(client):
    payload = {"username": "alice", "password": "secret123"}
    assert (
        client.post(f"{AUTH_URL_PREFIX}/register", json=payload).status_code
        == status.HTTP_201_CREATED
    )

    resp = client.post(f"{AUTH_URL_PREFIX}/register", json=payload)

    assert resp.status_code == status.HTTP_409_CONFLICT
    assert resp.json()["message"] == "Username already exists."


def test_login_returns_access_token(client):
    client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "bob", "password": "secret123"}
    )
    resp = client.post(
        f"{AUTH_URL_PREFIX}/login", json={"username": "bob", "password": "secret123"}
    )
    assert resp.status_code == status.HTTP_200_OK
    body = resp.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"


def test_login_invalid_credentials_returns_401(client):
    client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "bob", "password": "secret123"}
    )
    resp = client.post(
        f"{AUTH_URL_PREFIX}/login", json={"username": "bob", "password": "wrongpass"}
    )
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


def test_me_returns_logged_in_user(client):
    client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "bob", "password": "secret123"}
    )
    token = client.post(
        f"{AUTH_URL_PREFIX}/login", json={"username": "bob", "password": "secret123"}
    ).json()["access_token"]
    resp = client.get(
        f"{AUTH_URL_PREFIX}/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["username"] == "bob"


def test_logout_revokes_access_token(client):
    client.post(
        f"{AUTH_URL_PREFIX}/register", json={"username": "bob", "password": "secret123"}
    )
    token = client.post(
        f"{AUTH_URL_PREFIX}/login", json={"username": "bob", "password": "secret123"}
    ).json()["access_token"]

    resp = client.post(
        f"{AUTH_URL_PREFIX}/logout", headers={"Authorization": f"Bearer {token}"}
    )

    assert resp.status_code == status.HTTP_200_OK
    assert (
        client.get(
            f"{AUTH_URL_PREFIX}/me", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == status.HTTP_401_UNAUTHORIZED
    )


def test_logout_requires_authentication(client):
    resp = client.post(f"{AUTH_URL_PREFIX}/logout")

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp.json()["message"] == "Not authenticated"
    assert resp.headers["www-authenticate"] == "Bearer"


def test_protected_endpoint_rejects_missing_token(client):
    resp = client.get(f"{AUTH_URL_PREFIX}/me")
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    resp_bad = client.get(
        f"{AUTH_URL_PREFIX}/me", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert resp_bad.status_code == status.HTTP_401_UNAUTHORIZED
    assert resp_bad.headers["www-authenticate"] == "Bearer"
