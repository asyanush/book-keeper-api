# Roadmap

## Phase 1 — Authentication & Core API Setup

* Use the `main.py` to setup the FastAPI application instance

* Setup Authentication & JWT Infrastructure:
* Use `core/security.py` to manage password hashing (`passlib` with `bcrypt`) and JWT token creation/decoding (`PyJWT`)
* Create `db/database.py` with SQLite setup using `sqlite3` to store user credentials (`id`, `username`, `hashed_password`, `created_at`)
* Add `/register` endpoint (POST) to hash passwords and create new user accounts
* Add `/login` endpoint (POST) to verify credentials and return a Bearer JWT access token
* Create a `get_current_user` FastAPI dependency (`OAuth2PasswordBearer`) to protect secure routes
* Add `/logout` endpoint (GET) for logout
* Add `/me` endpoint (GET) to get current username of the user if the user authenticated.

## Phase 1 - Test suite

* Write test suite in `auth/tests/test_auth.py` using Starlette `TestClient`:
* `GET /` returns status 200 and response body contains the tagline text
* `POST /register` successfully creates a user
* `POST /login` returns HTTP 200 with a valid `access_token` for correct credentials
* `POST /login` returns HTTP 401 for incorrect credentials
* `GET /me` returns logged in user
* Protected endpoints reject requests missing a valid Bearer token