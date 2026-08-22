# Tech Stack

| Layer | Choice | Version | Notes |
| --- | --- | --- | --- |
| Language | Python 3 | — | Managed with `requirements-dev.txt` |
| Web framework | `fastapi` | - | ASGI, type-driven RESTful API & JSON routes |
| Auth & Security | PyJWT + Passlib | — | JWT access tokens & `bcrypt` password hashing |
| OCR Engine | `pytesseract` + Pillow | — | Optical character recognition for book cover image parsing |
| Data model | `pydantic` / `dataclasses` | — | User, Token, Book, and Metadata schemas |
| Storage | SQLite (`sqlite3`) | — | Relational database for users, tokens, and book catalog |
| Server | Uvicorn | — | ASGI server, run via `main.py` |
| Testing | pytest + `TestClient` | - | `starlette.testclient.TestClient`, no running server needed |

When running Python: you **must** use the virtual environment.