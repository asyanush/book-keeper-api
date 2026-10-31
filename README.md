# Book Keeper

A personal book-cataloging API. Snap a photo of a book's cover and let the
system extract the title, author, and ISBN via OCR so books can be catalogued
with minimal manual entry.

- **Web framework:** FastAPI (ASGI, type-driven)
- **Auth:** JWT (access + refresh) + bcrypt password hashing
- **OCR:** Tesseract (local) with Google Cloud Vision fallback/primary
- **Storage:** SQLite
- **Server:** Uvicorn

---

## Table of contents

1. [Architecture](#architecture)
2. [Project layout](#project-layout)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Running the API](#running-the-api)
6. [API reference](#api-reference)
7. [Cover recognition (OCR)](#cover-recognition-ocr)
8. [OCR CLI tool](#ocr-cli-tool)
9. [Testing](#testing)

---

## Architecture

Book Keeper follows a layered, per-domain structure:

```
Client ──▶ FastAPI (app/main.py)
              ├── /v1/auth/**   → Auth domain (register/login/refresh/logout/me)
              └── /v1/books/**  → Books domain (cover upload → OCR)
```

Each domain (`auth`, `books`) follows the same internal layering:

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| Router | `router.py` | HTTP endpoints, request validation, auth guards |
| Service | `service.py` | Business logic |
| Repository | `repository.py` | SQLite data access (auth domain) |
| Schemas | `schemas.py` | Pydantic request/response models |
| Exceptions | `exceptions.py` | Domain-specific `DomainException` subclasses |

Cross-cutting concerns live under `app/core` (config, logging, security,
shared error handling) and `app/db` (SQLite wrapper). The OCR pipeline is a
sub-package of the books domain: `app/books/ocr/`.

### Auth & security

- Passwords are hashed with **bcrypt** (passlib).
- Login issues an **access token** (short-lived, `type: "access"`) and a
  **refresh token** (longer-lived, `type: "refresh"`). Both are JWTs (HS256)
  carrying `sub`, `type`, `jti`, and `exp` claims.
- Logout/refresh rotate tokens against a server-side `revoked_tokens`
  table (storing SHA-256 hashes, never raw tokens).

### OCR pipeline

See [Cover recognition](#cover-recognition-ocr) for the full flow.

---

## Project layout

```
app/
├── main.py                 # FastAPI app instance + exception handlers
├── api/v1/api.py           # Aggregates auth + books routers under /v1
├── core/                   # config, logging, security, shared schemas/utils
├── db/                     # SQLite Database wrapper
├── auth/                   # register/login/refresh/logout/me
│   ├── router.py, service.py, repository.py
│   └── dependencies.py     # get_current_user / AuthenticatedUser guards
└── books/
    ├── router.py, service.py, schemas.py, exceptions.py
    ├── ocr/
    │   ├── image_processing.py   # grayscale + CLAHE preprocessing
    │   └── ocr_service.py        # engine orchestration + ISBN parsing
    └── tests/
scripts/                    # dev/CLI helpers
specs/                      # mission, roadmap, tech-stack
data/                       # SQLite database file (gitignored)
```

---

## Setup

> You **must** use the virtual environment when running Python.

```bash
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install pip-tools
python -m pip install pip-tools

# 5. Compile the dev requirements
pip-compile requirements-dev.in

# 6. Install dependencies
pip install -r requirements-dev.txt

# 7. Create the local environment file
cp .env.example .env.dev   # then fill in real values
```

### Tesseract

The Tesseract engine requires the native binary in addition to the
`pytesseract` Python package:

```bash
# macOS
brew install tesseract

# Debian/Ubuntu
sudo apt-get install tesseract-ocr
```

Google Cloud Vision is optional (see [Configuration](#configuration)).

---

## Configuration

Settings are read from `.env.{ENVIRONMENT}` (defaults to `.env.dev`), selected
by the `ENVIRONMENT` environment variable (`dev`, `development`, `prod`,
`production`). See `app/core/config.py` for the full list.

| Variable | Default | Description |
| --- | --- | --- |
| `JWT_SECRET_KEY` | *(required)* | HMAC secret for signing JWTs |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access-token lifetime |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh-token lifetime |
| `BCRYPT_ROUNDS` | `12` | bcrypt cost factor |
| `DATABASE_PATH` | `data/book_keeper.db` | SQLite file location |
| `OCR_STRATEGY` | `HYBRID` | `TESSERACT` \| `VISION` \| `HYBRID` |
| `MAX_UPLOAD_SIZE_MB` | `10` | Cover-upload size limit |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(unset)* | Path to GCP service-account JSON (enables Vision) |

### OCR strategy matrix

| `OCR_STRATEGY` | Primary engine | Fallback engine |
| --- | --- | --- |
| `TESSERACT` | Tesseract | Vision |
| `VISION` | Vision | Tesseract |
| `HYBRID` | Tesseract | Vision |

`HYBRID` and `TESSERACT` behave identically today; the fallback engine is
consulted when the primary result is empty, too short, or looks like noise
(see [Cover recognition](#cover-recognition-ocr)).

---

## Running the API

```bash
uvicorn app.main:app --reload
```

Interactive docs are auto-generated at `/docs` (Swagger UI) and `/redoc`.

---

## API reference

All domain routes are mounted under `/v1`. Errors are returned uniformly as
`{"message": "..."}` (see `app/core/exceptions.py`).

### System

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service info and health status |
| `GET` | `/health` | Health check |

### Auth — `/v1/auth`

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/v1/auth/register` | — | Create a user. `{username, password}` → `{id, username, created_at}` (201). 409 if username taken |
| `POST` | `/v1/auth/login` | — | `{username, password}` → `{access_token, refresh_token, token_type}`. 401 on bad credentials |
| `POST` | `/v1/auth/refresh` | — | `{refresh_token}` → new token pair |
| `POST` | `/v1/auth/logout` | Bearer | Revoke the presented access token → `{message}` |
| `GET` | `/v1/auth/me` | Bearer | Current user's `{username}` |

### Books — `/v1/books`

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `POST` | `/v1/books/upload-cover` | Bearer | Upload a cover image (`multipart/form-data`, field `file`) and return OCR results |

**`POST /v1/books/upload-cover`**

- Accepts `image/jpeg`, `image/png`, `image/webp` (400 otherwise).
- Rejects files over `MAX_UPLOAD_SIZE_MB` (413).
- Response:

```json
{
  "raw_text": "The Name of the Wind\nPatrick Rothfuss",
  "detected_isbns": [],
  "engine_used": "Tesseract (Fallback: None)"
}
```

| Field | Description |
| --- | --- |
| `raw_text` | Raw text extracted from the image |
| `detected_isbns` | ISBN-10 / ISBN-13 candidates found in the text (normalised, deduplicated) |
| `engine_used` | Which engines ran, e.g. `"Vision (Fallback: Tesseract)"` |

---

## Cover recognition (OCR)

The pipeline lives in `app/books/ocr/` and is invoked by the upload endpoint
(or the CLI tool):

```
image bytes
   │
   ▼
preprocess_image()          image_processing.py
   │  decode → grayscale → CLAHE contrast normalization
   ▼
primary engine              ocr_service.py
   │  Tesseract or Vision (per OCR_STRATEGY)
   ▼
fallback engine?            runs if primary output is empty,
   │                        under 2 words, OR "looks like noise"
   ▼
_parse_isbns()              ISBN-10 / ISBN-13 regex extraction
   │
   ▼
{"raw_text", "detected_isbns", "engine_used"}
```

### Preprocessing

`preprocess_image` decodes the image, converts it to grayscale, and applies
**CLAHE** (contrast-limited adaptive histogram equalization) to even out
shadows, reflections, and uneven lighting. The image is deliberately **not**
binarized: aggressive thresholding turns cover gradients and artwork into
speckle that Tesseract misreads as characters (e.g. `ARARARIRA`, `=<`).

### Engines

- **Tesseract** (`extract_text_tesseract`) — local, offline, fast. Strong on
  clean documents; weak on stylized/decorative cover fonts.
- **Google Vision** (`extract_text_vision`) — cloud text detection + barcode
  parsing. Handles stylized text and glossy/curved covers far better, but
  requires `GOOGLE_APPLICATION_CREDENTIALS`.

### Fallback heuristic

The fallback engine runs when the primary result is empty, has fewer than 2
words, or fails a noise check (`_looks_like_noise`), which flags output
dominated by artifact symbols (`\ = < > | …`) or implausible "words" (missing
vowels, 3+ repeated characters).

> **Tip for stylized covers:** Tesseract alone is unreliable on decorative
> cover fonts. Set `OCR_STRATEGY=VISION` (or keep `HYBRID`) and
> `GOOGLE_APPLICATION_CREDENTIALS` to route covers through Vision.

---

## OCR CLI tool

`scripts/ocr_image.py` runs the exact same pipeline against local files — no
server or auth required. Iterating here is the fastest way to tune
preprocessing or engine choice:

```bash
# From the project root
python scripts/ocr_image.py cover.jpg
python scripts/ocr_image.py a.jpg b.png                # multiple files
python scripts/ocr_image.py cover.jpg --strategy VISION
python scripts/ocr_image.py cover.jpg --save-processed out.png
```

---

## Testing

```bash
pytest
```

The OCR tests mock `cv2`, `numpy`, `pytesseract`, and `google.cloud.vision`
via `sys.modules` patching, so they run without the native Tesseract binary or
live GCP credentials.