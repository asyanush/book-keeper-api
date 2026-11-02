# Phase 3 — Web Scraping & API Integration (Book Metadata Retrieval)

## Overview

Phase 3 builds on the OCR functionality implemented in Phase 2. Its purpose is to transform extracted text and ISBN numbers into structured, normalized book metadata using external catalog APIs, with web scraping as a controlled fallback.

**Main design principle: API-first, scraping-second.**

Google Books and Open Library should be the primary sources. Amazon and Goodreads should not be mandatory dependencies.

**Architecture:**

`OCR → Metadata Search → Candidate Matching → Metadata Normalization → Response`

---

# Part 1 — Implementer Agent Instructions

**Role:** Backend Engineer

**Objective:** Implement a multi-provider book metadata retrieval system that accepts OCR-extracted information (ISBN, title, author) and returns reliable, normalized book metadata.

## 1. Dependency & Configuration Management

Use existing project dependencies where possible. Avoid introducing Selenium unless a demonstrated requirement cannot be handled with HTTP requests.

Update `app/core/config.py` with:

| Setting                | Default              | Description                 |
| ---------------------- | -------------------- | --------------------------- |
| `GOOGLE_BOOKS_API_URL` | Google Books API URL | Primary metadata endpoint   |
| `OPEN_LIBRARY_API_URL` | Open Library API URL | Secondary metadata endpoint |
| `GOOGLE_BOOKS_API_KEY` | `None`               | Optional API key            |
| `METADATA_TIMEOUT`     | `10`                 | HTTP timeout in seconds     |
| `METADATA_MAX_RETRIES` | `2`                  | Maximum retry attempts      |
| `METADATA_CACHE_TTL`   | `86400`              | Cache lifetime in seconds   |
| `METADATA_STRATEGY`    | `HYBRID`             | Provider selection strategy |
| `SCRAPING_ENABLED`     | `False`              | Enable scraping fallback    |

Requirements:

- Load configuration from environment variables.
- Never hardcode API credentials.
- Use configurable request timeouts.
- Do not expose API keys in logs or API responses.

## 2. Proposed Project Structure

```
app/
├── books/
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   ├── normalizer.py
│   │   ├── matcher.py
│   │   ├── cache.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── google_books.py
│   │       ├── open_library.py
│   │       └── scraper.py
```

## 3. Metadata Provider Interface

Implement a common provider interface in `providers/base.py`.

Each provider should expose:

- `search_by_isbn(isbn: str)`
- `search_by_title_author(title: str, author: str | None)`
- `get_book_details(provider_id: str)`

Requirements:

- Consistent return format across providers.
- Handle network errors independently.
- Return empty results instead of crashing the whole pipeline.
- Keep provider-specific response parsing inside each provider module.

## 4. Google Books API Integration (Primary)

Documentation: [https://developers.google.com/books/docs/v1/using](https://developers.google.com/books/docs/v1/using)

Implement `app/books/metadata/providers/google_books.py`.

Supported operations:

1. Search by ISBN using `q=isbn:{isbn}`.
2. Search by title using `q=intitle:{title}`.
3. Search by author using `q=inauthor:{author}`.
4. Retrieve detailed information using volume ID.

Extract the following fields:

- Title
- Subtitle
- Authors
- Publisher
- Published date
- Description
- ISBN-10
- ISBN-13
- Page count
- Categories
- Language
- Cover image URLs
- Google Books volume ID

Handle:

- HTTP 429 rate limiting.
- HTTP 5xx server errors.
- HTTP 400/404 responses.
- Empty search results.
- Missing or incomplete fields.
- Malformed JSON responses.

Do not assume every book has an ISBN or cover image.

## 5. Open Library API Integration (Secondary)

Documentation: [https://openlibrary.org/developers/api](https://openlibrary.org/developers/api)

Implement `app/books/metadata/providers/open_library.py`.

Support:

1. ISBN lookup using `/isbn/{isbn}.json`.
2. Search using `/search.json`.
3. Edition lookup.
4. Work metadata lookup where applicable.

Extract:

- Title
- Authors
- Publisher
- Publication date
- ISBN-10 / ISBN-13
- Number of pages
- Subjects
- Language
- Cover ID / cover URL
- Open Library edition ID
- Open Library work ID

Important:

- Resolve author IDs into author names when necessary.
- Handle missing edition and work relationships.
- Avoid making unnecessary requests for every search result.

## 6. Metadata Normalization

Implement `app/books/metadata/normalizer.py`.

All providers must return a common metadata structure.

Recommended schema:

```
class BookMetadata:
    title: str | None
    subtitle: str | None
    authors: list[str]
    publisher: str | None
    published_date: str | None
    description: str | None
    isbn_10: str | None
    isbn_13: str | None
    page_count: int | None
    categories: list[str]
    language: str | None
    cover_url: str | None
    source: str
    source_id: str | None
    confidence: float
```

Requirements:

- Normalize whitespace and Unicode characters.
- Normalize ISBN-10 and ISBN-13.
- Preserve original publication date information where possible.
- Convert provider-specific author formats into `list[str]`.
- Remove duplicate authors and categories.
- Validate ISBN checksums.
- Do not fabricate missing metadata.
- Keep source attribution for every candidate.

**Important architectural decision:** Preserve provider-specific metadata and provenance. Do not discard original source values during normalization.

## 7. Candidate Matching & Deduplication

Implement `app/books/metadata/matcher.py`.

This module determines whether returned results actually represent the book identified by OCR.

Matching priority:

1. Exact valid ISBN-13 match.
2. Exact valid ISBN-10 match.
3. Normalized title + author match.
4. Title similarity + author similarity.
5. Partial metadata match.

Requirements:

- ISBN mismatch must reduce candidate confidence significantly.
- Account for punctuation, capitalization and subtitle differences.
- Handle multiple authors.
- Handle different editions of the same book.
- Distinguish between a work and an individual edition.
- Never select a candidate solely because it is the first API result.

Use deterministic matching initially. Do not introduce an LLM dependency in this phase.

## 8. Metadata Retrieval Orchestrator

Implement `app/books/metadata/service.py`.

Expose:

```
def retrieve_book_metadata(
    isbn: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    ...
```

Suggested retrieval sequence:

1. Search Google Books by ISBN first when a valid ISBN is available.
2. If no reliable result is found, query Open Library.
3. If only title and author are available, search both providers.
4. Merge complementary metadata only when candidate matching establishes that results represent the same edition or sufficiently compatible book record.
5. Prefer exact ISBN matches over title similarity.
6. Preserve conflicting values rather than silently overwriting them.
7. Invoke scraping only if enabled and API results remain insufficient.
8. Return multiple candidates when confidence is ambiguous.

## 9. Web Scraping Fallback

Implement `app/books/metadata/providers/scraper.py`.

Initial scope:

- Goodreads (only if its public pages are accessible and permitted).
- Amazon (only if its public pages are accessible and permitted).

Requirements:

- Use `requests` and `BeautifulSoup4` or `lxml`.
- Extract metadata from structured page data such as JSON-LD where available.
- Respect `robots.txt`, website terms of service, rate limits and access restrictions.
- Do not bypass CAPTCHAs, login requirements, anti-bot measures or other access controls.
- Do not use Selenium in the initial implementation.
- Use scraping only as an optional fallback.
- Do not scrape search engine result pages.
- Log provider failures without exposing sensitive information.

If a website blocks access, skip it and continue without failing the metadata pipeline.

**Recommendation:** Keep Amazon Product Advertising API as a separate future integration. Do not make unofficial Amazon scraping a required feature.

## 10. Metadata Caching

Implement `app/books/metadata/cache.py`.

For the initial SQLite-based application:

- Cache results by normalized ISBN.
- Cache title/author searches using a normalized query key.
- Store provider, retrieval timestamp and response data.
- Use configurable TTL.
- Avoid repeated requests for identical searches.
- Handle expired cache entries.
- Do not cache failed or incomplete network responses indefinitely.

## 11. API Endpoint Integration

Add:

`POST /v1/books/metadata/search`

Request schema:

```
{
  "isbn": "9780135957059",
  "title": "Clean Code",
  "author": "Robert C. Martin"
}
```

All fields optional, but at least one meaningful search parameter is required.

Protect the endpoint using the existing JWT authentication dependency.

Suggested response:

```
{
  "status": "success",
  "query": {
    "isbn": "9780135957059",
    "title": "Clean Code",
    "author": "Robert C. Martin"
  },
  "candidates": [
    {
      "title": "Clean Code",
      "subtitle": null,
      "authors": ["Robert C. Martin"],
      "publisher": "Prentice Hall",
      "published_date": "2008",
      "isbn_13": "9780132350884",
      "page_count": 464,
      "categories": ["Computers"],
      "language": "en",
      "cover_url": null,
      "source": "google_books",
      "confidence": 0.92
    }
  ],
  "selected_candidate": null
}
```

The response is illustrative, not a promise that the queried ISBN will return that particular edition. The implementation must return actual provider results.

Endpoint requirements:

- HTTP 200 for successful searches, including no results.
- HTTP 400 for invalid request parameters.
- HTTP 401 for unauthenticated requests.
- HTTP 422 for invalid ISBN format if ISBN is supplied.
- HTTP 503 only when all required external providers fail and no usable cached results exist.
- No provider-specific exceptions should leak to clients.

---

# Part 2 — Tester Agent Instructions

**Role:** QA Engineer / Test Automation Engineer

**Objective:** Build comprehensive unit, integration and endpoint tests to verify provider integrations, metadata normalization, candidate matching, caching, error handling and authentication.

All external API calls must be mocked. The test suite must not require Google Books, Open Library, Amazon or Goodreads to be available.

## 12. Test Project Structure

```
app/
├── books/
│   ├── metadata/
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── conftest.py
│   │       ├── test_google_books.py
│   │       ├── test_open_library.py
│   │       ├── test_normalizer.py
│   │       ├── test_matcher.py
│   │       ├── test_metadata_service.py
│   │       ├── test_scraper.py
│   │       └── test_cache.py
│   └── tests/
│       └── test_metadata_endpoint.py
```

## 13. Google Books API Tests

File: `test_google_books.py`

Test cases:

| Test                 | Expected result               |
| -------------------- | ----------------------------- |
| Search by valid ISBN | Returns matching book         |
| Search by title      | Returns matching candidates   |
| Search by author     | Returns matching candidates   |
| Empty results        | Returns empty list            |
| Missing authors      | Returns empty authors list    |
| Missing ISBN         | Returns metadata without ISBN |
| HTTP 429             | Handles rate limiting         |
| HTTP 500             | Handles server error          |
| Timeout              | Handles request timeout       |
| Invalid JSON         | Handles malformed response    |
| Missing volume ID    | Does not crash                |

Mock `requests` calls or use an HTTP mocking library if the implementation uses HTTPX.

## 14. Open Library API Tests

File: `test_open_library.py`

Test cases:

- ISBN lookup returns valid edition.
- Search by title returns multiple results.
- Search by author returns relevant results.
- Author IDs are resolved correctly.
- Missing author data is handled.
- Missing edition/work relationships are handled.
- Missing cover ID is handled.
- HTTP 404 returns empty result.
- HTTP 429 and 5xx errors are handled.
- Timeout does not crash orchestration.
- Malformed JSON is handled.

## 15. Metadata Normalization Tests

File: `test_normalizer.py`

Test cases:

- Normalize complete Google Books response.
- Normalize complete Open Library response.
- Handle missing optional fields.
- Normalize Unicode characters.
- Remove duplicate authors.
- Remove duplicate categories.
- Normalize whitespace.
- Validate ISBN-10.
- Validate ISBN-13.
- Reject invalid ISBN checksums.
- Normalize ISBNs containing hyphens and spaces.
- Preserve publication date precision.
- Preserve provider source attribution.
- Do not invent missing metadata.

Include ISBN examples:

```
978-0-13-595705-9
9780135957059
0-321-12521-5
0321125215
```

## 16. Candidate Matching Tests

File: `test_matcher.py`

This is one of the most important parts of Phase 3.

Test cases:

1. Exact ISBN-13 match receives highest confidence.
2. Exact ISBN-10 match receives highest confidence.
3. Matching title and author produces a high score.
4. Different punctuation does not prevent matching.
5. Case differences do not prevent matching.
6. Subtitle differences are handled correctly.
7. Multiple authors are matched correctly.
8. Same title but different authors produces a lower score.
9. Same author but different title produces a lower score.
10. Different ISBN editions are not automatically considered identical editions.
11. Missing ISBN does not automatically reject a candidate.
12. Conflicting ISBNs significantly reduce confidence.
13. Duplicate candidates are identified.
14. Ambiguous candidates are not automatically selected.

**Important:** Add tests for books that have multiple editions. Two books with identical titles and authors can have different ISBNs, page counts, publishers and publication dates.

## 17. Metadata Service Integration Tests

File: `test_metadata_service.py`

Test the entire provider orchestration pipeline.

| Scenario                                  | Expected behavior                 |
| ----------------------------------------- | --------------------------------- |
| Google Books finds exact ISBN             | Returns matching candidate        |
| Google Books empty, Open Library succeeds | Returns Open Library result       |
| Both providers find same book             | Deduplicates candidates           |
| Providers return complementary fields     | Merges compatible metadata        |
| Providers return conflicting ISBNs        | Preserves conflict                |
| Both providers return no results          | Returns empty candidates          |
| Google Books timeout                      | Open Library is still queried     |
| Both APIs fail                            | Handles provider failure          |
| Only title supplied                       | Performs title search             |
| Only author supplied                      | Performs author search            |
| Invalid ISBN                              | Rejects request                   |
| Scraping disabled                         | Does not invoke scraper           |
| APIs insufficient, scraping enabled       | Invokes scraper fallback          |
| Multiple ambiguous candidates             | Does not automatically select one |

Test that one provider failure never prevents a successful result from another provider.

## 18. Scraper Tests

File: `test_scraper.py`

All scraping tests must use static HTML fixtures.

Test cases:

- Extract metadata from valid JSON-LD.
- Extract title and author from supported HTML markup.
- Extract ISBN-10 and ISBN-13.
- Handle missing metadata fields.
- Handle malformed JSON-LD.
- Handle empty HTML.
- Handle HTTP 403.
- Handle HTTP 404.
- Handle HTTP 429.
- Handle request timeout.
- Respect disabled scraping configuration.
- Verify that blocked websites do not crash the application.
- Verify that no CAPTCHA or access-control bypass is attempted.

Do not make live requests during tests.

## 19. Cache Tests

File: `test_cache.py`

Test cases:

- Store metadata in cache.
- Retrieve existing cached result.
- Expired entries are ignored.
- Cache keys normalize ISBNs.
- Title/author queries produce stable cache keys.
- Different ISBNs do not collide.
- Failed provider responses are not cached indefinitely.
- Cache can be invalidated.
- Cached data retains source attribution.

Use a temporary SQLite database for isolation.

## 20. Endpoint Tests

File: `test_metadata_endpoint.py`

Use Starlette `TestClient` and the existing authentication fixtures.

Test cases:

| Test                                            | Expected HTTP status    |
| ----------------------------------------------- | ----------------------- |
| Unauthenticated request                         | 401                     |
| Search by ISBN                                  | 200                     |
| Search by title                                 | 200                     |
| Search by author                                | 200                     |
| Missing all search parameters                   | 400                     |
| Invalid ISBN                                    | 422                     |
| No matching book                                | 200                     |
| External providers unavailable                  | 503, if no cache result |
| Google Books unavailable, Open Library succeeds | 200                     |
| Valid request with multiple candidates          | 200                     |

Also verify:

- Response follows the defined Pydantic schema.
- Candidate confidence values are within 0–1.
- No API credentials are exposed.
- Provider-specific internal errors are not exposed.
- The endpoint does not return raw HTML or unprocessed provider payloads.

## 21. Test Fixtures

Create reusable fixtures in `conftest.py`:

- Valid Google Books response.
- Valid Open Library response.
- Multiple editions of the same book.
- Conflicting provider responses.
- Empty provider response.
- Malformed provider response.
- Invalid ISBN.
- Valid authenticated user.
- Authenticated test client.
- Temporary SQLite cache database.
- Mock HTTP responses.
- Static Goodreads/Amazon HTML samples.

Fixtures should be reusable across all metadata tests.

## 22. Coverage & Quality Requirements

The Tester Agent must:

- Run the complete existing test suite from Phases 1 and 2.
- Run all Phase 3 tests independently.
- Verify that existing authentication and OCR behavior remains unchanged.
- Use `pytest` and `pytest-cov`.
- Avoid live API calls.
- Ensure tests are deterministic and repeatable.

Suggested coverage targets:

| Component             | Target coverage |
| --------------------- | --------------- |
| Google Books provider | 90%             |
| Open Library provider | 90%             |
| Normalizer            | 95%             |
| Matcher               | 95%             |
| Metadata service      | 90%             |
| Cache                 | 90%             |
| Scraper               | 85%             |
| API endpoint          | 90%             |

These are targets, not reasons to write artificial tests merely to increase coverage.

Run:

```
pytest app/books/metadata/tests/ -v

pytest app/books/tests/test_metadata_endpoint.py -v

pytest --cov=app.books.metadata \
       --cov-report=term-missing \
       --cov-report=html
```

---

# Part 3 — Definition of Done

Phase 3 is complete only when all the following acceptance criteria are satisfied.

- Google Books integration works with mocked API responses.
- Open Library integration works with mocked API responses.
- Both providers implement the same interface.
- ISBN validation and normalization work correctly.
- Metadata normalization produces consistent output.
- Candidate matching distinguishes editions.
- Duplicate results are handled.
- Conflicting provider metadata is preserved.
- Provider failures are handled independently.
- Scraping fallback works when explicitly enabled.
- Scraping respects website access restrictions.
- SQLite metadata caching works.
- Authenticated metadata search endpoint works.
- All Phase 1, Phase 2 and Phase 3 tests pass.
- No live external API calls are required for automated tests.
- Documentation and environment configuration are updated.

---

# Part 4 — Recommended Implementation Order

Split Phase 3 into six smaller implementation tasks rather than asking the Implementer Agent to build everything at once.

| Step | Task                                | Deliverable                  |
| ---- | ----------------------------------- | ---------------------------- |
| 1    | Provider interfaces and schemas     | Common metadata model        |
| 2    | Google Books integration            | Primary provider             |
| 3    | Open Library integration            | Secondary provider           |
| 4    | Normalization and matching          | Reliable candidate selection |
| 5    | Orchestration, caching and scraping | Complete retrieval pipeline  |
| 6    | FastAPI endpoint integration        | Authenticated API            |

The Tester Agent should work alongside the implementation, writing tests for each completed step rather than waiting until the end.