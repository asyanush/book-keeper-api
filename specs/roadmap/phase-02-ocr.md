## Phase 2 Specification: Image Recognition & Text Extraction (OCR)

### Overview
Phase 2 focuses on building an endpoint to upload book cover images and process them using a multi-engine OCR strategy. The goal is to extract book title, author, and ISBN metadata reliably using both local open-source tools and cloud fallback capabilities.

---

### 🤖 Implementor Agent Instructions

**Role:** Backend Engineer  
**Objective:** Implement the `POST /v1/books/upload-cover` endpoint, image preprocessing pipeline, and dual-engine OCR service (Tesseract + Google Cloud Vision fallback).

#### Key Responsibilities & Deliverables

1. **Dependency & Configuration Management**
   * Add required dependencies: `python-multipart`, `pytesseract`, `Pillow`, `opencv-python-headless`, `google-cloud-vision`.
   * Update `app/core/config.py` with settings:
     * `OCR_STRATEGY`: Options `TESSERACT`, `VISION`, or `HYBRID` (default: `HYBRID`).
     * `MAX_UPLOAD_SIZE_MB`: Default `10`.
     * `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account key.

2. **Image Preprocessing Pipeline (`app/books/ocr/image_processing.py`)**
   * Implement `preprocess_image(image_bytes: bytes) -> bytes`:
     * Convert binary image to grayscale using OpenCV.
     * Apply adaptive thresholding/binarization to enhance contrast between text and background.
     * Apply slight Gaussian blur to remove digital noise and artifacts.
     * Output cleaned image bytes suitable for OCR engine consumption.

3. **OCR Engine Service (`app/books/ocr/ocr_service.py`)**
   * **Tesseract Module:** `extract_text_tesseract(image_bytes: bytes) -> str`
     * Process preprocessed image via `pytesseract.image_to_string()`.
   * **Vision API Module:** `extract_text_vision(image_bytes: bytes) -> tuple[str, list[str]]`
     * Execute text detection and barcode parsing via `google-cloud-vision`.
   * **Hybrid Orchestrator:** `extract_book_text(image_bytes: bytes) -> dict`
     * Preprocess input image bytes.
     * Execute primary engine based on `OCR_STRATEGY`.
     * Execute fallback engine if primary engine returns empty string or text under word count threshold.
     * Parse ISBN patterns (ISBN-10 and ISBN-13) using regular expressions.
     * Return structured payload:
       ```json
       {
         "raw_text": "Cleaned extracted text content...",
         "detected_isbns": ["9780135957059"],
         "engine_used": "Tesseract (Fallback: None)"
       }
       ```

4. **Upload Router (`app/books/router.py`)**
   * Create endpoint `POST /v1/books/upload-cover`.
   * Secure endpoint with `authenticated_user: AuthenticatedUser = Depends()`.
   * Validate uploaded file payload:
     * File type check (`image/jpeg`, `image/png`, `image/webp`).
     * File size check (max 10MB limit).
   * Invoke `extract_book_text` via `app/books/service.py` and return HTTP 200 with extracted metadata.

---

## Phase 2 - Tester Agent Instructions

**Role:** QA & Test Engineer  
**Objective:** Author unit, integration, and endpoint tests covering upload validation, image preprocessing, OCR mocking, and authorization rules.

#### Key Responsibilities & Test Suite

1. **Endpoint Tests (`app/books/tests/test_upload.py`)**
   * `test_upload_cover_unauthenticated_returns_401`: Assert requests without valid Authorization header return HTTP 401.
   * `test_upload_invalid_file_type_returns_400`: Assert uploading non-image formats (e.g. `.pdf`, `.txt`, `.docx`) returns HTTP 400.
   * `test_upload_file_exceeds_max_size_returns_413`: Assert uploading files over 10MB returns HTTP 413 Payload Too Large.
   * `test_upload_valid_image_returns_200`: Mock OCR service response and verify success payload structure (`raw_text`, `detected_isbns`, `engine_used`).

2. **Preprocessing & Service Unit Tests (`app/books/tests/test_ocr_service.py`)**
   * `test_image_preprocessing_pipeline`: Pass valid and corrupted image byte samples to verify OpenCV processing and exception handling.
   * `test_tesseract_fallback_to_vision`: Mock `pytesseract` to return empty string; assert `google-cloud-vision` client is called.
   * `test_isbn_regex_parsing`: Test regex extractor against various formats (`978-0-13-595705-9`, `9780135957059`, `ISBN 0-321-12521-5`).

3. **Fixtures Setup (`app/books/tests/fixtures.py` & feature `conftest.py`)**
   * Create reusable sample image fixture (1x1 PNG or mock JPEG bytes).
   * Mock GCP Vision API client to avoid live API calls during unit test runs.