# Mission

# Mission Statement: Book Keeper

## 1. Project Purpose
**Book Keeper** is a personal book database application designed to organize, catalog, and track a collection of approximately 1000 physical books with minimal manual entry.

## 2. Core Problem
Manually entering metadata (title, author, publication date, ISBN) for books is tedious and error-prone. Search for the particular book is too complex without a quick access to book's list and information. 

## 3. Solution Overview
Users capture a photo of a book's cover or spine using a mobile camera. The system processes the image using Optical Character Recognition (OCR) and vision-based AI to automatically fetch complete metadata from web databases and store it locally.

---

## 4. Key Functional Requirements

* **Image Input:** Capture or upload photos of book covers, spines, or barcode/ISBN areas via mobile interface.
* **Automated Recognition:** Extract text and identify the book using AI vision/OCR processing.
* **Metadata Fetching:** Search free online book APIs (e.g., Open Library, Google Books API) to retrieve:
  * Book Title
  * Author(s)
  * Publication Date
  * ISBN-10 / ISBN-13 or ISSN
  * Cover Image URL
  * Publisher, Page Count, Genres/Categories, and Synopsis (when available)
* **Data Management:**
  * Display extracted data for user preview and confirmation/editing before saving.
  * Maintain a searchable, filterable catalog of all books.
  * Support custom tags, reading status (e.g., Unread, Reading, Finished), and personal notes/ratings.

---

## 5. Non-Functional Requirements

* **Mobile-First Design:** Optimized interface for smartphones to ensure rapid photo capture workflows.
* **High Accuracy:** Accurate ISBN matching or fallback to title/author search when barcode reading fails.
* **Performance:** Processing and fetching information should take under 3 seconds per book.