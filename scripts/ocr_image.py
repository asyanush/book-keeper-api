"""Run the OCR pipeline against local image files, without the API.

Useful for quickly iterating on preprocessing / engine selection when a cover
photo returns garbage. Reads each image from disk, runs the exact same
``extract_book_text`` pipeline the API uses, and prints the structured result.

Usage (run from the project root so ``app`` is importable)::

    python scripts/ocr_image.py cover.jpg
    python scripts/ocr_image.py a.jpg b.png --strategy VISION
    python scripts/ocr_image.py cover.jpg --save-processed preprocessed.png

Notes
-----
* The OCR engine order comes from ``OCR_STRATEGY`` in ``.env.dev`` unless
  overridden with ``--strategy``.
* Heavy dependencies (``cv2``, ``pytesseract``, ``google.cloud.vision``) are
  imported lazily by the pipeline, matching the API's behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable regardless of the caller's CWD, since
# running ``python scripts/ocr_image.py`` puts ``scripts/`` (not the root)
# at the front of ``sys.path``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text/ISBNs from a book-cover image via the OCR pipeline.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="One or more image files (JPEG/PNG/WebP) to run OCR on.",
    )
    parser.add_argument(
        "--strategy",
        choices=("TESSERACT", "VISION", "HYBRID"),
        default=None,
        help="Override the OCR_STRATEGY from settings for this run.",
    )
    parser.add_argument(
        "--save-processed",
        type=Path,
        default=None,
        help="Write the preprocessed (grayscale + CLAHE) image to this path "
        "for visual inspection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Lazy import so --help works even outside the project environment.
    from app.books.ocr.image_processing import preprocess_image
    from app.books.ocr.ocr_service import extract_book_text
    from app.core.config import settings

    if args.strategy:
        settings.OCR_STRATEGY = args.strategy
        print(f"OCR strategy override: {args.strategy}")

    for image_path in args.images:
        if not image_path.is_file():
            print(f"ERROR: file not found: {image_path}", file=sys.stderr)
            return 1

        image_bytes = image_path.read_bytes()

        if args.save_processed and len(args.images) == 1:
            processed = preprocess_image(image_bytes)
            args.save_processed.write_bytes(processed)
            print(f"Preprocessed image saved to: {args.save_processed}")

        result = extract_book_text(image_bytes)

        print(f"\n=== {image_path} ===")
        print(f"engine_used : {result['engine_used']}")
        print(f"isbns       : {result['detected_isbns'] or '[]'}")
        print("raw_text    :")
        print(result["raw_text"] or "  (empty)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())