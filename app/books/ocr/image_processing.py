"""Image preprocessing pipeline for OCR (grayscale + contrast normalization).

OpenCV (``cv2``) is imported lazily inside :func:`preprocess_image` so that
this module remains importable even when the optional ``opencv-python-headless``
package is not installed in the environment.
"""

from app.books.exceptions import ImageProcessingError


def preprocess_image(image_bytes: bytes) -> bytes:
    """Decode, clean, and re-encode a book-cover image for OCR.

    The pipeline:
    1. Decode raw bytes into a numpy array (OpenCV ``imdecode``).
    2. Convert to grayscale (``COLOR_BGR2GRAY``).
    3. Apply CLAHE contrast normalization to even out shadows, glossy
       reflections, and uneven lighting that plague cover photos.
    4. Encode the result as a PNG byte string.

    Parameters
    ----------
    image_bytes:
        Raw bytes of an image (JPEG, PNG, TIFF, etc.).

    Returns
    -------
    bytes
        Processed image bytes encoded as PNG.

    Raises
    ------
    ImageProcessingError
        If the input bytes cannot be decoded as an image, or if the
        processed image cannot be encoded back to PNG.

    Notes
    -----
    The image is intentionally **not** binarized here.  Aggressive adaptive
    thresholding flattens the gradients, artwork, and drop shadows on a book
    cover into hard black/white speckle, which Tesseract then misreads as
    text (producing garbage like ``ARARARIRA`` or ``=<\\``).  Grayscale +
    CLAHE preserves the real letterforms while leaving Tesseract / Vision to
    perform their own (better) internal binarization.
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageProcessingError("Could not decode image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast-limited adaptive histogram equalization: brightens text in
    # shadow while avoiding the noise amplification of global equalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ok, buf = cv2.imencode(".png", enhanced)
    if not ok:
        raise ImageProcessingError("Could not encode processed image")

    return buf.tobytes()