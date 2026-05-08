"""Document text extraction (PDF text layer, TXT, OCR fallback).

Primary methods:
1. PDF text layer extraction via pdfplumber (fast, accurate for digital PDFs).
2. TXT native read (UTF-8 encoded).
3. OCR fallback via EasyOCR (for scanned/digitized PDFs).

See persondet/models/README_easyocr.md for OCR model documentation.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import easyocr

logger = logging.getLogger(__name__)

# ISO 639-2/B codes used in config queries → PaddleOCR language codes.
QUERY_LANG_TO_PADDLE: dict[str, str] = {
    "bul": "bg",
    "hrv": "hr",
    "cze": "cs",
    "dan": "da",
    "dut": "nl",
    "eng": "en",
    "est": "et",
    "fin": "fi",
    "fre": "fr",
    "ger": "de",
    "gre": "el",
    "hun": "hu",
    "gle": "ga",
    "ita": "it",
    "lav": "lv",
    "lit": "lt",
    "mlt": "mt",
    "pol": "pl",
    "por": "pt",
    "rum": "ro",
    "slo": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
}


class DocumentExtractor:
    """Extract text from documents (PDF, TXT).

    Extraction strategy:
    1. PDF text layer via pdfplumber (fast, most digital PDFs).
    2. TXT natively.
    3. OCR fallback (PaddleOCR) if PDF has insufficient text (< min_text_length).

    Scanned PDFs without text layer will be processed via OCR if enabled,
    otherwise marked as method="ocr_unavailable".
    """

    TEXT_LAYER_MIN_CHARS = 100

    def __init__(self, gpu_id: int = 0, enable_ocr: bool = True) -> None:
        """Initialize DocumentExtractor.

        Args:
            gpu_id: GPU device ID (0-based) for OCR inference. -1 for CPU.
            enable_ocr: If True, use PaddleOCR fallback for scanned PDFs.
        """
        self.gpu_id = gpu_id
        self.enable_ocr = enable_ocr
        self._ocr = None  # Lazy-loaded PaddleOCR instance

    def extract(self, file_path: Path) -> dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._from_txt(file_path)
        if suffix == ".pdf":
            return self._from_pdf(file_path)
        return self._empty("unsupported")

    def _stats(self, text: str, method: str, page_count: int = 0) -> dict[str, Any]:
        return {
            "text": text,
            "method": method,
            "page_count": page_count,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    def _empty(self, method: str) -> dict[str, Any]:
        return self._stats("", method)

    def _from_txt(self, path: Path) -> dict[str, Any]:
        return self._stats(path.read_text(encoding="utf-8", errors="replace"), "native")

    def _from_pdf(self, path: Path) -> dict[str, Any]:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
            page_count = len(page_texts)
        text = "\n\n".join(t for t in page_texts if t)

        # If text layer extraction is sufficient, use it
        if len(text.strip()) >= self.TEXT_LAYER_MIN_CHARS:
            return self._stats(text, "text_layer", page_count=page_count)

        # Otherwise, try OCR fallback if enabled
        if self.enable_ocr:
            logger.debug(
                "%s: text layer insufficient (%d chars); falling back to OCR",
                path.name,
                len(text.strip()),
            )
            return self._from_pdf_ocr(path)
        logger.debug("%s: text layer insufficient; OCR disabled", path.name)
        return self._empty("ocr_unavailable")

    def _get_ocr(self) -> "easyocr.Reader":  # type: ignore
        """Lazy-load EasyOCR reader (expensive initialization).

        Uses locally-stored models from persondet/models/easyocr/ to avoid
        repeated downloads and ensure reproducibility.
        """
        if self._ocr is None:
            try:
                import easyocr
            except ImportError:
                logger.error("EasyOCR not installed. Install via: pip install easyocr")
                raise

            # Set model directory to use local models
            models_dir = Path(__file__).parent / "models" / "easyocr"

            if not models_dir.exists():
                logger.warning(
                    "EasyOCR models not found at %s. Run: "
                    "python persondet/models/download_easyocr_models.py",
                    models_dir,
                )

            self._ocr = easyocr.Reader(
                ["en"],
                model_storage_directory=str(models_dir),
                gpu=(self.gpu_id >= 0),
                verbose=False,
            )

            logger.debug(
                "✓ EasyOCR initialized (GPU: %s, models: %s)", self.gpu_id >= 0, models_dir
            )

        return self._ocr

    def _from_pdf_ocr(self, path: Path) -> dict[str, Any]:
        """Extract text from PDF via OCR (for scanned documents)."""
        try:
            import pdf2image
        except ImportError:
            logger.error("pdf2image not installed. Install via: pip install pdf2image")
            return self._empty("ocr_unavailable")

        try:
            ocr = self._get_ocr()
        except ImportError:
            return self._empty("ocr_unavailable")

        logger.info("Running EasyOCR on %s...", path.name)

        try:
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(str(path), dpi=150)
            page_texts = []

            for page_idx, image in enumerate(images):
                # Run OCR on image using EasyOCR
                # readtext() returns list of (bbox, text, confidence) tuples
                results = ocr.readtext(image)

                # Extract text from results
                if results:
                    page_text = "\n".join([result[1] for result in results])
                else:
                    page_text = ""

                page_texts.append(page_text)
                logger.debug("OCR page %d/%d: %d chars", page_idx + 1, len(images), len(page_text))

            text = "\n\n".join(page_texts)
            return self._stats(text, "ocr_easyocr", page_count=len(images))

        except Exception as e:
            logger.warning("OCR failed for %s: %s", path.name, e)
            return self._empty("ocr_failed")
