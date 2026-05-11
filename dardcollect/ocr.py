"""Document text extraction with PDF text layer, TXT, and OCR fallback.

Provides `DocumentExtractor` which extracts text from documents using
a three-tier strategy:

1. **PDF text layer** via pdfplumber (fast and accurate for digital PDFs).
2. **TXT native read** (UTF-8 with charset detection).
3. **OCR fallback** via PaddleOCR ONNX + PyMuPDF (for scanned/digitized PDFs).

Pages are rendered in-memory via PyMuPDF for OCR — no temporary files and no
external binaries like poppler are required.

Models live in dardcollect/models/:
    - ch_PP-OCRv4_det_infer.onnx  — text detection (language-agnostic)
    - ch_PP-OCRv4_rec_infer.onnx  — text recognition (Latin + CJK)
    - ch_ppocr_mobile_v2.0_cls_infer.onnx — text direction classifier
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR

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

_MODELS_DIR = Path(__file__).parent / "models"
_DET_MODEL = "ch_PP-OCRv4_det_infer.onnx"
_REC_MODEL = "ch_PP-OCRv4_rec_infer.onnx"  # Latin + CJK; covers EU diacritics
_CLS_MODEL = "ch_ppocr_mobile_v2.0_cls_infer.onnx"


class DocumentExtractor:
    """Extract text from PDF and TXT documents with automatic OCR fallback.

    Extraction strategy:
        1. PDF text layer via pdfplumber (fast, most digital PDFs).
        2. TXT natively with charset detection.
        3. OCR fallback via PaddleOCR ONNX if the text layer yields fewer than
           :attr:`TEXT_LAYER_MIN_CHARS` characters.

    Uses three ONNX models from dardcollect/models/ for OCR:
        - ch_PP-OCRv4_det_infer.onnx  — text detection
        - ch_PP-OCRv4_rec_infer.onnx  — text recognition (Latin + CJK)
        - ch_ppocr_mobile_v2.0_cls_infer.onnx — text direction classification
    """

    TEXT_LAYER_MIN_CHARS = 100

    def __init__(
        self,
        gpu_id: int = 0,
        enable_ocr: bool = True,
        languages: list[str] | None = None,
    ) -> None:
        """Initialize the document text extractor.

        Args:
            gpu_id: GPU device ID for OCR inference. -1 for CPU.
            enable_ocr: Whether to enable PaddleOCR fallback for scanned PDFs.
            languages: ISO 639-2/B language codes expected in the corpus
                (e.g., ["eng", "fre", "ger"]). Currently informational —
                the recognition model covers Latin + CJK.
        """
        self.gpu_id = gpu_id
        self.enable_ocr = enable_ocr
        self.languages = languages or ["eng"]
        self._ocr: RapidOCR | None = None

    def extract(self, file_path: Path) -> dict[str, Any]:
        """Extract text from a .pdf or .txt file.

        Args:
            file_path: Path to the document file.

        Returns:
            dict: Statistics dict with keys:
                - "text": Extracted text content.
                - "method": Extraction method used ("native", "text_layer",
                  "ocr_paddleocr", "ocr_failed", or "unsupported").
                - "page_count": Number of pages (PDF only).
                - "word_count": Number of whitespace-separated words.
                - "char_count": Total character count.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._from_txt(file_path)
        if suffix == ".pdf":
            return self._from_pdf(file_path)
        return self._empty("unsupported")

    def _stats(self, text: str, method: str, page_count: int = 0) -> dict[str, Any]:
        """Build a result statistics dictionary.

        Args:
            text: Extracted text content.
            method: Extraction method name.
            page_count: Number of pages processed (default: 0).

        Returns:
            dict: Statistics with text, method, page_count, word_count, char_count.
        """
        return {
            "text": text,
            "method": method,
            "page_count": page_count,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    def _empty(self, method: str) -> dict[str, Any]:
        """Return an empty result with the given method label.

        Args:
            method: Method name for the empty result.

        Returns:
            dict: Empty statistics dict with zero counts.
        """
        return self._stats("", method)

    def _from_txt(self, path: Path) -> dict[str, Any]:
        """Extract text from a plain text file with charset detection.

        Args:
            path: Path to the .txt file.

        Returns:
            dict: Statistics dict with extracted text and method="native".
        """
        from charset_normalizer import from_path

        result = from_path(path).best()
        text = (
            str(result)
            if result is not None
            else path.read_text(encoding="utf-8", errors="replace")
        )
        return self._stats(text, "native")

    def _from_pdf(self, path: Path) -> dict[str, Any]:
        """Extract text from a PDF, falling back to OCR if the text layer is insufficient.

        Args:
            path: Path to the .pdf file.

        Returns:
            dict: Statistics dict with text and method label.
        """
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
            page_count = len(page_texts)
        text = "\n\n".join(t for t in page_texts if t)

        if len(text.strip()) >= self.TEXT_LAYER_MIN_CHARS:
            return self._stats(text, "text_layer", page_count=page_count)

        if self.enable_ocr:
            logger.debug(
                "%s: text layer insufficient (%d chars); falling back to OCR",
                path.name,
                len(text.strip()),
            )
            return self._from_pdf_ocr(path)
        logger.debug("%s: text layer insufficient; OCR disabled", path.name)
        return self._empty("ocr_unavailable")

    def _get_ocr(self) -> "RapidOCR":
        """Lazy-load the PaddleOCR ONNX engine from dardcollect/models/.

        The engine is created on first use and cached for subsequent calls.

        Returns:
            RapidOCR: Initialized PaddleOCR ONNX engine.

        Raises:
            ImportError: If rapidocr-onnxruntime is not installed.
            FileNotFoundError: If any required ONNX model is missing.
        """
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise ImportError(
                    "rapidocr-onnxruntime not installed. Run: pip install rapidocr-onnxruntime"
                ) from exc

            for model_path in (_MODELS_DIR / _DET_MODEL, _MODELS_DIR / _REC_MODEL):
                if not model_path.exists():
                    raise FileNotFoundError(f"PaddleOCR ONNX model not found: {model_path}")

            self._ocr = RapidOCR(
                det_model_path=str(_MODELS_DIR / _DET_MODEL),
                rec_model_path=str(_MODELS_DIR / _REC_MODEL),
                cls_model_path=str(_MODELS_DIR / _CLS_MODEL),
            )
            logger.debug("PaddleOCR ONNX initialized (det: %s, rec: %s)", _DET_MODEL, _REC_MODEL)
        return self._ocr

    def _from_pdf_ocr(self, path: Path) -> dict[str, Any]:
        """Extract text from a scanned PDF using PaddleOCR ONNX.

        Renders pages to 150 DPI RGB bitmaps in memory using PyMuPDF,
        then runs OCR on each page. No temporary files are created.

        Args:
            path: Path to the scanned .pdf file.

        Returns:
            dict: Statistics dict with text and method="ocr_paddleocr" or
                method="ocr_unavailable"/"ocr_failed" on error.
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            logger.error("pymupdf not installed. Run: pip install pymupdf")
            return self._empty("ocr_unavailable")

        try:
            ocr = self._get_ocr()
        except (ImportError, FileNotFoundError) as exc:
            logger.error("PaddleOCR unavailable: %s", exc)
            return self._empty("ocr_unavailable")

        logger.info("Running PaddleOCR on %s...", path.name)

        try:
            doc = fitz.open(str(path))
            # 150 DPI: scale factor relative to PDF's default 72 DPI
            scale = 150 / 72
            mat = fitz.Matrix(scale, scale)
            page_texts: list[str] = []

            page_count = len(doc)
            for page_idx in range(page_count):
                page: Any = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                result, _ = ocr(img_array)
                page_text = "\n".join(item[1] for item in result) if result else ""
                page_texts.append(page_text)
                logger.debug("OCR page %d/%d: %d chars", page_idx + 1, page_count, len(page_text))

            text = "\n\n".join(page_texts)
            return self._stats(text, "ocr_paddleocr", page_count=page_count)

        except Exception as exc:
            logger.warning("OCR failed for %s: %s", path.name, exc)
            return self._empty("ocr_failed")
