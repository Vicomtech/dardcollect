"""Document text extraction (PDF text layer, TXT, OCR fallback via PaddleOCR ONNX).

Primary methods:
1. PDF text layer extraction via pdfplumber (fast, accurate for digital PDFs).
2. TXT native read (UTF-8 encoded).
3. OCR fallback via PaddleOCR ONNX + PyMuPDF (for scanned/digitized PDFs).
   Pages are rendered in-memory via PyMuPDF — no temp files, no external binaries.

Models live in persondet/models/ (same directory as all other ONNX models).
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
    """Extract text from documents (PDF, TXT).

    Extraction strategy:
    1. PDF text layer via pdfplumber (fast, most digital PDFs).
    2. TXT natively.
    3. OCR fallback via PaddleOCR ONNX if the text layer yields < 100 chars.

    Uses three ONNX models from persondet/models/:
      ch_PP-OCRv4_det_infer.onnx           — text detection (language-agnostic)
      ch_PP-OCRv4_rec_infer.onnx           — text recognition (Latin + CJK)
      ch_ppocr_mobile_v2.0_cls_infer.onnx  — text direction classifier
    """

    TEXT_LAYER_MIN_CHARS = 100

    def __init__(
        self,
        gpu_id: int = 0,
        enable_ocr: bool = True,
        languages: list[str] | None = None,
    ) -> None:
        """Initialize DocumentExtractor.

        Args:
            gpu_id: GPU device ID for OCR inference. -1 for CPU.
            enable_ocr: Enable PaddleOCR fallback for scanned PDFs.
            languages: ISO 639-2/B language codes expected in the corpus
                (e.g. ["eng", "fre", "ger"]). Used to select the recognition model.
        """
        self.gpu_id = gpu_id
        self.enable_ocr = enable_ocr
        self.languages = languages or ["eng"]
        self._ocr: RapidOCR | None = None

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
        from charset_normalizer import from_path

        result = from_path(path).best()
        text = (
            str(result)
            if result is not None
            else path.read_text(encoding="utf-8", errors="replace")
        )
        return self._stats(text, "native")

    def _from_pdf(self, path: Path) -> dict[str, Any]:
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
        """Lazy-load PaddleOCR ONNX engine from persondet/models/."""
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
        """Extract text from a scanned PDF via PaddleOCR ONNX.

        Uses PyMuPDF to render pages directly to numpy arrays in memory (no
        temp files, no external binary like poppler).
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
