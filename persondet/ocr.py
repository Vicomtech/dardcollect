"""Document text extraction (PDF text layer, TXT). No OCR."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ISO 639-2/B codes used in config queries → Tesseract language codes (reference only).
QUERY_LANG_TO_TESS: dict[str, str] = {
    "bul": "bul",
    "hrv": "hrv",
    "cze": "ces",
    "dan": "dan",
    "dut": "nld",
    "eng": "eng",
    "est": "est",
    "fin": "fin",
    "fre": "fra",
    "ger": "deu",
    "gre": "ell",
    "hun": "hun",
    "gle": "gle",
    "ita": "ita",
    "lav": "lav",
    "lit": "lit",
    "mlt": "mlt",
    "pol": "pol",
    "por": "por",
    "rum": "ron",
    "slo": "slk",
    "slv": "slv",
    "spa": "spa",
    "swe": "swe",
}


class DocumentExtractor:
    """Extract text from documents (PDF, TXT).

    PDF text layer via pdfplumber; TXT natively.
    Scanned-only PDFs produce an empty result with method="ocr_skipped".
    """

    TEXT_LAYER_MIN_CHARS = 100

    def __init__(self) -> None:
        pass

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
        if len(text.strip()) >= self.TEXT_LAYER_MIN_CHARS:
            return self._stats(text, "text_layer", page_count=page_count)
        return self._empty("ocr_skipped")
