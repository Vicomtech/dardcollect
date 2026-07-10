"""Document text extraction with PDF text layer, TXT, and OCR fallback.

Provides `DocumentExtractor` which extracts text from documents using
a three-tier strategy:

1. **PDF text layer** via pdfplumber (fast and accurate for digital PDFs).
2. **TXT native read** (UTF-8 with charset detection).
3. **OCR fallback** via PaddleOCR ONNX + PyMuPDF (for scanned/digitized PDFs).

Pages are rendered in-memory via PyMuPDF for OCR — no temporary files and no
external binaries like poppler are required.

Models live in dardcollect/models/ (PP-OCRv5):
    - ch_PP-OCRv5_det_server.onnx             — text detection (script-agnostic)
    - ch_PP-LCNet_x1_0_textline_ori_cls_server — text direction classifier
    - latin_PP-OCRv5_rec_mobile.onnx          — recognition: Latin script (default, 22/24 EU langs)
    - cyrillic_PP-OCRv5_rec_mobile.onnx       — recognition: Cyrillic (Bulgarian etc.)
    - el_PP-OCRv5_rec_mobile.onnx             — recognition: Greek
    Selected automatically from the `languages` parameter passed to DocumentExtractor.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rapidocr import RapidOCR

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

# ISO 639-2/B codes whose script requires a non-Latin rec model.
_CYRILLIC_LANGS: frozenset[str] = frozenset({"bul", "rus", "ukr", "srp", "mkd", "bel"})
_GREEK_LANGS: frozenset[str] = frozenset({"gre", "ell"})

_MODELS_DIR = Path(__file__).parent / "models"

# Detection + classification models (script-agnostic)
_DET_MODEL = "ch_PP-OCRv5_det_server.onnx"
_CLS_MODEL = "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx"

# Recognition models — one per script family
_REC_MODELS: dict[str, str] = {
    "latin": "latin_PP-OCRv5_rec_mobile.onnx",
    "cyrillic": "cyrillic_PP-OCRv5_rec_mobile.onnx",
    "greek": "el_PP-OCRv5_rec_mobile.onnx",
}
# Charset dict files extracted from each rec model.
# TRT engines discard ONNX-embedded metadata — the dict must be supplied explicitly
# or RapidOCR defaults to the Chinese dict, silently corrupting all output.
_REC_DICTS: dict[str, str] = {
    "latin": "latin_PP-OCRv5_rec_dict.txt",
    "cyrillic": "cyrillic_PP-OCRv5_rec_dict.txt",
    "greek": "el_PP-OCRv5_rec_dict.txt",
}


def _lang_to_script(languages: list[str]) -> str:
    """Map ISO 639-2/B language codes to OCR script family.

    Returns 'cyrillic', 'greek', or 'latin' (default).
    Uses the first language in the list that maps to a non-Latin script.
    """
    for lang in languages:
        if lang.lower() in _CYRILLIC_LANGS:
            return "cyrillic"
        if lang.lower() in _GREEK_LANGS:
            return "greek"
    return "latin"


def extract_rec_dict(model_path: Path, dict_path: Path) -> int:
    """Extract a rec model's embedded character dict to a sidecar .txt file.

    PP-OCRv5 rec ONNX models embed their charset in the `character` metadata
    field. TensorRT engines discard this metadata, so the dict must be supplied
    via `rec_keys_path`. This writes that dict file from the model.

    Args:
        model_path: Path to the rec ONNX model.
        dict_path: Output path for the dict .txt file.

    Returns:
        int: Number of characters written.

    Raises:
        FileNotFoundError: If the model does not exist.
        KeyError: If the model has no embedded `character` metadata.
    """
    import onnx

    if not model_path.exists():
        raise FileNotFoundError(f"Rec model not found: {model_path}")

    model = onnx.load(str(model_path))
    meta = {x.key: x.value for x in model.metadata_props}
    if "character" not in meta:
        raise KeyError(f"No embedded 'character' metadata in {model_path.name}")

    charset = meta["character"]
    dict_path.write_text(charset, encoding="utf-8")
    return len(charset.splitlines())


def setup_rec_dicts(overwrite: bool = False) -> dict[str, int]:
    """Regenerate the charset dict .txt for every configured rec model.

    Iterates `_REC_MODELS` / `_REC_DICTS` and extracts each model's embedded
    charset to its paired dict file in dardcollect/models/.

    Args:
        overwrite: If False, skip dicts that already exist.

    Returns:
        dict: script → number of chars written (or -1 if skipped, -2 if model missing).
    """
    results: dict[str, int] = {}
    for script, model_name in _REC_MODELS.items():
        model_path = _MODELS_DIR / model_name
        dict_path = _MODELS_DIR / _REC_DICTS[script]
        if dict_path.exists() and not overwrite:
            results[script] = -1
            continue
        if not model_path.exists():
            results[script] = -2
            continue
        results[script] = extract_rec_dict(model_path, dict_path)
    return results


class DocumentExtractor:
    """Extract text from PDF and TXT documents with automatic OCR fallback.

    Extraction strategy:
        1. PDF text layer via pdfplumber (fast, most digital PDFs).
        2. TXT natively with charset detection.
        3. OCR fallback via PaddleOCR ONNX if the text layer yields fewer than
           :attr:`TEXT_LAYER_MIN_CHARS` characters.

    Uses PP-OCRv5 ONNX models from dardcollect/models/ for OCR.
    Recognition model is selected per-script from the `languages` parameter:
    Latin (default), Cyrillic (bul/rus/ukr), or Greek (gre/ell).
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
                (e.g., ["eng", "fre", "ger"]). Selects the recognition model:
                Latin script (default), Cyrillic (bul/rus/ukr), or Greek (gre/ell).
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
        """Lazy-load the PaddleOCR engine from dardcollect/models/.

        Provider priority: TensorRT → CUDA (onnxruntime) → CPU (onnxruntime).
        Falls back gracefully if TRT or CUDA are unavailable.

        Returns:
            RapidOCR: Initialized PaddleOCR engine.

        Raises:
            ImportError: If rapidocr is not installed.
            FileNotFoundError: If any required ONNX model is missing.
        """
        if self._ocr is not None:
            return self._ocr

        try:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import EngineType
        except ImportError as exc:
            raise ImportError("rapidocr not installed. Run: pip install rapidocr") from exc

        script = _lang_to_script(self.languages)
        rec_model = _MODELS_DIR / _REC_MODELS[script]
        rec_dict = _MODELS_DIR / _REC_DICTS[script]

        # Verify every model RapidOCR could otherwise auto-download.
        for required in (_MODELS_DIR / _DET_MODEL, rec_model, _MODELS_DIR / _CLS_MODEL):
            if not required.exists():
                raise FileNotFoundError(
                    f"PaddleOCR model not found: {required}\n"
                    f"Download it manually — automatic downloads are disabled."
                )

        # The rec dict is critical under TensorRT: the engine loses its embedded
        # charset, and a missing dict makes RapidOCR fetch the default Chinese dict,
        # silently corrupting all output. If absent, regenerate it from the model
        # (offline, no network) rather than failing.
        if not rec_dict.exists():
            logger.info("Rec dict missing — extracting from %s", rec_model.name)
            extract_rec_dict(rec_model, rec_dict)

        logger.info("PaddleOCR rec model: %s (script: %s)", rec_model.name, script)

        from rapidocr.utils.typings import OCRVersion

        base_params = {
            # Explicit paths for all models — RapidOCR only downloads when model_path is None,
            # so supplying all three prevents any network calls.
            "Global.model_root_dir": str(_MODELS_DIR),
            # All model_paths set explicitly — RapidOCR downloads only when a
            # model_path is None, so this prevents any network fetch (det/rec/cls).
            "Det.model_path": str(_MODELS_DIR / _DET_MODEL),
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_path": str(rec_model),
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Cls.model_path": str(_MODELS_DIR / _CLS_MODEL),
            "Cls.ocr_version": OCRVersion.PPOCRV5,
        }
        gpu_id = max(self.gpu_id, 0)
        use_gpu = self.gpu_id >= 0

        # TRT → CUDA → CPU fallback
        if use_gpu:
            trt_cache = str(Path.cwd() / ".cache" / "trt_engines" / "rapidocr")
            try:
                logger.info(
                    "⚠️  TensorRT is enabled — processing may pause while "
                    "compiling GPU engines on first use"
                )
                self._ocr = RapidOCR(
                    params={
                        **base_params,
                        "Det.engine_type": EngineType.TENSORRT,
                        "Rec.engine_type": EngineType.TENSORRT,
                        "Cls.engine_type": EngineType.TENSORRT,
                        # TRT engines discard embedded ONNX metadata — supply charset
                        # explicitly to prevent RapidOCR defaulting to the Chinese dict
                        # (silent corruption)
                        "Rec.rec_keys_path": str(rec_dict),
                        "EngineConfig.tensorrt.device_id": gpu_id,
                        # FP32: benchmarked FP16 det recall at ~1/4 of FP32 (SVT, IoU 0.1-0.5) —
                        # FP16 silently drops text regions on degraded scans. Rec is
                        # precision-insensitive, but RapidOCR precision is global, so FP32 wins.
                        #
                        # MIXED-PRECISION OPTION (not implemented, ~20% speedup):
                        # Two RapidOCR instances: FP32 for det+cls, FP16 for rec.
                        # API: det_out = ocr_fp32.text_det(img)
                        #      crops = ocr_fp32.crop_text_regions(img, det_out.boxes)
                        #      cls_out = ocr_fp32.text_cls(crops)
                        #      rec_out = ocr_fp16.text_rec(TextRecInput(img=cls_out.img_list))
                        # Tradeoffs: 2x GPU memory, 4 calls/page vs 1, manual result assembly.
                        "EngineConfig.tensorrt.use_fp16": False,
                        "EngineConfig.tensorrt.cache_dir": trt_cache,
                        # det profile pinned explicitly so it can't silently change if
                        # RapidOCR's defaults drift. Values verified against current
                        # defaults (engine_builder.py): min 32 (the /32 DB floor),
                        # opt 736 (PP-OCR det's canonical resolution), max 2048. The full
                        # ocr() pipeline downscales inputs to Global.max_side_len (2000)
                        # before det, so 2048 suffices. CAUTION: max_shape must stay
                        # >= max_side_len (rounded up to a /32 multiple); if max_side_len
                        # is raised above ~2016, raise max_shape too AND delete the engine
                        # cache — profile changes are NOT encoded in the .engine filename,
                        # so a stale engine is reused.
                        "EngineConfig.tensorrt.det_profile.min_shape": [1, 3, 32, 32],
                        "EngineConfig.tensorrt.det_profile.opt_shape": [1, 3, 736, 736],
                        "EngineConfig.tensorrt.det_profile.max_shape": [1, 3, 2048, 2048],
                        # v5 cls_server needs height 80 (not default 48) — otherwise TRT build fails
                        "EngineConfig.tensorrt.cls_profile.min_shape": [1, 3, 80, 160],
                        "EngineConfig.tensorrt.cls_profile.opt_shape": [6, 3, 80, 160],
                        "EngineConfig.tensorrt.cls_profile.max_shape": [6, 3, 80, 160],
                        # rec crops from historical scanned docs exceed default 2048px max
                        # measured on corpus: median=320, p90=679, p99=1303, max=2726
                        "EngineConfig.tensorrt.rec_profile.opt_shape": [6, 3, 48, 512],
                        "EngineConfig.tensorrt.rec_profile.max_shape": [6, 3, 48, 3072],
                    }
                )
                logger.info("PaddleOCR using TensorRT (gpu_id: %d)", gpu_id)
                return self._ocr
            except Exception as e:
                logger.warning("PaddleOCR TRT unavailable — falling back to CUDA: %s", e)

            try:
                self._ocr = RapidOCR(
                    params={
                        **base_params,
                        "EngineConfig.onnxruntime.use_cuda": True,
                        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,
                    }
                )
                logger.info("PaddleOCR using CUDA onnxruntime (gpu_id: %d)", gpu_id)
                return self._ocr
            except Exception as e:
                logger.warning("PaddleOCR CUDA unavailable — falling back to CPU: %s", e)

        self._ocr = RapidOCR(params=base_params)
        logger.info("PaddleOCR using CPU")
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
                result = ocr(img_array)
                # `ocr()` (full pipeline) returns RapidOCROutput with a `.txts`
                # list; use getattr to narrow the union return type for the type
                # checker without changing runtime behavior.
                txts = getattr(result, "txts", None)
                page_text = "\n".join(txts) if result and txts else ""
                page_texts.append(page_text)
                logger.debug("OCR page %d/%d: %d chars", page_idx + 1, page_count, len(page_text))

            text = "\n\n".join(page_texts)
            return self._stats(text, "ocr_paddleocr", page_count=page_count)

        except Exception as exc:
            logger.warning("OCR failed for %s: %s", path.name, exc)
            return self._empty("ocr_failed")
