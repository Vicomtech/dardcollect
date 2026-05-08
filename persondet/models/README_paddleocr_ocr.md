# PaddleOCR ONNX Models — Document OCR

## 1. General Description

**Intended purpose:** OCR fallback for scanned PDF pages that have no embedded text layer (or fewer than 100 characters extracted by pdfplumber). Called by `persondet/ocr.py → DocumentExtractor._from_pdf_ocr()`.

**Provider:** PaddlePaddle / Baidu, distributed via the [RapidOCR project](https://github.com/RapidAI/RapidOCR) (`rapidocr-onnxruntime` package, bundled models).

**Version:** PP-OCRv4 (detection + recognition), PP-OCRv2 (classifier).

**Files:**

| File | Size | Role |
|------|------|------|
| `ch_PP-OCRv4_det_infer.onnx` | 4.5 MB | Text detection (DB++) |
| `ch_PP-OCRv4_rec_infer.onnx` | 10.4 MB | Text recognition (Latin + CJK) |
| `ch_ppocr_mobile_v2.0_cls_infer.onnx` | 0.6 MB | Text direction classifier |

## 2. Development Elements

**Detection model (`ch_PP-OCRv4_det_infer.onnx`):**
- Architecture: DB++ (Differentiable Binarization), ResNet-18 backbone
- Language-agnostic — detects text bounding boxes regardless of script
- Input: RGB image, arbitrary resolution (resized internally to ≤ 736 px on the short side)
- Output: binary map → polygon bounding boxes

**Recognition model (`ch_PP-OCRv4_rec_infer.onnx`):**
- Architecture: SVTR (Scene Text Recognition Transformer)
- Character set: Latin (including common EU diacritics: é, è, ê, ü, ö, ä, ß, å, etc.) + CJK
- Input: cropped text-line image, 48 × variable width
- Output: character sequence via CTC decoding

**Classifier (`ch_ppocr_mobile_v2.0_cls_infer.onnx`):**
- Detects 0° vs 180° text rotation before feeding lines to the recogniser

**Inference runtime:** ONNX Runtime (via `rapidocr-onnxruntime`). Uses `onnxruntime-gpu` if available.

**Page rendering:** PyMuPDF (`fitz`) renders PDF pages to RGB numpy arrays at 150 DPI — no external binaries (poppler), no temp files.

## 3. Integration & Usage

Models are loaded lazily on first OCR call:

```python
from persondet.ocr import DocumentExtractor

extractor = DocumentExtractor(gpu_id=0, enable_ocr=True)
result = extractor.extract(Path("scanned.pdf"))
# result["method"] == "ocr_paddleocr"
# result["text"]   == extracted text
```

OCR is only triggered when the PDF text layer yields fewer than 100 characters. Digital PDFs with an embedded text layer use `pdfplumber` directly (`method="text_layer"`).

**Configuration (`config.yaml`):**
```yaml
document_preprocessing:
  enable_ocr: true     # set false to disable OCR fallback entirely
  ocr_languages: [eng, fre, ger, ...]   # informational; single rec model used for all
```

## 4. Known Limitations

- The recognition model (`ch_PP-OCRv4_rec_infer.onnx`) is optimised for Chinese + English. Coverage of EU-specific diacritics (e.g. Maltese Ħ/ħ, Irish fada) may be incomplete.
- Greek and Cyrillic (Bulgarian) scripts are not reliably recognised by this model. For Greek/Bulgarian-only corpora, a dedicated language model exported from PaddleOCR via `paddle2onnx` would be needed.
- 150 DPI rendering is a trade-off between accuracy and speed; very small print (< 8pt) may OCR poorly.

## 5. Provenance & Reproducibility

These ONNX files are copied directly from the `rapidocr-onnxruntime` Python package (v1.4.4). They are identical to the models shipped with that package and can be reproduced by:

```python
import shutil, rapidocr_onnxruntime
from pathlib import Path
src = Path(rapidocr_onnxruntime.__file__).parent / "models"
for f in src.glob("*.onnx"):
    shutil.copy2(f, Path("persondet/models") / f.name)
```

Upstream source: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), Apache-2.0 licence.

## 6. EU AI Act Compliance (Annex IV)

- **Intended purpose:** Automated text extraction from historical public-domain documents (1900–1955) for dataset construction. No decisions affecting persons are made from OCR output.
- **Human oversight:** Extracted text is written to `.text.txt` sidecars for human review before any downstream use.
- **Accuracy metric:** Not formally benchmarked on this corpus. Method tag (`ocr_paddleocr`) in annotation JSON enables post-hoc quality filtering.
- **Licence:** Apache-2.0 (PaddleOCR / RapidOCR). Commercial use permitted.
