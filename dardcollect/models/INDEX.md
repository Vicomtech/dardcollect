# Model Documentation Index

This directory contains pre-trained neural network models and their documentation, structured according to EU AI Act Annex IV.

## Available Models

| Model | File | Purpose | Status | Docs |
|-------|------|---------|--------|------|
| YOLOX (Person Detection) | `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx` | Detects humans in video/images | ✅ Embedded | [README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md](README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md) |
| CIGPose (Pose Estimation) | `cigpose-m_coco-wholebody_256x192.onnx` | Estimates 133 body keypoints | ✅ Embedded | [README_cigpose-m_coco-wholebody_256x192.md](README_cigpose-m_coco-wholebody_256x192.md) |
| BiSeNet (Face Segmentation) | `bisenet_400.onnx` | Segments face regions | ✅ Embedded | [README_bisenet_400.md](README_bisenet_400.md) |
| MagFace (Face Quality) | `magface_iresnet50_norm.onnx` | Unified quality score (ISO/IEC 29794-5) | ✅ Embedded | [README_magface_iresnet50_norm.md](README_magface_iresnet50_norm.md) |
| Whisper (Transcription) | `openai_whisper_small.pt` | Speech-to-text in 98+ languages | ✅ Embedded | [README_openai_whisper_small.md](README_openai_whisper_small.md) |
| OFIQ (Quality Models) | Various `.onnx`, `.xml.gz` | 7-dimensional quality metrics | ✅ Embedded | Individual `README_*.md` per model |
| PaddleOCR Det (Text Detection) | `ch_PP-OCRv4_det_infer.onnx` | Detect text regions in scanned PDF pages | ✅ Embedded | [README_paddleocr_ocr.md](README_paddleocr_ocr.md) |
| PaddleOCR Rec (Text Recognition) | `ch_PP-OCRv4_rec_infer.onnx` | Recognise text (Latin + CJK character set) | ✅ Embedded | [README_paddleocr_ocr.md](README_paddleocr_ocr.md) |
| PaddleOCR Cls (Direction Classifier) | `ch_ppocr_mobile_v2.0_cls_infer.onnx` | Detect 0°/180° text rotation | ✅ Embedded | [README_paddleocr_ocr.md](README_paddleocr_ocr.md) |

## Download & Setup

### Standard Models (Embedded)
Most models are already in this directory. No additional download needed—they load automatically at pipeline startup.

### PaddleOCR ONNX (Optional OCR Support)
The three PaddleOCR ONNX files are already embedded in this directory. Just install
the runtime dependencies:

```bash
pip install rapidocr-onnxruntime pymupdf
```

See [README_paddleocr_ocr.md](README_paddleocr_ocr.md) for full documentation.

## Documentation Structure

Each model has a dedicated `README_<model>.md` file covering:

1. **General Description**
   - Intended purpose & task
   - Provider/authors
   - Version & variants
   - Hardware/software requirements

2. **Development Elements**
   - Training methods & architecture
   - Algorithm specifications
   - Training data composition
   - Known biases & limitations

3. **Integration & Usage**
   - Setup instructions
   - API interface
   - Configuration options
   - Output format

4. **Risk Assessment**
   - Known limitations
   - Mitigation strategies
   - Performance metrics

5. **Provenance & Reproducibility**
   - Version pinning
   - Model versioning
   - Traceability (via CSV logs)

6. **EU AI Act Compliance**
   - Annex IV coverage
   - Transparency statements
   - Human oversight mechanisms

## Quick Reference

### Person Detection → Pose → Face Crops
```
Videos/Images
    ↓
[YOLOX] Detect humans
    ↓
[CIGPose] Estimate keypoints
    ↓
[BiSeNet] Segment faces
    ↓
[Face Geometry] Extract 616×616 OFIQ crops
```

### Quality Annotation
```
Face Crops (Video or Image)
    ↓
[MagFace] Unified quality score
[OFIQ Models] 7 quality dimensions
    ↓
.quality.json Sidecars
```

### Document Processing
```
PDF/TXT Documents
    ↓
[pdfplumber] Text layer extraction (primary)
    ↓ (if < 100 chars)
[PaddleOCR] OCR fallback (optional)
    ↓
.text.txt + .annotation.json
```

### Audio Transcription
```
Video Clips / Audio Files
    ↓
[Whisper] Speech-to-text
    ↓
.transcription.json Sidecars
```

## Configuration

Configure model usage in `config.yaml`:

```yaml
gpu_id: 6                 # GPU device for inference

document_preprocessing:
  enable_ocr: true        # Enable PaddleOCR fallback (default: true)
  min_text_length: 50     # Trigger OCR if text layer < this chars

face_quality_annotation:
  max_frames: 30          # Max frames to sample for quality
  frame_stride: 5         # Sample every Nth frame
```

## Validation & Testing

Run validation to ensure models are loaded correctly:

```bash
python -c "from dardcollect.detector import PersonDetector; d = PersonDetector(); print('✓ Models ready')"
```

Test document extraction with OCR:

```bash
python scripts/extract_text_from_doc.py
```

## References

- **EU AI Act Annex IV:** AI systems requiring documentation
- **ISO/IEC 29794-5 (OFIQ):** Face image quality standard
- **Individual model cards:** See `README_*.md` files for detailed references

## Contact & Updates

For model updates, licensing questions, or issues:
- See individual README files for upstream project links
- Consult `dardcollect/models/README_*.md` for version pinning & dependencies
- Check git history of this directory for past model changes
