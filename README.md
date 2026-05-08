# DARDcollect: DETECTOR Archive Data Collector

A GPU-accelerated multi-modal pipeline for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org). Originally developed for the [DETECTOR project](https://detector-project.eu/), it downloads videos, images, audio, and documents organised by language; extracts person detections and pose keypoints; transcribes speech; extracts text from PDFs and plain-text files; and produces 616×616 face crops with rich `.json` sidecars — bounding boxes, keypoints, quality scores, transcriptions, and full provenance — with [FAIR](https://www.go-fair.org/fair-principles/) metadata throughout.

## Key Features

*   **Ten decoupled stages** — download, person detection, face crop extraction, quality filtering, quality annotation, video transcription, audio transcription, document text extraction, frame extraction — each resumable and independently re-runnable.
*   **Pose-based filtering** — face visibility, size, frontal orientation, and duplicate suppression all derived from CIGPose 133-keypoint poses; robust to the grain and low resolution of pre-1960 film.
*   **GPU accelerated** — YOLOX, CIGPose, and Whisper run via ONNX on CUDA 12; CPU-only mode activates automatically.
*   **FAIR + EU AI Act** — every artifact gets a UUID and full provenance chain; every model and rule-based algorithm is documented per Annex IV.

---

## Quick Setup

**Complete setup guide:** [docs/0-QUICKSTART.md](docs/0-QUICKSTART.md) (5 minutes)

```bash
git clone https://github.com/Vicomtech/dardcollect.git && cd dardcollect
python -m venv .venv && source .venv/bin/activate  # Linux/macOS; .venv\Scripts\activate on Windows
pip install -e .

# 1. Download (all active media types run concurrently)
python scripts/download_media_from_archive.py

# 2. Video pipeline
python scripts/extract_person_clips_from_videos.py
python scripts/extract_face_crops_from_videos.py
python scripts/transcribe_video_clips.py

# 2. Image pipeline (parallel to video)
python scripts/extract_persons_from_images.py
python scripts/extract_face_crops_from_images.py

# 2. Audio pipeline (parallel)
python scripts/transcribe_audio_files.py

# 2. Document pipeline (parallel)
python scripts/extract_text_from_doc.py

# 3. Quality filtering & annotation (video + image crops converge here)
python scripts/filter_face_crops_by_quality.py
python scripts/annotate_face_quality.py

# Optional: extract PNG frames from crops
python scripts/extract_frames_from_videos.py
```

> **VS Code users:** all pipeline stages are available as launch configurations in the Run and Debug panel (`.vscode/launch.json`).

---

## Pipeline

Four modality tracks run in parallel after download. Video and image tracks converge at quality filtering:

```
                        ┌─ person clips ── face crops ─┐
Videos  ─── download ───┤                              ├── filter ── annotate
                        └─ transcriptions              │
                                                       │
Images  ─── download ─── detections ──── face crops ───┘

Audio   ─── download ─── transcriptions

Documents── download ─── extracted text
```

For the full workflow — script-by-script execution, file relationships, UUID provenance chain, output formats, and configuration reference — see [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md) and [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md).

---

## Output Structure

```
DARD/
├── dataset_provenance.json               # Provenance record (updated after each stage)
├── archive_org_public_domain/            # Downloaded source files
│   ├── videos/eng/, videos/spa/, ...     # Language-organised video downloads
│   ├── images/                           # Image downloads
│   ├── audio/eng/, audio/spa/, ...       # Language-organised audio downloads
│   ├── texts/ger/, texts/fra/, ...       # Language-organised text downloads
│   └── dataset.csv                       # Unified metadata (one row per file)
├── traceability/                         # 10 CSV logs (one per stage)
├── extracted_person_clips/               # Person clip videos + JSON sidecars
├── extracted_image_detections/           # Per-image detection JSON
├── extracted_face_crops/                 # 616×616 OFIQ-aligned crops + quality JSON
├── filtered_face_crops/                  # Quality-filtered subset
├── extracted_frames/                     # Optional PNG frames
├── transcriptions/                       # Whisper transcription sidecars
└── preprocessed_documents/              # Extracted text + annotation JSON
```

Every artifact is linked to its source via UUID: Archive.org ID → Download → Clip → Crop → Quality scores. See [docs/2-LOGGING.md](docs/2-LOGGING.md) for CSV schemas and traceability queries, and [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) for sidecar JSON formats.

---

## Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[docs/0-QUICKSTART.md](docs/0-QUICKSTART.md)** | 5-minute installation & execution walkthrough | Everyone (start here) |
| **[docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md)** | System overview, pipeline diagrams, modality workflows, detection models | Technical users |
| **[docs/2-LOGGING.md](docs/2-LOGGING.md)** | CSV schemas, FAIR compliance, provenance chains, querying examples | Data users |
| **[docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md)** | Annotation formats, OFIQ quality dimensions, sidecar JSON structure | ML practitioners |
| **[docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md)** | GPU setup, configuration reference, detailed pipeline walkthrough, development workflow | Developers |
| **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** | Code style, pre-commit hooks, AI system documentation, contributing guidelines | Contributors |

---

## AI Systems (EU AI Act Annex IV)

Each automated component is documented as an AI system per Annex IV, regardless of whether it uses a learned model or a rule-based algorithm. Face quality measures follow [ISO/IEC 29794-5](https://www.iso.org/standard/81694.html) via the [OFIQ reference implementation](https://github.com/BSI-OFIQ/OFIQ-Project).

| Task | System | Type | Implementation | Documentation |
| :--- | :----- | :--- | :------------- | :------------ |
| **Detection** | YOLOX-Tiny (HumanArt) | Neural network (ONNX) | `persondet/detector.py` | [Model card](persondet/models/README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md) |
| **Tracking** | OC-SORT | Algorithm (model-free) | `persondet/tracker.py` | [System card](persondet/models/README_ocsort.md) |
| **Pose estimation** | CIGPose Wholebody (COCO 133) | Neural network (ONNX) | `persondet/poser.py` | [Model card](persondet/models/README_cigpose-m_coco-wholebody_256x192.md) |
| **Scene change detection** | Luminance histogram + bbox area | Algorithm (rule-based) | `scripts/extract_person_clips_from_videos.py` | [System card](persondet/models/README_scene_change_detector.md) |
| **Clip segmentation** | Face/duration/frontal rules | Algorithm (rule-based) | `scripts/extract_person_clips_from_videos.py` | [System card](persondet/models/README_clip_segmentation.md) |
| **Face quality — unified score** | MagFace IResNet50 (ISO/IEC 29794-5) | Neural network (ONNX) | `scripts/filter_face_crops_by_quality.py`, `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_magface_iresnet50_norm.md) |
| **Face quality — sharpness** | Face sharpness random forest (OFIQ `Sharpness`) | Algorithm (random forest) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_sharpness_rtree.md) |
| **Face quality — compression** | SSIM CNN (OFIQ `CompressionArtifacts`) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_ssim_248_model.md) |
| **Face quality — expression neutrality** | HSEmotion EfficientNet + AdaBoost (OFIQ `ExpressionNeutrality`) | Neural network + algorithm | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_expression_neutrality.md) |
| **Face quality — head coverings / occlusion** | BiSeNet face parsing (OFIQ `NoHeadCoverings` / `FaceOcclusionPrevention`) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_bisenet_400.md) |
| **Face quality — face occlusion segmentation** | Face occlusion segmentation CNN | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_occlusion_segmentation_ort.md) |
| **Face quality — head pose** | MobileNetV1 3DDFAV2 (OFIQ `HeadPose`) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_mb1_120x120.md) |
| **Audio transcription** | Whisper-Small | Neural network (PyTorch) | `scripts/transcribe_video_clips.py`, `scripts/transcribe_audio_files.py` | [Model card](persondet/models/README_openai_whisper_small.md) |
| **Document OCR** | PaddleOCR PP-OCRv4 (det + rec + cls) | Neural network (ONNX) | `scripts/extract_text_from_doc.py` | [Model card](persondet/models/README_paddleocr_ocr.md) |

---

## Contributing

Contributions are welcome. Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
- Development setup and pre-commit hooks
- Code style: [Ruff](https://docs.astral.sh/ruff/) (linting & formatting) + [ty](https://docs.astral.sh/ty/) (type checking)
- PR guidelines — including the requirement to document any new pipeline component as an AI system per EU AI Act Annex IV

---

## License

The **source code** is licensed under the [Apache License 2.0](LICENSE).

The **bundled model weights** in `persondet/models/` carry separate licenses and are **not** covered by the Apache 2.0 license:

| Model | File | License |
| :---- | :--- | :------ |
| YOLOX-Tiny (HumanArt) | `yolox_tiny_...onnx` | Architecture: Apache 2.0 — weights trained on HumanArt data ⚠ verify before commercial use |
| CIGPose Wholebody | `cigpose-m_...onnx` | Architecture: Apache 2.0 — trained on COCO WholeBody ⚠ verify before commercial use |
| Whisper Small | `openai_whisper_small.pt` | MIT |
| MagFace IResNet50 | `magface_iresnet50_norm.onnx` | Apache 2.0 ([MagFace](https://github.com/IrvingMeng/MagFace)); ONNX packaging MIT ([OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project)) |
| HSEmotion EfficientNet-B0/B2 | `enet_b0_...onnx`, `enet_b2_...onnx` | MIT ([HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)) |
| AdaBoost neutrality classifier | `hse_1_2_C_adaboost.yml.gz` | MIT ([OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project)) |
| BiSeNet, occlusion, head pose, sharpness, SSIM | various `.onnx` / `.xml.gz` | MIT ([OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project)) |
| PaddleOCR PP-OCRv4 (3 files) | `ch_PP-OCRv4_*.onnx`, `ch_ppocr_*.onnx` | Apache 2.0 ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)) |

See [NOTICE](NOTICE) for full third-party attributions and dependency licenses (including ⚠ PyMuPDF AGPL-3.0).
