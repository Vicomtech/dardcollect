# DARDcollect — DETECTOR Archive Data Collector

A GPU-accelerated multi-modal pipeline for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org). Originally developed for the [DETECTOR project](https://detector-project.eu/), it downloads videos, images, audio, and documents organised by language; extracts person detections and pose keypoints; transcribes speech; extracts text from PDFs and plain-text files; and produces 616×616 face crops with rich `.json` sidecars — bounding boxes, keypoints, quality scores, transcriptions, and full provenance — with [FAIR](https://www.go-fair.org/fair-principles/) metadata throughout.

**Use it two ways:**
- **Complete pipeline** for bulk processing of historical media collections
- **Modular library** — import individual components (detection, transcription, OCR, face crops, quality scoring) into custom workflows

## Key Features

*   **Ten decoupled stages** — download, person detection, face crop extraction, quality filtering, quality annotation, video transcription, audio transcription, document text extraction, frame extraction — each resumable and independently re-runnable.
*   **Pose-based filtering** — face visibility, size, frontal orientation, and duplicate suppression all derived from CIGPose 133-keypoint poses; robust to the grain and low resolution of pre-1960 film.
*   **Speech transcription** — Whisper-Small transcribes both person-clip audio (video pipeline) and standalone audio files, writing `.transcription.json` sidecars with language detection.
*   **Document text extraction** — extracts text from PDFs (text layer or PaddleOCR fallback for scanned pages) and plain-text files with encoding detection, producing `.text.txt` + `.annotation.json` pairs.
*   **GPU accelerated** — YOLOX, CIGPose, Whisper, and PaddleOCR run via ONNX on CUDA 12; CPU-only mode activates automatically.
*   **FAIR + EU AI Act** — every artifact gets a UUID and full provenance chain; every model and rule-based algorithm is documented per Annex IV.

---

## Installation

### As a Pipeline (Recommended)

For bulk processing of Archive.org media with the complete 10-stage workflow:

```bash
git clone https://github.com/Vicomtech/dardcollect.git && cd dardcollect
uv sync
```

The pipeline processes media through four parallel modality tracks that converge at quality filtering:

```
                        ┌─ person clips ── face crops ─┐
Videos  ─── download ───┤                              ├── filter ── annotate
                        └─ transcriptions              │
                                                       │
Images  ─── download ─── detections ──── face crops ───┘

Audio   ─── download ─── transcriptions

Documents── download ─── extracted text
```

Then follow the step-by-step walkthrough in [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md).

### As a Library (Custom Workflows)

To use individual components in your own Python scripts:

```bash
pip install git+https://github.com/Vicomtech/dardcollect.git
```

Then import and use components:
```python
from dardcollect import PersonDetector, AudioTranscriber, download_item
```

For detailed examples, see [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md).

---

## Output Structure

```
DARD/
├── archive_org_public_domain/            # Downloaded source files
│   ├── videos/eng/, videos/spa/, ...     # Language-organised video downloads (ISO 639-2)
│   ├── videos/und/                       # Videos with no language metadata on Archive.org
│   ├── images/                           # Image downloads (no language subfolder)
│   ├── audio/eng/, audio/spa/, ...       # Language-organised audio downloads (ISO 639-2)
│   ├── audio/und/                        # Audio with no language metadata on Archive.org
│   ├── texts/ger/, texts/fra/, ...       # Language-organised text downloads (ISO 639-2)
│   ├── texts/und/                        # Texts with no language metadata on Archive.org
│   └── downloads.csv                       # Unified metadata (one row per file)
├── extracted_person_clips/               # Person clip videos + JSON sidecars + clips_extraction.csv + transcriptions_extraction.csv
├── extracted_image_detections/           # Per-image detection JSON + image_person_detection.csv
├── face_crops/                           # 616×616 OFIQ-aligned crops + quality JSON + face_crops_extraction.csv
├── filtered_face_crops/                  # Quality-filtered subset + filtered_face_crops.csv + face_quality_annotation.csv
├── extracted_frames/                     # Optional PNG frames + frames_extraction.csv
├── audio_transcriptions/                 # Whisper sidecars for audio files + audio_transcriptions_extraction.csv
└── preprocessed_documents/              # Extracted text + annotation JSON + document_text_extraction.csv
```

Every artifact is linked to its source via UUID: Archive.org ID → Download → Clip → Crop → Quality scores. See [docs/2-LOGGING.md](docs/2-LOGGING.md) for CSV schemas and traceability queries, and [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) for sidecar JSON formats.

---

## AI Systems (EU AI Act Annex IV)

Each automated component is documented as an AI system per Annex IV, regardless of whether it uses a learned model or a rule-based algorithm. Face quality measures follow [ISO/IEC 29794-5](https://www.iso.org/standard/81694.html) via the [OFIQ reference implementation](https://github.com/BSI-OFIQ/OFIQ-Project).

| Task | System | Type | Implementation | Documentation |
| :--- | :----- | :--- | :------------- | :------------ |
| **Detection** | YOLOX-Tiny (HumanArt) | Neural network (ONNX) | `dardcollect/detector.py` | [Model card](dardcollect/models/README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md) |
| **Tracking** | OC-SORT | Algorithm (model-free) | `dardcollect/tracker.py` | [System card](dardcollect/models/README_ocsort.md) |
| **Pose estimation** | CIGPose Wholebody (COCO 133) | Neural network (ONNX) | `dardcollect/poser.py` | [Model card](dardcollect/models/README_cigpose-m_coco-wholebody_256x192.md) |
| **Scene change detection** | Luminance histogram + bbox area | Algorithm (rule-based) | `pipeline/extract_person_clips_from_videos.py` | [System card](dardcollect/models/README_scene_change_detector.md) |
| **Clip segmentation** | Face/duration/frontal rules | Algorithm (rule-based) | `pipeline/extract_person_clips_from_videos.py` | [System card](dardcollect/models/README_clip_segmentation.md) |
| **Face quality — unified score** | MagFace IResNet50 (ISO/IEC 29794-5) | Neural network (ONNX) | `pipeline/filter_face_crops_by_quality.py`, `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_magface_iresnet50_norm.md) |
| **Face quality — sharpness** | Face sharpness random forest (OFIQ `Sharpness`) | Algorithm (random forest) | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_face_sharpness_rtree.md) |
| **Face quality — compression** | SSIM CNN (OFIQ `CompressionArtifacts`) | Neural network (ONNX) | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_ssim_248_model.md) |
| **Face quality — expression neutrality** | HSEmotion EfficientNet + AdaBoost (OFIQ `ExpressionNeutrality`) | Neural network + algorithm | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_expression_neutrality.md) |
| **Face quality — head coverings / occlusion** | BiSeNet face parsing (OFIQ `NoHeadCoverings` / `FaceOcclusionPrevention`) | Neural network (ONNX) | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_bisenet_400.md) |
| **Face quality — face occlusion segmentation** | Face occlusion segmentation CNN | Neural network (ONNX) | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_face_occlusion_segmentation_ort.md) |
| **Face quality — head pose** | MobileNetV1 3DDFAV2 (OFIQ `HeadPose`) | Neural network (ONNX) | `pipeline/annotate_face_quality.py` | [Model card](dardcollect/models/README_mb1_120x120.md) |
| **Audio transcription** | Whisper-Small | Neural network (PyTorch) | `pipeline/transcribe_video_clips.py`, `pipeline/transcribe_audio_files.py` | [Model card](dardcollect/models/README_openai_whisper_small.md) |
| **Document OCR** | PaddleOCR PP-OCRv4 (det + rec + cls) | Neural network (ONNX) | `pipeline/extract_text_from_doc.py` | [Model card](dardcollect/models/README_paddleocr_ocr.md) |

---

## Using Components as a Library

DARDcollect is **primarily a complete pipeline** for processing Archive.org media at scale. However, because its components are decoupled, you can use individual functions in custom workflows if you only need specific capabilities (detection, transcription, OCR, face crops, etc.):

```python
# Example: Custom transcription + face detection workflow
from dardcollect import PersonDetector, AudioTranscriber, download_item
from pathlib import Path

# Download from archive.org with FAIR metadata
result = download_item("example_item_id", dest_dir=Path("media/"))

if result["success"]:
    # Transcribe audio
    transcriber = AudioTranscriber(model_size="small")
    text = transcriber.transcribe_file(result["path"])
    
    # Detect people in video
    detector = PersonDetector(config, model_path="models/yolox_tiny.onnx")
    bboxes, scores = detector.get_detections(frame)
```

**Available components:**
- `PersonDetector`, `PersonTracker`, `PoseEstimator` — Detection & tracking
- `AudioTranscriber` — Whisper speech-to-text
- `DocumentExtractor` — OCR for scanned PDFs
- `process_image()`, `process_video()` — Face crop extraction (OFIQ 616×616)
- `load_models()`, `score_video()` — OFIQ 7-dimensional quality scoring
- `add_fair_metadata()`, `generate_uuid()` — Provenance tracking
- `check_face_visibility()`, `check_frontal_face()` — Face validation
- `extract_frames()` — Video to PNG frames
- `download_item()` — Archive.org downloads

For detailed examples and API reference, see [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md).

---

## Contributing

Contributions are welcome. Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
- Development setup and pre-commit hooks
- Code style: [Ruff](https://docs.astral.sh/ruff/) (linting & formatting) + [ty](https://docs.astral.sh/ty/) (type checking)
- PR guidelines — including the requirement to document any new pipeline component as an AI system per EU AI Act Annex IV

---

## License

The **source code** is licensed under the [Apache License 2.0](LICENSE).

The **bundled model weights** in `dardcollect/models/` carry separate licenses and are **not** covered by the Apache 2.0 license:

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
