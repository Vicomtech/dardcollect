# DETECTOR Archive Data Collector

This repository contains a GPU-accelerated pipeline for building a labelled audiovisual dataset from public-domain archive film. Originally developed for the [DETECTOR project](https://detector-project.eu/), it can be adapted for any task requiring annotated video of real people — speaker recognition, facial analysis, action recognition, and similar.

The pipeline covers the full journey from raw video to annotated dataset: it downloads public-domain films from the [Internet Archive](https://archive.org), detects and tracks persons frame-by-frame, filters clips by face visibility and frontal orientation, extracts face crops, filters them by quality, annotates each crop with a full suite of OFIQ face quality measures, and transcribes speech — producing per-clip `.mp4` files with rich `.json` sidecars containing bounding boxes, pose keypoints, face quality scores, and transcription.

## 🚀 Key Features

*   **End-to-end pipeline**: Six decoupled stages — download, person-clip extraction, face-crop extraction, face quality filtering, face quality annotation, and audio transcription — each resumable and independently re-runnable.
*   **Pose-based face filtering**: Face visibility, minimum size, frontal orientation, and mouth-open detection are all derived from CIGPose wholebody keypoints — robust to the low resolution and grain of pre-1960 film stock where pixel-based face detectors struggle.
*   **Keypoint-based duplicate suppression**: Overlapping tracklets are removed by comparing pose keypoint positions, so a single person never generates two competing tracks — robust even when persons are close together.
*   **Scene-aware segmentation**: Hard cuts reset the tracker and close the current clip, preventing track IDs and clip content from bleeding across shots.
*   **GPU accelerated**: YOLOX and CIGPose run via ONNX/TensorRT on CUDA 12; Whisper transcription runs on the same GPU.
*   **Resumable at every stage**: All scripts checkpoint progress; interrupted runs continue from the last processed frame or clip without data loss.
*   **Provenance record**: Each pipeline stage appends a signed run entry to `DARD/dataset_provenance.json`, capturing dataset origin (Archive.org identifiers and URLs), collection timestamp, model names and SHA-256 checksums, and the full configuration snapshot used — satisfying formal data provenance requirements.
*   **EU AI Act documented**: Every automated component — including rule-based algorithms — is documented as an AI system per Annex IV.

---

## 🛠️ Setup

### Prerequisites

*   **OS**: Linux (tested on Ubuntu) or Windows 10/11.
*   **Python**: **3.12** required.
*   **GPU**: NVIDIA GPU with CUDA 12.x support (Driver 550+ recommended).
*   **Git LFS**: Required to download model files (`.onnx`, `.pt`). Install from [git-lfs.com](https://git-lfs.com) and run `git lfs install` once after installing.

### Installation

1.  **Install `uv`** (Fast package manager):
    ```bash
    pip install uv
    ```

2.  **Create & Activate Environment**:
    ```bash
    uv venv --python 3.12
    ```
    ***Windows:***
    ```bash
    .venv\Scripts\activate
    ```
    ***Linux/Mac:***
    ```bash
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    # This automatically installs the correct pinned NVIDIA libraries (CUDA 12.4 compatible)
    uv pip install -e ".[dev]"
    ```

    > [!NOTE]
    > **Linux Shared Environments**: The setup script automatically finds and preloads the `nvidia-*` pip packages (cuBLAS, cuDNN, TensorRT, etc.). No manual `LD_LIBRARY_PATH` or system-wide CUDA installation is required.

### Code Quality & Pre-commit Hooks

To ensure every commit is properly formatted and type-safe:

1.  **Install pre-commit** (optional but recommended):
    ```bash
    uv pip install pre-commit
    pre-commit install
    ```

2.  **Run checks manually**:
    ```bash
    ruff check .           # Lint
    ruff format .          # Auto-format
    ty check .             # Type checking
    ```

Pre-commit hooks will automatically run [Ruff](https://docs.astral.sh/ruff/) (linter + formatter) and [ty](https://docs.astral.sh/ty/) (type checker) before each commit. For details, see [CONTRIBUTING.md](CONTRIBUTING.md#code-style).

---

## ⚙️ Configuration

All settings are in `config.yaml`, which is fully commented. Key options:

```yaml
# Download settings
output_dir: "DARD/archive_org_public_domain"  # Where to save downloaded videos
max_total_size_gb: 10                          # Stop downloading once limit reached
search_query: "..."                            # Archive.org search query
min_duration_minutes: 20                        # Skip clips shorter than this

person_extraction:
  input_dir: "DARD/archive_org_public_domain"     # ← must match output_dir above
  output_clips_dir: "DARD/extracted_person_clips" # ← must match face_crop_extraction.input_dir
  detection_threshold: 0.4        # Min YOLOX confidence to accept a detection
  tracking_score_threshold: 0.4   # Min IoU match score for OC-SORT
  min_clip_duration_seconds: 2.0  # Discard clips shorter than this
  max_clip_duration_seconds: 60.0 # Split clips longer than this into sub-clips
  require_face_visibility: true   # Discard clips with no visible frontal face frames
  min_face_visible_frames: 15     #   ↳ min total frames with a visible face (ignored if false)
  min_consecutive_face_frames: 5  #   ↳ min unbroken run of face-visible frames (ignored if false)
  min_free_disk_gb: 2.0           # Abort if free disk space drops below this

# Face crop settings
face_crop_extraction:
  output_dir: "DARD/face_crops"  # 616×616 OFIQ crops written flat (no subdirs)
  # Sidecar JSONs include arcface_crop_corners_in_ofiq for downstream MagFace scoring

# Quality filtering settings
face_quality_filtering:
  input_dir: "DARD/face_crops"            # ← must match face_crop_extraction.output_dir
  output_dir: "DARD/filtered_face_crops"  # passing videos moved here
  quality_threshold: 75.0                  # OFIQ score [0–100]; raise for stricter filtering
  frame_sample_interval: 5                 # Assess every Nth frame (1 = all frames)
```

**Important parameters:**

- **`max_total_size_gb`**: Download stops immediately when this limit is reached. Prevents runaway downloads filling your disk. Default: 100 GB.

> **Note:** The paths between sections are linked — `output_dir` (download) feeds `person_extraction.input_dir`, and `person_extraction.output_clips_dir` feeds `face_crop_extraction.input_dir`. Change them together.

---

## 🎞️ Processing Pipeline

```
Internet Archive  →  raw .mp4 files  →  person clips + sidecars  →  face crops  →  filtered face crops  →  quality-annotated crops  →  transcriptions
                     (download)          (extract_person_clips)      (face_crops)    (filter_quality)         (annotate_quality)           (transcribe)
```

> **VS Code users:** all pipeline stages are available as launch configurations in the Run and Debug panel (`.vscode/launch.json`) — no need to type commands manually.

### 1. Download videos
```bash
python scripts/download_videos_from_archive.py
```
Searches the Internet Archive for public-domain films matching the query in `config.yaml` and downloads them to `input_dir`. Skips files already present. Respects `max_total_size_gb`.

> **Safety note**: Downloads use atomic temp files (`.tmp` extension). If interrupted mid-download, the incomplete `.tmp` file is cleaned up automatically on the next run — you won't end up with corrupted video files.

### 2. Extract person clips
```bash
python scripts/extract_person_clips.py
```
Reads videos from `input_dir`. For each video, detects and tracks persons frame-by-frame, filters by face visibility and frontal orientation, and writes accepted clips to `output_clips_dir` as `.mp4` + `.json` pairs. Each `.json` sidecar contains per-frame bounding boxes, pose keypoints, and clip-level statistics.

### 3. Extract face crops
```bash
python scripts/extract_face_crops.py
```
Reads clips from `output_clips_dir`. For each tracked person, produces one **OFIQ-aligned face crop video** (616×616) directly in `face_crop_extraction.output_dir/` — same flat layout as `extracted_person_clips/`.

OFIQ crops use the BSI-OFIQ canonical framing (eyes at y≈272, nose at y≈336 on a 616-tall canvas) — the format expected by all OFIQ quality models: sharpness, expression neutrality, head pose, compression artifacts, and face occlusion.

The sidecar JSON alongside each video includes an **`arcface_crop_corners_in_ofiq`** field: the 4 corners (TL, TR, BR, BL) of the ArcFace 112×112 region in OFIQ pixel coordinates. Because both formats derive from the same fixed canonical landmark positions, this is a constant across all frames and all clips. Downstream scripts (`filter_face_crops_by_quality.py`, `annotate_face_quality.py`) use it to extract 112×112 ArcFace crops from OFIQ frames for MagFace scoring — no separate ArcFace video file is produced. See `persondet/face_geometry.py` (`ARCFACE_CROP_CORNERS_IN_OFIQ`) for the exact coordinates.

Optional — run this if you need quality-annotated face video rather than full-body clips.

### 4. Filter face crops by quality
```bash
python scripts/filter_face_crops_by_quality.py
```
Reads OFIQ crops from `face_quality_filtering.input_dir/`. For each video, scores frames using the **OFIQ unified quality score** ([ISO/IEC 29794-5](https://www.iso.org/standard/81694.html), reference implementation: [BSI-OFIQ/OFIQ-Project](https://github.com/BSI-OFIQ/OFIQ-Project)): the output magnitude of MagFace IResNet50, which measures how confidently a face recognition model can embed a crop. MagFace requires 112×112 ArcFace crops — these are extracted on-the-fly from each OFIQ frame using the constant `arcface_crop_corners_in_ofiq` region from the sidecar JSON. When a clip passes, the OFIQ video and its sidecar are moved to `face_quality_filtering.output_dir/ofiq/`. Optional — run this if you need a quality-controlled subset.

Key config option under `face_quality_filtering`:
- **`quality_threshold`**: MagFace score required to pass [0, 100]. Every frame is scored; a clip passes as soon as one frame meets the threshold.

### 5. Annotate face quality
```bash
python scripts/annotate_face_quality.py <face_crops_dir> [--gpu-id 0] [--frame-stride 5] [--max-frames 30]
```
Reads OFIQ-aligned crops from the given directory (output of step 3, or `filtered_face_crops/` after step 4). For each video, samples frames at the requested stride and runs all OFIQ quality models, then writes a sibling `.quality.json` file next to each `.mp4`. Already-annotated videos are skipped on re-run; pass `--overwrite` to re-score them.

The following measures are computed per video, each stored as `{max, mean, p10, p50, p90}` over the sampled frames:

| Measure | OFIQ component | Calibrated score range |
| :--- | :--- | :--- |
| `unified_score` | `UnifiedQualityScore` (MagFace magnitude) | [0, 100] |
| `sharpness` | `Sharpness` (Laplacian/Sobel RTrees) | [0, 100] |
| `compression_artifacts` | `CompressionArtifacts` (SSIM CNN) | [0, 100] |
| `expression_neutrality` | `ExpressionNeutrality` (HSEmotion EfficientNet-B0/B2 + AdaBoost) | [0, 100] |
| `no_head_coverings` | `NoHeadCoverings` (BiSeNet parsing — hat/cloth fraction) | [0, 100] |
| `face_occlusion_prevention` | `FaceOcclusionPrevention` (FaceOcclusionSegmentation CNN) | [0, 100] |
| `head_pose` | `HeadPose` (MobileNetV1 3DDFAV2) | yaw/pitch/roll degrees + cosine² quality scores |

> **Note:** OFIQ quality measures read from `ofiq/` (616×616), which is the format they were designed for. MagFace (`unified_score`) is the exception — it needs 112×112 ArcFace crops. The script extracts these on-the-fly from each OFIQ frame using `arcface_crop_corners_in_ofiq` from the sidecar JSON (present in all output of `extract_face_crops.py`). If the field is absent (old-format sidecar), `unified_score` is omitted. Scores are directly comparable to those produced by the full OFIQ pipeline.

### 6. Transcribe audio
```bash
python scripts/transcribe_clips.py
```
Reads clips from `output_clips_dir` and transcribes their audio using Whisper-Small, writing a `transcription` field into each `.json` sidecar in-place. Optional — only needed if speech content is relevant to your use case.

### 7. Interactive Viewer
**File**: `viewer/detection_viewer.html`

A local HTML tool to inspect extraction results: browse clips, play video with overlaid bounding boxes and keypoints, review face quality scores, and inspect face crops.

Supports two folder types — drop either into the viewer:

| Folder | What you see |
| :--- | :--- |
| `extracted_person_clips/` | Full-body clips with bounding boxes, keypoints, and ArcFace/OFIQ crop quads overlaid on the source video |
| `face_crops/` or `filtered_face_crops/` | 616×616 OFIQ face crop videos with the ArcFace region (yellow quad) overlaid, plus the quality panel from `.quality.json` if present |

#### Local Usage (Desktop)
1. Open `detection_viewer.html` in a web browser.
2. Click **Select Folder** and choose `extracted_person_clips`, `face_crops`, or `filtered_face_crops`.

#### Remote Usage (VSCode / SSH)
Since web browsers cannot access remote files via "Select Folder", use the **Server Mode**:

1.  **Index the Data** (Run this after extracting new clips):
    ```bash
    cd viewer
    python index_data.py
    ```
    *Creates `data_index.json` and ensures the `data_link` symlink exists.*

2.  **Start Web Server**:
    ```bash
    python -m http.server 8080
    ```

3.  **Access the Viewer**:
    *   **VSCode (Easiest)**: Go to the **Ports** tab (next to Terminal), right-click port `8080`, and select **Open in Browser** (the globe icon). This opens it in your local Chrome/Firefox/Safari.
    *   **SSH Tunnel**: Run `ssh -L 8080:localhost:8080 user@host` on your local machine, then open `http://localhost:8080` in your browser.

---

## 📂 Output Structure

```
DETDF/
├── dataset_provenance.json               # Formal provenance record (updated after each stage)
├── archive_org_public_domain/            # Downloaded source videos
│   └── *.mp4
├── extracted_person_clips/               # Output of extract_person_clips.py
│   ├── VideoTitle_00m15s-01m03s.mp4      # Person clip
│   ├── VideoTitle_00m15s-01m03s.json     # Sidecar: bboxes, keypoints, face_crop_corners_arcface/ofiq, stats
│   └── VideoTitle_progress.json          # Resume checkpoint (internal)
├── face_crops/                           # Output of extract_face_crops.py (616×616 OFIQ crops)
│   ├── VideoTitle_00m15s-01m03s_face_1.mp4
│   ├── VideoTitle_00m15s-01m03s_face_1.json          # crop_format: "ofiq", arcface_crop_corners_in_ofiq, ...
│   ├── VideoTitle_00m15s-01m03s_face_1.quality.json  # written by annotate_face_quality.py
│   └── ...
└── filtered_face_crops/                  # Output of filter_face_crops_by_quality.py (quality-passing crops)
    ├── VideoTitle_00m15s-01m03s_face_1.mp4
    ├── VideoTitle_00m15s-01m03s_face_1.json
    ├── VideoTitle_00m15s-01m03s_face_1.quality.json  # written by annotate_face_quality.py
    └── ...
```

Each `.json` sidecar (in `extracted_person_clips/`) contains:
- Clip metadata: source video (forward-slash path), timestamps, duration, FPS, `track_ids` list
- Per-frame data: bounding boxes, pose keypoints (1 dp), confidence scores for each tracked person, `face_crop_corners_arcface` and `face_crop_corners_ofiq` (4 source-frame corners for each alignment)
- Summary statistics: `face_visible_frames`, `max_consecutive_face_frames`, `mouth_open_frames`
- `transcription`: filled in by `transcribe_clips.py` (empty string until then)

Face crop sidecars (in `face_crops/`) contain: `source_video`, `track_id`, `fps`, `crop_format` (`"ofiq"`), `output_size` (616), frame range, timestamps, `valid_face_frames`, and `arcface_crop_corners_in_ofiq` — the 4 corners (TL, TR, BR, BL) of the ArcFace 112×112 extraction region in OFIQ pixel coordinates. This constant value is the same for every video and enables downstream scripts to extract 112×112 ArcFace crops from OFIQ frames without needing a separate video file.

Quality annotation files (`.quality.json`, written by `annotate_face_quality.py` alongside each video) contain: per-measure statistics (`max`, `mean`, `p10`, `p50`, `p90`) for `unified_score` (MagFace, extracted on-the-fly from OFIQ frames), `sharpness`, `compression_artifacts`, `expression_neutrality`, `no_head_coverings`, `face_occlusion_prevention`, and `head_pose` (yaw/pitch/roll angles + cosine² quality scores).

---

## ⏸️ Interrupting & Resuming

All pipeline scripts support **resumable checkpointing**. You can interrupt processing at any time and restart from where you left off.

### How to interrupt
Press `Ctrl+C` in the terminal where the script is running.

### How to resume
Run the same script again with the same `config.yaml`:
```bash
python scripts/extract_person_clips.py
# Ctrl+C after frame 5000 of a video
python scripts/extract_person_clips.py
# Resumes from frame 5001 of the same video
```

### Prerequisites
- Do **not** delete progress checkpoint files (`.json` files with "progress" in the name)
- Keep input/output paths unchanged between runs

### What gets checkpointed
- **Download**: Which videos have been downloaded (skips existing files)
- **Extract person clips**: Which frames have been processed per video
- **Extract face crops**: Which clips have been processed
- **Filter face crops by quality**: Which videos are already in the output directory (skipped on re-run)
- **Annotate face quality**: Which videos already have a sibling `.quality.json` file in the `ofiq/` directory (skipped on re-run; use `--overwrite` to re-score)
- **Transcribe**: Which clips have been transcribed

---

## 🧠 AI Systems

Each automated component of the pipeline is documented as an AI system in accordance with EU AI Act Annex IV, regardless of whether it uses a learned model or a rule-based algorithm.

Face quality measures follow [ISO/IEC 29794-5](https://www.iso.org/standard/81694.html), using models from the [OFIQ reference implementation](https://github.com/BSI-OFIQ/OFIQ-Project) (Open Source Face Image Quality).

| Task | System | Type | Implementation | Documentation |
| :--- | :----- | :--- | :------------- | :------------ |
| **Detection** | YOLOX-Tiny (HumanArt) | Neural network (ONNX) | `persondet/detector.py` | [Model card](persondet/models/README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md) |
| **Tracking** | OC-SORT | Algorithm (model-free) | `persondet/tracker.py` | [System card](persondet/models/README_ocsort.md) |
| **Pose estimation** | CIGPose Wholebody (COCO 133) | Neural network (ONNX) | `persondet/poser.py` | [Model card](persondet/models/README_cigpose-m_coco-wholebody_256x192.md) |
| **Scene change detection** | Luminance histogram + bbox area | Algorithm (rule-based) | `scripts/extract_person_clips.py` | [System card](persondet/models/README_scene_change_detector.md) |
| **Clip segmentation** | Face/duration/frontal rules | Algorithm (rule-based) | `scripts/extract_person_clips.py` | [System card](persondet/models/README_clip_segmentation.md) |
| **Face quality — unified score** | MagFace IResNet50 (OFIQ unified quality score, ISO/IEC 29794-5) | Neural network (ONNX) | `scripts/filter_face_crops_by_quality.py`, `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_magface_iresnet50_norm.md) |
| **Face quality — sharpness** | Face sharpness random forest (OFIQ `Sharpness` measure) | Algorithm (random forest) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_sharpness_rtree.md) |
| **Face quality — compression** | SSIM compression artifacts CNN (OFIQ `CompressionArtifacts` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_ssim_248_model.md) |
| **Face quality — expression neutrality** | HSEmotion EfficientNet-B0/B2 + AdaBoost (OFIQ `ExpressionNeutrality` measure) | Neural network (ONNX) + Algorithm | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_expression_neutrality.md) |
| **Face quality — head coverings / occlusion** | BiSeNet face parsing (OFIQ `NoHeadCoverings` / `FaceOcclusionPrevention` measures) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_bisenet_400.md) |
| **Face quality — face occlusion segmentation** | Face occlusion segmentation CNN (OFIQ `FaceOcclusionPrevention` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_occlusion_segmentation_ort.md) |
| **Face quality — head pose** | MobileNetV1 head pose estimator (OFIQ `HeadPose` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_mb1_120x120.md) |
| **Audio transcription** | Whisper-Small | Neural network (PyTorch) | `scripts/transcribe_clips.py` | [Model card](persondet/models/README_openai_whisper_small.md) |

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and pre-commit hooks
- Code style rules: [Ruff](https://docs.astral.sh/ruff/) (linting & formatting) + [ty](https://docs.astral.sh/ty/) (type checking)
- PR guidelines — including the requirement to document any new pipeline component as an AI system per EU AI Act Annex IV

---

## 📄 License

The **source code** is licensed under the [Apache License 2.0](LICENSE).

The **bundled model weights** in `persondet/models/` carry separate licenses and are **not** covered by the Apache 2.0 license:

| Model | File | License |
| :---- | :--- | :------ |
| YOLOX-Tiny (HumanArt) | `yolox_tiny_...onnx` | Architecture: Apache 2.0 — weights trained on HumanArt data ⚠ verify before commercial use |
| CIGPose Wholebody | `cigpose-m_...onnx` | Architecture: Apache 2.0 — trained on COCO WholeBody ⚠ verify before commercial use |
| Whisper Small | `openai_whisper_small.pt` | MIT — commercial use permitted |
| MagFace IResNet50 | `magface_iresnet50_norm.onnx` | Apache 2.0 ([MagFace](https://github.com/IrvingMeng/MagFace)); ONNX packaging MIT ([OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| HSEmotion EfficientNet-B0 | `enet_b0_8_best_vgaf_embed_zeroed.onnx` | MIT ([HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)) |
| HSEmotion EfficientNet-B2 | `enet_b2_8_embed_zeroed.onnx` | MIT ([HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)) |
| AdaBoost neutrality classifier | `hse_1_2_C_adaboost.yml.gz` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| BiSeNet face parsing | `bisenet_400.onnx` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| Face occlusion segmentation | `face_occlusion_segmentation_ort.onnx` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| MobileNetV1 head pose | `mb1_120x120.onnx` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| SSIM compression CNN | `ssim_248_model.onnx` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |
| Face sharpness random forest | `face_sharpness_rtree.xml.gz` | MIT ([OFIQ Project](https://github.com/BSI-OFIQ/OFIQ-Project), © 2024 BSI) |

See [NOTICE](NOTICE) for full third-party attributions and dependency licenses.

---
