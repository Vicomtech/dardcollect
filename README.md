# DETECTOR Archive Data Collector

This repository contains a GPU-accelerated pipeline for building a labelled audiovisual dataset from public-domain archive film. Originally developed for the [DETECTOR project](https://cordis.europa.eu/project/id/101225942), it can be adapted for any task requiring annotated video of real people — speaker recognition, facial analysis, action recognition, and similar.

The pipeline covers the full journey from raw video to annotated dataset: it downloads public-domain films from the [Internet Archive](https://archive.org), detects and tracks persons frame-by-frame, filters clips by face visibility and frontal orientation, extracts face crops, and transcribes speech — producing per-clip `.mp4` files with rich `.json` sidecars containing bounding boxes, pose keypoints, and transcription.

## 🚀 Key Features

*   **End-to-end pipeline**: Four decoupled stages — download, person-clip extraction, face-crop extraction, and audio transcription — each resumable and independently re-runnable.
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
output_dir: "DETDF/archive_org_public_domain"  # Where to save downloaded videos
max_total_size_gb: 10                          # Stop downloading once limit reached
search_query: "..."                            # Archive.org search query
min_duration_minutes: 20                        # Skip clips shorter than this

person_extraction:
  input_dir: "DETDF/archive_org_public_domain"     # ← must match output_dir above
  output_clips_dir: "DETDF/extracted_person_clips" # ← must match face_crop_extraction.input_dir
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
  output_size: 224  # Square output pixels (112 for ArcFace, 224 for ViT-based models)
```

**Important parameters:**

- **`max_total_size_gb`**: Download stops immediately when this limit is reached. Prevents runaway downloads filling your disk. Default: 100 GB.

> **Note:** The paths between sections are linked — `output_dir` (download) feeds `person_extraction.input_dir`, and `person_extraction.output_clips_dir` feeds `face_crop_extraction.input_dir`. Change them together.

---

## 🎞️ Processing Pipeline

```
Internet Archive  →  raw .mp4 files  →  person clips + sidecars  →  face crops  →  transcriptions
                     (download)          (extract_person_clips)      (face_crops)    (transcribe)
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
Reads clips from `output_clips_dir`. For each tracked person, extracts an aligned, padded face crop and writes a per-track `.mp4` to `face_crop_extraction.output_dir`. Optional — run this if you need identity-level face video rather than full-body clips.

### 4. Transcribe audio
```bash
python scripts/transcribe_clips.py
```
Reads clips from `output_clips_dir` and transcribes their audio using Whisper-Small, writing a `transcription` field into each `.json` sidecar in-place. Optional — only needed if speech content is relevant to your use case.

### 5. Interactive Viewer
**File**: `viewer/detection_viewer.html`

A local HTML tool to inspect extraction results: browse clips, play video with overlaid bounding boxes and keypoints, and review transcriptions.

#### Local Usage (Desktop)
1. Open `detection_viewer.html` in a web browser.
2. Click **Select Folder** and choose `extracted_person_clips`.

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
├── dataset_provenance.json             # Formal provenance record (updated after each stage)
├── archive_org_public_domain/          # Downloaded source videos
│   └── *.mp4
├── extracted_person_clips/             # Output of extract_person_clips.py
│   ├── VideoTitle_00m15s-01m03s.mp4    # Person clip
│   ├── VideoTitle_00m15s-01m03s.json   # Sidecar: bboxes, keypoints, stats
│   └── VideoTitle_progress.json        # Resume checkpoint (internal)
└── face_crops/                         # Output of extract_face_crops.py
    ├── VideoTitle_00m15s-01m03s_track1.mp4
    └── ...
```

Each `.json` sidecar contains:
- Clip metadata: source video (forward-slash path), timestamps, duration, FPS, `track_ids` list
- Per-frame data: bounding boxes, pose keypoints (1 dp), confidence scores for each tracked person
- Summary statistics: `face_visible_frames`, `max_consecutive_face_frames`, `mouth_open_frames`
- `transcription`: filled in by `transcribe_clips.py` (empty string until then)

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
- **Transcribe**: Which clips have been transcribed

---

## 🧠 AI Systems

Each automated component of the pipeline is documented as an AI system in accordance with EU AI Act Annex IV, regardless of whether it uses a learned model or a rule-based algorithm.

| Task | System | Type | Implementation | Documentation |
| :--- | :----- | :--- | :------------- | :------------ |
| **Detection** | YOLOX-Tiny (HumanArt) | Neural network (ONNX) | `persondet/detector.py` | [Model card](persondet/models/README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md) |
| **Tracking** | OC-SORT | Algorithm (model-free) | `persondet/tracker.py` | [System card](persondet/models/README_ocsort.md) |
| **Pose estimation** | CIGPose Wholebody (COCO 133) | Neural network (ONNX) | `persondet/poser.py` | [Model card](persondet/models/README_cigpose-m_coco-wholebody_256x192.md) |
| **Scene change detection** | Luminance histogram + bbox area | Algorithm (rule-based) | `scripts/extract_person_clips.py` | [System card](persondet/models/README_scene_change_detector.md) |
| **Clip segmentation** | Face/duration/frontal rules | Algorithm (rule-based) | `scripts/extract_person_clips.py` | [System card](persondet/models/README_clip_segmentation.md) |
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

See [NOTICE](NOTICE) for full third-party attributions and dependency licenses.

---
