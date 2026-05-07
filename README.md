# DARDcollect: DETECTOR Archive Data Collector

This repository contains a GPU-accelerated multi-modal pipeline for downloading, processing, and annotating public-domain archive media (videos, images, audio, texts). Originally developed for the [DETECTOR project](https://detector-project.eu/), it extracts person detections, transcribes audio, and produces annotated face crops — adaptable for any task requiring rich multi-modal data with FAIR metadata.

The pipeline downloads public-domain content (videos, images, audio, texts) from the [Internet Archive](https://archive.org), extracts person detections from videos and images, transcribes audio from videos and audio files, and produces standardized face crops with rich `.json` sidecars containing bounding boxes, pose keypoints, FAIR metadata, face quality scores, and transcriptions — enabling reproducible construction of high-quality face datasets.

For a detailed guide to quality annotation formats, see [ANNOTATIONS.md](ANNOTATIONS.md).

## 🚀 Key Features

*   **Multi-media pipeline**: Download videos, images, audio, or texts from Archive.org; extract faces from both videos and static images using the same face crop and quality annotation pipeline.
*   **Balanced concurrent downloads**: When downloading multiple media types, tasks are submitted in round-robin fashion to ensure balanced bandwidth allocation across modalities.
*   **Language-based organization**: Videos, audio, and text files are automatically organized into language subfolders (e.g., `eng/`, `spa/`, `fra/`) for easy language-stratified processing.
*   **Filtered media discovery**: Customizable Archive.org search queries — voice-only audio (excluding music), people-only photographs, curated feature films, etc.
*   **End-to-end pipeline**: Eight decoupled stages — download, person-clip extraction (videos), image detection (images), face-crop extraction, face quality filtering, face quality annotation, audio transcription (videos and audio files), and frame extraction — each resumable and independently re-runnable.
*   **Parallel processing**: Video and image pipelines run independently until the face crop stage, where they converge for unified quality filtering and annotation.
*   **Pose-based face filtering**: Face visibility, minimum size, frontal orientation, and mouth-open detection are all derived from CIGPose wholebody keypoints — robust to the low resolution and grain of pre-1960 film stock where pixel-based face detectors struggle.
*   **Keypoint-based duplicate suppression**: Overlapping tracklets are removed by comparing pose keypoint positions, so a single person never generates two competing tracks — robust even when persons are close together.
*   **Scene-aware segmentation**: Hard cuts reset the tracker and close the current clip, preventing track IDs and clip content from bleeding across shots.
*   **GPU accelerated**: YOLOX and CIGPose run via ONNX/TensorRT on CUDA 12; Whisper transcription runs on the same GPU.
*   **Resumable at every stage**: All scripts checkpoint progress; interrupted runs continue from the last processed frame or clip without data loss.
*   **Provenance record**: Each pipeline stage appends a signed run entry to `DARD/dataset_provenance.json`, capturing dataset origin (Archive.org identifiers and URLs), collection timestamp, model names and SHA-256 checksums, and the full configuration snapshot used — satisfying formal data provenance requirements.
*   **EU AI Act documented**: Every automated component — including rule-based algorithms — is documented as an AI system per Annex IV.
*   **FAIR compliant**: All dataset sidecars embed unique identifiers (UUIDs), schema versioning, source tracking, and Archive.org metadata directly — enabling reproducibility, data citation, and interoperability across tools without external registries or separate metadata files.

---

## 📋 FAIR Principles: Findability, Accessibility, Interoperability, Reusability

The dataset produced by this pipeline adheres to FAIR principles through **embedded metadata** in sidecars:

### What's Tracked

| Aspect | How It's Implemented |
| :--- | :--- |
| **Unique Identity** | Every person clip, face crop, transcription, and quality annotation gets a UUID v4 at creation — allows permanent linking and citation |
| **Schema Versioning** | Every sidecar includes `schema_version` (e.g., `"1.0"`) — enables format evolution and backwards compatibility |
| **Source Tracing** | Person clips include Archive.org metadata (`archive_org_id`, `archive_org_url`, `license`) — full provenance chain to original source |
| **Parent References** | Face crops link to parent person clip (UUID + filename); quality annotations link to parent crop; transcriptions link to parent clip — enables reproducible reconstruction of full lineage |
| **Automatic Validation** | `jsonschema` validates all sidecars during write operations via formal JSON schemas (`schemas/*.json`) — invalid sidecars raise detailed errors immediately |

### Example Sidecar Chain

```
Person Clip (UUID: 550e8400...)
  ├─ [transcription parent ref] → Transcription (UUID: 550e8400..., parent_clip.uuid: 550e8400...)
  └─ [face crop parent ref] → Face Crop (UUID: 550e8400..., parent_clip.uuid: 550e8400...)
       └─ [quality parent ref] → Quality Annotation (UUID: 550e8400..., parent_crop.uuid: 550e8400...)
```

Every step maintains full traceability back to the original Internet Archive source through UUIDs and parent references. No external registry or separate metadata files needed.

For complete details on FAIR metadata fields, see [ANNOTATIONS.md § FAIR Principles](ANNOTATIONS.md#fair-principles-data-findability-accessibility-interoperability-reusability).

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
# Multi-media download — choose which types to download
media_types: ["video"]              # Options: ["video"], ["image"], ["audio"], ["text"], or combinations

# Download settings — per-media-type (configured in media_download section)
media_download:
  video:
    output_subdir: "videos"
    min_duration_minutes: 20        # Skip videos shorter than this
    search_query: "..."             # Archive.org search query for feature films
  image:
    output_subdir: "images"
    search_query: "..."             # Archive.org search query for people photos (portraits, family, groups)
  audio:
    output_subdir: "audio"
    search_query: "..."             # Archive.org search query for spoken word (audiobooks, radio, sermons)
  text:
    output_subdir: "texts"
    search_query: "..."             # Archive.org search query for documents

base_output_dir: "DARD/archive_org_public_domain"  # Base; creates subdirectories per media type
max_total_size_gb: 20               # Stop downloading once total size reached
max_workers: 10                     # Parallel download threads (balanced round-robin across media types)

# Audio transcription settings (VIDEOS AND AUDIO FILES)
transcription:
  person_clips_dir: "DARD/extracted_person_clips"  # ← videos from person extraction
  audio_files_dir: "DARD/archive_org_public_domain/audio"  # ← audio files from media_download.audio
  overwrite: false                  # Set true to re-transcribe files that already have .transcription.json

# Video person extraction (VIDEOS ONLY)
person_extraction:
  input_dir: "DARD/archive_org_public_domain/archive_org_videos"  # ← videos from media_download.video
  output_clips_dir: "DARD/extracted_person_clips"  # ← must match face_crop_extraction.input_dir
  detection_threshold: 0.4
  require_face_visibility: true
  min_face_visible_frames: 15

# Image person detection (IMAGES ONLY — parallel to person_extraction)
image_extraction:
  input_dir: "DARD/archive_org_public_domain/archive_org_images"  # ← images from media_download.image
  output_detections_dir: "DARD/extracted_image_detections"        # Detection JSONs (not videos)
  detection_threshold: 0.4
  require_face_visibility: true
  min_face_size_percent: 10.0

# Face crop settings (BOTH videos and images)
face_crop_extraction:
  input_dir: "DARD/extracted_person_clips"  # Can also point to extracted_image_detections
  output_dir: "DARD/face_crops"
```

**Important parameters:**

- **`media_types`**: List of media types to download. Examples:
  - `["video"]` — Videos only (default)
  - `["video", "image"]` — Download both videos and images  
  - `["image"]` — Images only
  - Downloads from multiple types happen **concurrently with balanced load** (round-robin task submission)
- **`max_total_size_gb`**: Download stops when this limit is reached. Prevents runaway downloads. Default: 20 GB.
- **`max_workers`**: Number of parallel download threads. Default: 10. Applies across all media types.

**Language-based organization:**
- Videos, audio files, and texts are automatically organized into **language subfolders** (e.g., `eng/`, `spa/`, `fra/`)
- Images are stored at the root level (no language subfolders)
- Files without language metadata go to the root directory

> **Note:** Path linking across sections:
> - Video path: download (`videos/eng/`, `videos/spa/`, etc.) → `person_extraction.input_dir` → `face_crop_extraction.input_dir`
> - Image path: download (`images/`) → `image_extraction.input_dir` → (skip to) `face_crop_extraction.input_dir` (if extracting face crops from images)
> - Audio path: download (`audio/eng/`, `audio/spa/`, etc.) → `transcription` (voices are indexed by language)

---

## 🎞️ Processing Pipeline

### Video Pipeline
```
Internet Archive  →  raw .mp4  →  person clips  →  face crops  →  filtered crops  →  quality annotated  →  transcriptions
                     (download)   (extract_from_videos)  (extract_crops)  (filter_quality)   (annotate_quality)   (transcribe)
```

### Audio Pipeline (Parallel)
```
Internet Archive  →  audio files  →  transcriptions
                     (download)      (transcribe_audio)
```

### Image Pipeline (Parallel)
```
Internet Archive  →  .jpg images  →  person detections  ↘
                     (download)      (extract_persons_from_images)  ↘
                                                                     → face crops (video OR image)
                                                                       (extract_face_crops_from_videos
                                                                        or _from_images)  
                                                                     → filtered crops  →  quality annotated
                                                                       (filter_quality)  (annotate_quality)
```

> **Note:** Videos and images run on separate tracks until the face crop stage. From that point, both pipelines converge — face crops from videos (`.mp4` from `extract_face_crops_from_videos.py`) and images (`.jpg`/`.png` from `extract_face_crops_from_images.py`) use the same processing for quality filtering and annotation.

> **VS Code users:** all pipeline stages are available as launch configurations in the Run and Debug panel (`.vscode/launch.json`) — no need to type commands manually.

---

## 📊 Workflow & Annotations Integration

### Step-by-Step Execution & File Creation

**Video pipeline** (if `media_types` includes `"video"`):

| Stage | Script | Input | Creates | Modifies | Output Dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1️⃣ | `download_media_from_archive.py` | Internet Archive | `.mp4` videos | — | `videos/` |
| 2️⃣ | `extract_person_clips_from_videos.py` | `.mp4` videos | `.mp4` + `.json` (UUID) | — | `extracted_person_clips/` |
| 7️⃣ | `transcribe_video_clips.py` | Person clips | `.transcription.json` (parent=UUID from 2️⃣) | — | `extracted_person_clips/` |

**Audio pipeline** (if `media_types` includes `"audio"` — parallel to video):

| Stage | Script | Input | Creates | Modifies | Output Dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1️⃣ | `download_media_from_archive.py` | Internet Archive | `.mp3`/`.wav`/etc audio files | — | `audio/` |
| 7️⃣ | `transcribe_audio_files.py` | Audio files | `.transcription.json` (parent=audio filename) | — | `audio/` |

**Image pipeline** (if `media_types` includes `"image"` — runs in parallel):

| Stage | Script | Input | Creates | Modifies | Output Dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1️⃣ | `download_media_from_archive.py` | Internet Archive | `.jpg` images | — | `images/` |
| 2️⃣ | `extract_persons_from_images.py` | `.jpg` images | `.json` detections (UUID) | — | `extracted_image_detections/` |

**Convergent pipeline** (both video and image crops):

| Stage | Script | Input | Creates | Modifies | Output Dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3️⃣ | `extract_face_crops_from_videos.py` | Person clip videos | `.mp4` + `.json` (parent=UUID from 2️⃣) | — | `face_crops/` |
| 3️⃣ | `extract_face_crops_from_images.py` | Image detections | `.jpg` + `.json` (parent=UUID from 2️⃣) | — | `face_crops/` |
| 4️⃣ | `filter_face_crops_by_quality.py` | Video OR image crops | (moves files) | — | `filtered_face_crops/` |
| 5️⃣ | `annotate_face_quality.py` | Filtered crops | `.quality.json` (annotation) | Detection JSON (back-prop) | `filtered_face_crops/` |
| 8️⃣ | `extract_frames_from_videos.py` | Any video crop | `.png` frames + `.json` per-frame | — | `extracted_frames/` |

### File Organization & Sidecar Relationships

**Download Metadata** (Unified Dataset CSV):
```
archive_org_public_domain/
  └─ dataset.csv          ← All downloads (videos, audio, images, texts) in one row per file
                             {uuid, archive_org_identifier, media_type, filename, language, 
                              title, creator, download_stage_script, download_stage_timestamp, ...}
```
This single unified CSV contains all file-level metadata from all media types. Resumable downloads filter this CSV by `media_type` + `archive_org_identifier` to skip already-downloaded files.

**Person Clips** (Base layer):
```
extracted_person_clips/
  ├─ VideoTitle.mp4
  ├─ VideoTitle.json                    ← UUID: A (root of tree)
  │                                        {uuid: A, schema_version, source (Archive.org), frame_data, ...}
  └─ VideoTitle.transcription.json      ← UUID: B, parent_clip.uuid: A
                                           {uuid: B, parent_clip: {uuid: A, ...}, transcription: "..."}
```

**Face Crops** (extracted from person clip persons):
```
face_crops/ (or filtered_face_crops/)
  ├─ VideoTitle_face_0.mp4
  ├─ VideoTitle_face_0.json             ← UUID: C, parent_clip.uuid: A
  │                                        {uuid: C, parent_clip: {uuid: A, ...}, ...}
  └─ VideoTitle_face_0.quality.json     ← UUID: D, parent_crop.uuid: C (ANNOTATION)
                                           {uuid: D, parent_crop: {uuid: C, ...}, 
                                            unified_score: {...}, sharpness: {...}, ...}
```

### UUID Linking (FAIR Traceability)

```
Person Clip Sidecar (VideoTitle.json)
  UUID: A
  ├─→ Transcription (VideoTitle.transcription.json)
  │   UUID: B, parent_clip.uuid: A
  │
  └─→ Face Crop (VideoTitle_face_0.json)
      UUID: C, parent_clip.uuid: A
      └─→ Quality Annotation (VideoTitle_face_0.quality.json)
          UUID: D, parent_crop.uuid: C
```

Every sidecar includes:
- **`uuid`**: Unique identifier (UUID v4) for this data item — enables permanent linking and citation
- **`parent_clip` or `parent_crop`**: Reference to parent (UUID + filename) — enables traceability back to source

This creates a **complete audit trail** from transcription → person clip → Archive.org, without needing external databases.

### Back-Propagation of Quality Annotations

After `annotate_face_quality.py` runs, quality scores are **automatically written back** to the source person clip's `.json`:

**Before** (person clip JSON):
```json
{
  "uuid": "550e8400...",
  "track_ids": [0, 1],
  "frame_data": {...}
  // No face_quality field yet
}
```

**After** (person clip JSON updated):
```json
{
  "uuid": "550e8400...",
  "track_ids": [0, 1],
  "frame_data": {...},
  "face_quality": {
    "0": {
      "unified_score": {"max": 52.3, "mean": 45.8, ...},
      "sharpness": {"max": 95.0, "mean": 87.1, ...},
      ...,
      "face_crop": "VideoTitle_face_0.mp4"
    },
    "1": {
      ...
    }
  }
}
```

**Result**: The viewer can show quality scores directly in person clip view without needing separate file I/O.

### Resumability Per Script

Each script is **independently resumable**:

| Script | How It Works | Resumable? | Config Parameter |
| :--- | :--- | :--- | :--- |
| `download_media_from_archive.py` | Reads unified `dataset.csv`; skips files already present (filtered by media_type + archive_org_identifier) | ✅ Yes | — |
| Script | How It Works | Resumable? | Config Parameter |
| :--- | :--- | :--- | :--- |
| `download_media_from_archive.py` | Reads unified `dataset.csv`; skips files already present (filtered by media_type + archive_org_identifier) | ✅ Yes | — |
| `extract_person_clips_from_videos.py` | Processes each video once (no check) | ⚠️ No | — |
| `extract_persons_from_images.py` | Processes each image once (no check) | ⚠️ No | — |
| `extract_face_crops.py` | Processes each person track once (no check) | ⚠️ No | — |
| `filter_face_crops_by_quality.py` | Moves files; skips if already in output | ✅ Yes | — |
| `annotate_face_quality.py` | Skips clips that already have `.quality.json` | ✅ Yes | `overwrite: true` to re-annotate |
| `transcribe_video_clips.py` | Skips clips that already have `.transcription.json` | ✅ Yes | `overwrite: true` to re-transcribe |
| `transcribe_audio_files.py` | Skips audio files that already have `.transcription.json` | ✅ Yes | `overwrite: true` to re-transcribe |
| `extract_frames_from_videos.py` | Skips already-extracted frames | ✅ Yes | `overwrite: false` (default) |

**Key**: Stages 1, 4, 5, 6, 7 are resumable. Stages 2, 3 are destructive but only run once per clip.

### Annotation Workflow Example

Start with `clip_001.mp4`:

```bash
# Step 1: Extract person clip from video
python scripts/extract_person_clips_from_videos.py
# Creates: extracted_person_clips/clip_001.json (UUID: A)
#          extracted_person_clips/clip_001.mp4

# Step 2: Extract face crops
python scripts/extract_face_crops.py
# Creates: face_crops/clip_001_face_0.json (parent_clip.uuid: A)
#          face_crops/clip_001_face_0.mp4

# Step 3: Annotate quality
python scripts/annotate_face_quality.py
# Creates: face_crops/clip_001_face_0.quality.json (parent_crop.uuid: C)
# Updates: extracted_person_clips/clip_001.json (adds face_quality[0])

# Step 4: Transcribe video audio
python scripts/transcribe_video_clips.py
# Creates: extracted_person_clips/clip_001.transcription.json (parent_clip.uuid: A)

# Step 5: Index for viewer
cd viewer && python index_data.py
# Reads all files and creates data_index.json mapping all relationships

# Result: Viewer can display all annotations unified
```

---

### ⚡ GPU Acceleration & TensorRT

The pipeline uses **TensorRT** for GPU-accelerated ONNX model inference when available. On first run, TensorRT compiles models to optimized engine formats and caches them — this makes the first invocation of each script **slow** (minutes) but subsequent runs **fast** (seconds). Cached engines are stored in `.cache/trt_engines/`.

**First run**: Slow (TensorRT compilation) ⏱️  
**Subsequent runs**: Fast (cached engines) ⚡

If you see warnings like "⚠️ TensorRT is enabled — compiling X model on first use (may take a moment)", this is normal and expected. The slowness is a one-time cost for significant speedup.

#### Verifying GPU Execution

The quality annotation scripts (`annotate_face_quality.py` and `filter_face_crops_by_quality.py`) log the **actual execution provider** during inference. On the first frame, you'll see output like:

```
  Actual execution provider during inference: TensorrtExecutionProvider
```

This confirms which GPU provider was used. If TensorRT is available and enabled, it will compile on first use; subsequent invocations skip compilation and use cached engines (visible in logs: "✓ TensorRT engines cached: N files").

### 1. Download media
```bash
python scripts/download_media_from_archive.py
```
Searches the Internet Archive for public-domain content matching queries in `config.yaml` and downloads them based on `media_types`. Downloads from **all active media types happen concurrently** in a single thread pool with **round-robin task scheduling** to ensure balanced bandwidth allocation across modalities.

- **Videos** (`"video"` in `media_types`): Downloaded to `videos/` — feature films, classic TV
- **Images** (`"image"` in `media_types`): Downloaded to `images/` — people photos (portraits, family, group photographs)
- **Audio** (`"audio"` in `media_types`): Downloaded to `audio/` — spoken word only (audiobooks, radio programs, sermons, no music)
- **Texts** (`"text"` in `media_types`): Downloaded to `texts/` — books, documents, scripts

**All downloads are recorded in a single unified CSV** at `archive_org_public_domain/dataset.csv`:
```
uuid,archive_org_identifier,media_type,filename,language,title,creator,download_stage_script,download_stage_timestamp,...
550e8400...,film_001,video,film_001.mp4,eng,The Great Film,Griffith,scripts/download_media_from_archive.py,2026-05-07T12:51:23,...
550e8401...,radio_001,audio,radio_001.mp3,eng,Old Time Radio,NBC,scripts/download_media_from_archive.py,2026-05-07T12:51:25,...
550e8402...,photo_001,image,photo_001.jpg,,Family Photo,Unknown,scripts/download_media_from_archive.py,2026-05-07T12:51:27,...
550e8403...,book_001,text,book_001.pdf,ger,German Grammar,Goethe,scripts/download_media_from_archive.py,2026-05-07T12:51:30,...
```

**Files are automatically organized by language** (when language metadata is available):
```
archive_org_public_domain/
├── videos/
│   ├── eng/          # English-language videos
│   ├── spa/          # Spanish-language videos
│   └── ...
├── audio/
│   ├── eng/          # English-language audio
│   ├── fra/          # French-language audio
│   └── ...
├── texts/
│   ├── ger/          # German-language texts
│   └── ...
├── images/           # (no language subfolders; images at root)
└── dataset.csv       # Unified metadata for all downloads (one row per file)
```

Each media type uses a different Archive.org search query (configured in `config.yaml` → `media_download.<type>.search_query`). Skips files already present (resumable). Respects `max_total_size_gb`.

> **Safety note**: Downloads use atomic temp files (`.tmp` extension). If interrupted mid-download, the incomplete `.tmp` file is cleaned up automatically on the next run — you won't end up with corrupted files.
>
> **Balanced concurrency & resumability**: When multiple media types are active (e.g., `media_types: ["video", "audio", "image"]`), the download script interleaves tasks by media type in round-robin fashion for consistent bandwidth allocation. Re-running the script resumes downloads by checking the unified `dataset.csv` — each media type is filtered separately, so you can selectively re-download one modality without re-downloading others.

**Example** — To download both videos and images:
```yaml
media_types: ["video", "image"]

media_download:
  video:
    enabled: true        # Will be downloaded
    search_query: "..."
  image:
    enabled: true        # Will be downloaded
    search_query: "..."
```

### 2a. Extract person clips from videos
```bash
python scripts/extract_person_clips_from_videos.py
```
**VIDEOS ONLY.** Reads videos from `person_extraction.input_dir` (e.g., `archive_org_public_domain/videos/eng/`, `archive_org_public_domain/videos/spa/`, etc. — searches recursively across language subfolders). For each video, detects and tracks persons frame-by-frame, filters by face visibility and frontal orientation, and writes accepted clips to `output_clips_dir` as `.mp4` + `.json` pairs. Each `.json` sidecar contains per-frame bounding boxes, pose keypoints, and clip-level statistics.

### 2b. Extract persons from images
```bash
python scripts/extract_persons_from_images.py
```
**IMAGES ONLY.** Parallel to `extract_person_clips_from_videos.py` — reads static images from `image_extraction.input_dir` (e.g., `archive_org_public_domain/images/`). For each image, detects persons and filters by face visibility and size. Writes detection JSONs (no video output) to `output_detections_dir`. Each JSON contains:
- Detected persons (bounding boxes, pose keypoints)
- Face size and visibility metadata
- UUID and FAIR source tracking
- Parent reference back to source image

**Key difference from video pipeline**: No temporal tracking, no clip segmentation — just a single JSON per image with all detected persons.

### 3. Extract face crops (convergent pipeline)
Two modality-specific scripts, both producing 616×616 OFIQ-aligned face crops:

**3a. Extract face crops from videos**
```bash
python scripts/extract_face_crops_from_videos.py
```
**VIDEOS ONLY.** Reads person clip videos from `face_crop_extraction.input_dir/` (output of `extract_person_clips_from_videos.py`). For each detected person track, extracts **face crop videos** (`.mp4`) with the same frame count as the original clip. Each crop is aligned to OFIQ standard geometry (616×616). Outputs to `output_dir/` with sidecars containing per-frame detection data.

**3b. Extract face crops from images**
```bash
python scripts/extract_face_crops_from_images.py
```
**IMAGES ONLY.** Reads static image detections from `face_crop_extraction.input_dir/` (output of `extract_persons_from_images.py`). For each detected person, extracts one **face crop image** (`.jpg`). Aligned to OFIQ standard geometry (616×616). Outputs to `output_dir/` with sidecars containing detection data.

**Convergence Point:** Both scripts output 616×616 OFIQ crops with identical sidecar JSON structure (start_frame, end_frame, video_info, per-frame frame_data). Downstream stages (filter, annotate) treat crops uniformly regardless of source.

### 4. Filter face crops by quality
```bash
python scripts/filter_face_crops_by_quality.py
```
Reads OFIQ crops from `face_quality_filtering.input_dir/` — accepts **both video (`.mp4`) and image (`.jpg`/`.png`) crops** from stages 3a and 3b. For each crop, scores frames using the **OFIQ unified quality score** ([ISO/IEC 29794-5](https://www.iso.org/standard/81694.html), reference implementation: [BSI-OFIQ/OFIQ-Project](https://github.com/BSI-OFIQ/OFIQ-Project)): the output magnitude of MagFace IResNet50, which measures how confidently a face recognition model can embed a crop. MagFace requires 112×112 ArcFace crops — these are extracted on-the-fly from each OFIQ frame using the constant region from `persondet/face_geometry.py`. When a crop passes, the file and its sidecar are moved to `face_quality_filtering.output_dir/`. Optional — run this if you need a quality-controlled subset.

Key config option under `face_quality_filtering`:
- **`quality_threshold`**: MagFace score required to pass [0, 100]. For videos, every frame is scored; a clip passes as soon as one frame meets the threshold. For images, the single frame is scored.

### 5. Annotate face quality
```bash
python scripts/annotate_face_quality.py [<config_file>]
```
Reads OFIQ-aligned crops from the directory specified in `config.yaml` → `face_quality_annotation.input_dir` (typically `filtered_face_crops/`). Accepts **both video (`.mp4`) and image (`.jpg`/`.png`) crops** from stage 4. For each crop, samples frames at the stride and max-frames specified in config, runs all OFIQ quality models, then writes a sibling `.quality.json` file next to each crop file. Already-annotated crops are skipped on re-run; set `face_quality_annotation.overwrite: true` in config to re-score them.

Pass `[<config_file>]` to use a custom config file; defaults to `config.yaml` in project root.

The following measures are computed per video/image crop, each stored as `{max, mean, p10, p50, p90}` over the sampled frames:

| Measure | OFIQ component | Calibrated score range |
| :--- | :--- | :--- |
| `unified_score` | `UnifiedQualityScore` (MagFace magnitude) | [0, 100] |
| `sharpness` | `Sharpness` (Laplacian/Sobel RTrees) | [0, 100] |
| `compression_artifacts` | `CompressionArtifacts` (SSIM CNN) | [0, 100] |
| `expression_neutrality` | `ExpressionNeutrality` (HSEmotion EfficientNet-B0/B2 + AdaBoost) | [0, 100] |
| `no_head_coverings` | `NoHeadCoverings` (BiSeNet parsing — hat/cloth fraction) | [0, 100] |
| `face_occlusion_prevention` | `FaceOcclusionPrevention` (FaceOcclusionSegmentation CNN) | [0, 100] |
| `head_pose` | `HeadPose` (MobileNetV1 3DDFAV2) | yaw/pitch/roll degrees + cosine² quality scores |

**For detailed documentation of the annotation file formats, per-frame quality data, and integration with the viewer, see [ANNOTATIONS.md](ANNOTATIONS.md).**

> **Note:** OFIQ quality measures operate on 616×616 frames, which is the format they were designed for. MagFace (`unified_score`) is the exception — it needs 112×112 ArcFace crops. The script extracts these on-the-fly from each OFIQ frame using the constant region from `persondet/face_geometry.py` (detected via `crop_format: "ofiq"` in the sidecar). If absent (old-format sidecar), `unified_score` is omitted. Scores are directly comparable to those produced by the full OFIQ pipeline.

### 7. Transcribe audio

**7A. Transcribe video clips**
```bash
python scripts/transcribe_video_clips.py
```
Transcribes audio from extracted person clip videos. Reads from `person_clips_dir` (e.g., `DARD/extracted_person_clips`), transcribes each `.mp4` file using Whisper, and writes `.transcription.json` sidecars next to clips. Each transcription includes:
- UUID (unique identifier)
- Transcribed speech text
- Parent reference (clip UUID)
- Transcriber metadata (model size, timestamp)
- Detected language

Optional — only needed if speech content is relevant to your use case.

**7B. Transcribe audio files**
```bash
python scripts/transcribe_audio_files.py
```
Transcribes standalone archive audio files. Reads from language subfolders (e.g., `audio/eng/`, `audio/spa/`, etc. — searches recursively), transcribes each audio file using Whisper, and writes `.transcription.json` sidecars next to audio files. Each transcription includes:
- UUID (unique identifier)
- Transcribed speech text
- Parent reference (audio filename)
- Transcriber metadata (model size, timestamp)
- Detected language

Configure in `config.yaml`:
```yaml
transcription:
  person_clips_dir: "DARD/extracted_person_clips"  # ← Videos from video extraction
  audio_files_dir: "DARD/archive_org_public_domain/audio"  # ← Audio files (searched recursively)
  overwrite: false                              # Set true to re-transcribe
```

### 8. Extract frames (optional)
```bash
python scripts/extract_frames_from_videos.py
```
Exports video frames as PNG images with FAIR-compliant per-frame JSON sidecars and a `frames_manifest.json` for discovery. Each frame gets a UUID linking back to its parent video via UUID (full traceability). Useful for:
- Training deep learning models that require frame inputs
- Frame-level analysis and visualization
- Exporting data in non-video formats

Configure in `config.yaml`:
```yaml
frame_extraction:
  input_dir: "DARD/extracted_person_clips"  # Source: person clips, face crops, or filtered crops
  output_dir: "DARD/extracted_frames"
  overwrite: false                          # Type (person_clip/face_crop/filtered_face_crop) inferred from input_dir
```

Output structure per video:
```
DARD/extracted_frames/VideoTitle/
  frame_000000.png             ← Video frame
  frame_000000.json            ← Frame metadata (UUID, parent link, detections)
  frame_000001.png
  frame_000001.json
  ...
  frames_manifest.json         ← Index of all frames + UUIDs
```

### 8. Interactive Viewer
**File**: `viewer/detection_viewer.html`

A local HTML tool to inspect extraction results: browse clips, play video with overlaid bounding boxes and keypoints, review face quality scores, and inspect face crops.

Supports two folder types — drop either into the viewer:

| Folder | What you see |
| :--- | :--- |
| `extracted_person_clips/` | Full-body clips with bounding boxes, keypoints, and ArcFace/OFIQ crop quads overlaid on the source video. Each tracked person shows their quality summary in an accordion panel (if quality annotations exist) |
| `face_crops/` or `filtered_face_crops/` | 616×616 OFIQ face crop videos with the ArcFace region (yellow quad) overlaid. Quality metrics display dynamically as you play the video, showing **per-frame scores** from the `.quality.json` sidecar. Metrics update in real-time when you scrub through the timeline |

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
├── extracted_person_clips/               # Output of extract_person_clips_from_videos.py
│   ├── VideoTitle_00m15s-01m03s.mp4      # Person clip
│   ├── VideoTitle_00m15s-01m03s.json     # Sidecar: bboxes, keypoints, face_crop_corners_arcface/ofiq, stats
│   └── VideoTitle_progress.json          # Resume checkpoint (internal)
├── face_crops/                           # Output of extract_face_crops.py (616×616 OFIQ crops)
│   ├── VideoTitle_00m15s-01m03s_face_1.mp4
│   ├── VideoTitle_00m15s-01m03s_face_1.json          # crop_format: "ofiq", same format as person clip sidecars
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
- `transcription`: filled in by `transcribe_video_clips.py` (empty string until then)

Face crop sidecars (in `face_crops/`) use the same format as person clip sidecars: `start_frame` (always 0), `end_frame`, `start_seconds` (always 0.0), `end_seconds`, `duration_seconds`, `video_info`, and `frame_data` with per-frame entries containing `bbox` (in OFIQ space), `score`, `keypoints`, `keypoint_scores`, and `face_crop_corners_arcface`. Additional metadata: `source_video`, `track_id`, `crop_format` (`"ofiq"`), `output_size` (616), `valid_face_frames`.

Quality annotation files (`.quality.json`, written by `annotate_face_quality.py` alongside each video) contain: 
- **Per-measure aggregate statistics**: `max`, `mean`, `p10`, `p50`, `p90` for each quality measure (`unified_score`, `sharpness`, `compression_artifacts`, `expression_neutrality`, `no_head_coverings`, `face_occlusion_prevention`, and `head_pose`).
- **Per-frame quality data**: A `frame_data` array with per-frame scores and frame indices, allowing dynamic per-frame visualization in the viewer as you play the video.
- **Back-propagation**: Quality summaries are automatically written into the source person clip sidecars under `face_quality[track_id]`, so you can browse per-track quality when reviewing person clips in the viewer.

Head pose additionally stores raw angles (degrees, signed) and per-angle cosine² quality scores.

---

## ⏸️ Interrupting & Resuming

All pipeline scripts support **resumable checkpointing**. You can interrupt processing at any time and restart from where you left off.

### How to interrupt
Press `Ctrl+C` in the terminal where the script is running.

### How to resume
Run the same script again with the same `config.yaml`:
```bash
python scripts/extract_person_clips_from_videos.py
# Ctrl+C after frame 5000 of a video
python scripts/extract_person_clips_from_videos.py
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
- **Annotate face quality**: Which videos already have a sibling `.quality.json` file (skipped on re-run; set `face_quality_annotation.overwrite: true` in config to re-score)
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
| **Scene change detection** | Luminance histogram + bbox area | Algorithm (rule-based) | `scripts/extract_person_clips_from_videos.py` | [System card](persondet/models/README_scene_change_detector.md) |
| **Clip segmentation** | Face/duration/frontal rules | Algorithm (rule-based) | `scripts/extract_person_clips_from_videos.py` | [System card](persondet/models/README_clip_segmentation.md) |
| **Face quality — unified score** | MagFace IResNet50 (OFIQ unified quality score, ISO/IEC 29794-5) | Neural network (ONNX) | `scripts/filter_face_crops_by_quality.py`, `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_magface_iresnet50_norm.md) |
| **Face quality — sharpness** | Face sharpness random forest (OFIQ `Sharpness` measure) | Algorithm (random forest) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_sharpness_rtree.md) |
| **Face quality — compression** | SSIM compression artifacts CNN (OFIQ `CompressionArtifacts` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_ssim_248_model.md) |
| **Face quality — expression neutrality** | HSEmotion EfficientNet-B0/B2 + AdaBoost (OFIQ `ExpressionNeutrality` measure) | Neural network (ONNX) + Algorithm | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_expression_neutrality.md) |
| **Face quality — head coverings / occlusion** | BiSeNet face parsing (OFIQ `NoHeadCoverings` / `FaceOcclusionPrevention` measures) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_bisenet_400.md) |
| **Face quality — face occlusion segmentation** | Face occlusion segmentation CNN (OFIQ `FaceOcclusionPrevention` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_face_occlusion_segmentation_ort.md) |
| **Face quality — head pose** | MobileNetV1 head pose estimator (OFIQ `HeadPose` measure) | Neural network (ONNX) | `scripts/annotate_face_quality.py` | [Model card](persondet/models/README_mb1_120x120.md) |
| **Audio transcription** | Whisper-Small | Neural network (PyTorch) | `scripts/transcribe_video_clips.py`, `scripts/transcribe_audio_files.py` | [Model card](persondet/models/README_openai_whisper_small.md) |

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
