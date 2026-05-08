# DARDcollect: DETECTOR Archive Data Collector

This repository contains a GPU-accelerated multi-modal pipeline for downloading, processing, and annotating historical public-domain media (videos, images, audio, and documents) from the [Internet Archive](https://archive.org). Originally developed for the [DETECTOR project](https://detector-project.eu/), it extracts person detections from video and image archives, transcribes speech from audio, extracts and processes text from historical documents (PDFs and plain text), and produces annotated face crops — adaptable for any task requiring rich multi-modal historical data with [FAIR](https://www.go-fair.org/fair-principles/) metadata and full provenance tracking.

The complete pipeline downloads legacy content (videos, images, audio, texts) from the Internet Archive's public-domain collections organized by language, extracts person detections and pose keypoints from videos and images, transcribes audio from both video and standalone audio files, extracts and analyzes text from PDF and TXT documents, and produces standardized face crops with rich `.json` sidecars containing bounding boxes, pose keypoints, FAIR metadata, face quality scores, transcriptions, extracted document text, and full provenance records — enabling reproducible construction of high-quality historical multi-modal datasets (audiovisual and textual) with integrated annotations.

## 🚀 Key Features

*   **Multi-media pipeline**: Download videos, images, audio, or texts from Archive.org; extract face crops from videos (with temporal tracking) and static images (single-frame detection), then process both through the same quality annotation pipeline; extract text from PDF and TXT documents with automatic language detection.
*   **Balanced concurrent downloads**: When downloading multiple media types, tasks are submitted in round-robin fashion to ensure balanced bandwidth allocation across modalities.
*   **Language-based organization**: Videos, audio, and text files are automatically organized into language subfolders (e.g., `eng/`, `spa/`, `fra/`) for easy language-stratified processing.
*   **Filtered media discovery**: Customizable Archive.org search queries — voice-only audio (excluding music), people-only photographs, curated feature films, etc.
*   **End-to-end pipeline**: Ten decoupled stages — download, person-clip extraction (videos), image detection (images), face-crop extraction, face quality filtering, face quality annotation, audio transcription (videos and audio files), document text extraction (PDFs/TXT), frame extraction, and text annotation — each resumable and independently re-runnable.
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

**📋 FAIR Compliance:** The pipeline adheres to FAIR principles (Findable, Accessible, Interoperable, Reusable) through **embedded metadata** in sidecars — every artifact gets a UUID, timestamp, source link, and metadata. For complete details on FAIR strategy, architecture, and FAIR principles, see [docs/1-ARCHITECTURE.md § FAIR Compliance Strategy](docs/1-ARCHITECTURE.md#fair-compliance-strategy).

---

## ⚡ Quick Setup

**Complete setup guide:** [docs/0-QUICKSTART.md](docs/0-QUICKSTART.md) (5 minutes)

```bash
git clone https://github.com/Vicomtech/dardcollect.git && cd dardcollect
python -m venv .venv && source .venv/bin/activate  # Linux/macOS; .venv\Scripts\activate on Windows
pip install -e .
python scripts/download_media_from_archive.py       # Download from Archive.org
python scripts/extract_person_clips_from_videos.py  # Extract clips (videos only)
python scripts/extract_persons_from_images.py       # Detect persons (images only)
python scripts/extract_face_crops_from_videos.py    # Extract face crops
python scripts/annotate_face_quality.py             # Quality annotation
```

**Detailed setup**, GPU configuration, configuration reference, and development workflow are in the **[Documentation](#-documentation)** section below.

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

### Document Pipeline (Parallel)
```
Internet Archive  →  PDF/TXT files  →  extracted text + metadata
                     (download)         (extract_text_from_doc)
```

> **Note:** Videos, images, and documents run on independent tracks. Videos and images converge at the face crop stage for unified quality filtering and annotation. Documents are processed separately for text extraction with automatic language detection.

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
| `extract_face_crops_from_videos.py` | Processes each person track once (no check) | ⚠️ No | — |
| `extract_face_crops_from_images.py` | Processes each image once (no check) | ⚠️ No | — |
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
python scripts/extract_face_crops_from_videos.py  # For video person clips
# OR
python scripts/extract_face_crops_from_images.py  # For image detections
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
## 🎯 Processing Pipeline Overview

The pipeline has **10 stages** with incremental CSV logging at each:

| # | Stage | Input | Output | Logger |
|---|-------|-------|--------|--------|
| 1 | Download | Archive.org query | Videos, images, audio, texts | `dataset.csv` |
| 2 | Person Detection (Video) | Video files | Person clip videos + metadata | `clips_extraction.csv` |
| 2b | Person Detection (Image) | Image files | Detection JSON per image | `image_person_detection.csv` |
| 3 | Frame Extraction | Clips | PNG frames + per-frame metadata | `frames_extraction.csv` |
| 4 | Face Crop Extraction | Clips/Images | 616×616 OFIQ-aligned crops | `face_crops_extraction.csv` + `image_face_crops_extraction.csv` |
| 5 | Transcription (Clips) | Clip audio | `.transcription.json` sidecars | `transcriptions_extraction.csv` |
| 5b | Transcription (Audio) | Audio files | `.transcription.json` sidecars | `audio_transcriptions_extraction.csv` |
| 6 | Text Extraction | Documents | `.text.txt` + metadata | `document_text_extraction.csv` |
| 7 | Quality Filtering | Face crops | Filtered by MagFace score | `filtered_face_crops.csv` |
| 8 | Quality Annotation | Crops | `.quality.json` (OFIQ 7D) | `face_quality_annotation.csv` |

**Key features:**
- **Video → Image convergence:** Video clips and image detections both produce face crops, then converge at quality filtering/annotation
- **Language-based organization:** Downloaded videos/audio organized into `eng/`, `spa/`, `fra/` subfolders; images at root
- **Pose-based filtering:** CigPose keypoints detect face visibility, size, frontal orientation — robust to low-res film stock
- **Resumable:** Stages 1, 4, 5-8 checkpoint progress; run again to resume from where you left off
- **Parallel processing:** Download, video extraction, and image extraction run independently until face crop stage

**For detailed pipeline walkthrough, configuration reference, and GPU setup:** See [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md)

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

**For detailed documentation of the annotation file formats, per-frame quality data, and integration with the viewer, see [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md).**

> **Note:** OFIQ quality measures operate on 616×616 frames, which is the format they were designed for. MagFace (`unified_score`) is the exception — it needs 112×112 ArcFace crops. The script extracts these on-the-fly from each OFIQ frame using the constant region from `persondet/face_geometry.py` (detected via `crop_format: "ofiq"` in the sidecar). If absent (old-format sidecar), `unified_score` is omitted. Scores are directly comparable to those produced by the full OFIQ pipeline.

### 6. Transcribe audio

**6A. Transcribe video clips**
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

**6B. Transcribe audio files**
```bash
python scripts/transcribe_audio_files.py
```
Transcribes standalone archive audio files. Reads from language subfolders (e.g., `audio/eng/`, `audio/spa/`, etc. — searches recursively), transcribes each audio file using Whisper, and writes `.transcription.json` sidecars next to audio files. Each transcription includes:
- UUID (unique identifier)
- Transcribed speech text
- Parent reference (audio filename)
- Transcriber metadata (model size, timestamp)
- Detected language

### 6C. Extract text from documents
```bash
python scripts/extract_text_from_doc.py
```
**DOCUMENTS ONLY.** Extracts text from PDF and TXT files downloaded from Archive.org. Reads from language subfolders (e.g., `texts/eng/`, `texts/fra/`, etc. — searches recursively), extracts text content, and writes `.text.txt` files with FAIR-compliant metadata sidecars. Each extraction includes:
- UUID (unique identifier)
- Extracted text content
- Language (from folder structure)
- Character and word count
- Document metadata (title, creator from Archive.org, timestamp)
- Parent reference (source filename)

Useful for:
- Training language models on historical texts
- Text-based content analysis
- Building multi-modal datasets with text + face crops from same source collection

### 7. Extract frames (optional)
```bash
python scripts/extract_frames_from_videos.py
```
Exports video frames as PNG images with FAIR-compliant per-frame JSON sidecars and a `frames_manifest.json` for discovery. Each frame gets a UUID linking back to its parent video via UUID (full traceability). Useful for:
- Training deep learning models that require frame inputs
- Frame-level analysis and visualization
- Exporting data in non-video formats

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
DARD/
├── dataset_provenance.json               # Formal provenance record (updated after each stage)
├── archive_org_public_domain/            # Downloaded source files (videos, images, audio, texts)
│   ├── videos/eng/, videos/spa/, ...     # Language-organized video downloads
│   ├── images/                           # Image downloads (no language subfolders)
│   ├── audio/eng/, audio/spa/, ...       # Language-organized audio downloads
│   ├── texts/ger/, texts/fra/, ...       # Language-organized text downloads
│   └── dataset.csv                       # Unified metadata for all downloads (1 row per file)
├── traceability/                         # 10 CSV logs (one per processing stage)
│   ├── clips_extraction.csv
│   ├── image_person_detection.csv
│   ├── frames_extraction.csv
│   ├── face_crops_extraction.csv
│   ├── image_face_crops_extraction.csv
│   ├── transcriptions_extraction.csv
│   ├── audio_transcriptions_extraction.csv
│   ├── document_text_extraction.csv
│   ├── face_quality_annotation.csv
│   └── filtered_face_crops.csv
├── extracted_person_clips/               # Output of extract_person_clips_from_videos.py
│   ├── VideoTitle_00m15s-01m03s.mp4      # Person clip video
│   └── VideoTitle_00m15s-01m03s.json     # Sidecar: bboxes, keypoints, face crop geometry
├── extracted_image_detections/           # Output of extract_persons_from_images.py (JSON only)
│   ├── image_001.json
│   └── image_002.json
├── extracted_face_crops/                 # Output of face crop extraction
│   ├── VideoTitle_00m15s-01m03s_face_1.mp4
│   ├── VideoTitle_00m15s-01m03s_face_1.json
│   └── VideoTitle_00m15s-01m03s_face_1.quality.json    # Written by annotate_face_quality.py
├── filtered_face_crops/                  # Output of quality filtering (quality-passing crops)
│   ├── VideoTitle_00m15s-01m03s_face_1.mp4
│   └── VideoTitle_00m15s-01m03s_face_1.json
├── extracted_frames/                     # Output of frame extraction (optional)
│   ├── VideoTitle/frame_000000.png
│   └── VideoTitle/frame_000000.json
├── transcriptions/                       # Transcription sidecars (linked from clips/audio)
│   ├── VideoTitle_00m15s-01m03s.transcription.json
│   └── audio_001.transcription.json
└── preprocessed_documents/               # Output of document text extraction
    ├── document_001.txt
    └── document_001.annotation.json
```

**Every artifact is traced back to its source:**
- Each CSV has a `source_id` column linking to parent artifact or Archive.org ID
- Face crops link back to source clip/image via UUID
- Quality annotations link back to source crop via UUID
- Complete chain: Archive.org ID → Download UUID → Video → Clip UUID → Crop UUID → Quality scores

---

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[docs/0-QUICKSTART.md](docs/0-QUICKSTART.md)** | 5-minute installation & execution walkthrough | Everyone (start here) |
| **[docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md)** | System overview, pipeline diagrams, modality workflows, detection models | Technical users |
| **[docs/2-LOGGING.md](docs/2-LOGGING.md)** | CSV schemas, FAIR compliance, provenance chains, querying examples | Data users |
| **[docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md)** | Annotation formats, OFIQ quality dimensions, sidecar JSON structure | ML practitioners |
| **[docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md)** | GPU setup, configuration reference, detailed pipeline walkthrough, development workflow | Developers |
| **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** | Code style, pre-commit hooks, AI system documentation, contributing guidelines | Contributors |

---

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

Contributions are welcome. Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
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
