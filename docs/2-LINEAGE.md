# 📊 DARDcollect Data Provenance & Traceability


## Contents

- [Overview](#overview)
- [CSV Schemas](#csv-schemas)
  - [Downloads](#1-download-manifest-downloadscsv)
  - [Clips](#2-clips-extraction-log-csv)
  - [Frames](#3-frames-extraction-log-csv)
  - [Face Crops](#4-face-crops-extraction-log-csv)
  - [Transcriptions](#5-transcriptions-extraction-log-csv)
  - [Filtered Crops](#6-filtered-face-crops-log-csv)
  - [Quality Annotations](#7-face-quality-annotation-log-csv)
- [Tracing Scenarios](#8-how-to-trace-artifacts-through-the-complete-pipeline)
- [FAIR Compliance](#9-fair-compliance)
- [Provenance by Modality](#10-provenance-by-modality)
- [Data Lineage Example](#11-data-lineage-example)
- [Integration into Scripts](#12-integration-into-your-scripts)
- [Querying Traceability Data](#13-querying-traceability-data)
- [Additional Loggers](#14-additional-loggers-images-audio-documents)
- [Custom Data Sources](#15-custom-data-sources-non-archiveorg-workflows)
- [References](#16-references)

---

## Overview

**CSV files track everything through the entire extraction pipeline:**

| Stage | CSV (co-located with output) | Purpose | Links To |
|-------|------------------------------|---------|----------|
| **1. Download** | `archive_org_public_domain/downloads.csv` | Archive.org downloads with UUID + metadata | Archive.org |
| **2. Person Clips** | `extracted_person_clips/clips_extraction.csv` | Video clips with people detected | downloads.csv |
| **3. Frames** | `extracted_frames/frames_extraction.csv` | Individual frames extracted from clips | clips_extraction.csv |
| **4. Face Crops (video)** | `face_crops/face_crops_extraction.csv` | Face regions from person clips | clips_extraction.csv |
| **4. Face Crops (image)** | `face_crops/image_face_crops_extraction.csv` | Face regions from static images | image_person_detection.csv |
| **5. Transcriptions (video)** | `extracted_person_clips/transcriptions_extraction.csv` | Speech transcribed from person clips | clips_extraction.csv |
| **5. Transcriptions (audio)** | `audio_transcriptions/audio_transcriptions_extraction.csv` | Speech transcribed from audio files | downloads.csv |
| **6. Image Detection** | `extracted_image_detections/image_person_detection.csv` | Person detections in static images | downloads.csv |
| **7. Quality Filter** | `filtered_face_crops/filtered_face_crops.csv` | High-quality crops after MagFace filtering | face_crops_extraction.csv |
| **8. Quality Annotation** | `filtered_face_crops/face_quality_annotation.csv` | OFIQ 7-dimension quality scores | filtered_face_crops.csv |
| **9. Documents** | `preprocessed_documents/document_text_extraction.csv` | Text extracted from PDFs/TXTs | downloads.csv |

All paths are relative to `DARD/`.

**Complete workflow visualization:**

```
                     Archive.org
                          ↓
           download_media_from_archive.py
                   ↓          ↓
          downloads.csv (all media types)
              ↓
    extract_person_clips_from_videos.py
         ↓               ↓
    clips_extraction.csv (person clips)
      ↙      ↓       ↖
  extract_frames  extract_face_crops  transcribe_clips
     ↓               ↓                    ↓
frames_extraction  face_crops_extraction  transcriptions_extraction
     │               ↓
     │    filter_face_crops_by_quality
     │               ↓
     │    filtered_face_crops.csv
     │               ↓
     │    annotate_face_quality
     │               ↓
     │    face_quality_annotation.csv
     └→ (frames available for any downstream task)
```

**Trace any artifact to its source:**
```bash
# 1. Find a face crop's source clip
grep "crop_xyz" DARD/face_crops/face_crops_extraction.csv
# → source_clip: "Finger_Man_02m09s-02m12s.mp4"

# 2. Find that clip's source video
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/clips_extraction.csv
# → source_video: "Finger Man (1955).mp4"

# 3. Find the download record
grep "Finger Man (1955).mp4" DARD/archive_org_public_domain/downloads.csv
# → UUID + creator, license, download timestamp

# Complete chain: Archive.org → Video → Clip → Face Crop → Quality Annotation
```

---

## Full Technical Documentation

### Overview: Complete Traceability Chain

DARDcollect implements end-to-end traceability from Archive.org source through extraction to final products:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Archive.org (Internet Archive)                                      │
│ Source: Public domain films (1900-1955)                             │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ↓ download_media_from_archive.py
         ├─ Creates: DARD/archive_org_public_domain/downloads.csv
         │  UUID + metadata for each download
         │  (title, creator, year, license, download timestamp)
         │
┌────────┴────────────────────────────────────────────────────────────┐
│ DARD/archive_org_public_domain/                                     │
│ ├─ Finger_Man_1955.mp4                                              │
│ ├─ The_Crooked_Web_1955.mp4                                         │
│ └─ downloads.csv (traceability starts here)                           │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ↓ extract_person_clips_from_videos.py
         ├─ Creates: DARD/extracted_person_clips/clips_extraction.csv
         │  Real-time log: each clip + source video + confidence
         │
         ├─ Creates: DARD/extracted_person_clips/
         │  ├─ Finger_Man_02m09s-02m12s.mp4 (clip)
         │  └─ Finger_Man_02m09s-02m12s.json (FAIR metadata + source UUID)
         │
┌────────┴────────────────────────────────────────────────────────────┐
│ DARD/extracted_person_clips/                                        │
│ └─ clips_extraction.csv (links to source)                           │
└─────────────────────────────────────────────────────────────────────┘
```

**Traceability Path:** Archive.org ID → Download UUID → Video filename → Extracted clips → Detection metadata

Each artifact is uniquely identifiable and can be traced back to its source through the CSV files and JSON sidecars.

---

## CSV Schemas

## 1. Download Manifest (downloads.csv)

**File:** `DARD/archive_org_public_domain/downloads.csv`

Records all media files downloaded from Archive.org. This is the **starting point** of the complete traceability chain.

**Schema:** dynamic — columns grow as new Archive.org metadata fields are encountered across items.

Fixed pipeline fields (always present, always first):
```
uuid, archive_org_identifier, filename_downloaded, media_type, downloaded_at,
download_stage_script, download_stage_timestamp
```

Followed by all fields from Archive.org's `item.metadata` for that item. Standard Archive.org fields
that commonly appear include `title`, `creator`, `date`, `year`, `description`, `licenseurl`,
`subject`, `collection`, `language`, `mediatype`, `addeddate`, `publicdate`, `uploader`, and others.
Items with no value for a given field leave that cell empty. When a new item introduces a field not
yet seen, the CSV is rewritten to add that column (existing rows get an empty cell for it).

**Empty cells are expected.** Archive.org metadata is heterogeneous across media types: OCR fields
(`ocr`, `ocr_detected_lang`, `pdf_module_version`…) only appear on scanned texts; film fields
(`sound`, `color`, `director`, `runtime`…) only on videos. A unified CSV across all types will
always be sparse — filtering by `media_type` and dropping all-empty columns is enough to get a
clean view for a specific type.

**Known Archive.org field redundancies** (kept as-is — filtering them would require maintaining a
manual exclusion list against an evolving schema):

| Field | Redundant with | Note |
|-------|---------------|------|
| `license` | `licenseurl` | `licenseurl` is Archive.org's standard field; `license` is rarely populated |
| `identifier-access` | `archive_org_identifier` | the access URL is `https://archive.org/details/{identifier}` — derivable |
| `year` | `date` | usually the first four characters of `date`; Archive.org stores both |
| `keywords` | `subject` | both are topic tags set by the uploader |
| `collection_added` | `collection` | secondary collections added after upload |
| `mediatype` | `media_type` | Archive.org's own type label vs. the pipeline's classification from `config.yaml` |

**Key characteristics:**
- ✅ **Unique identifier (UUID):** Every download gets a uuid4 for permanent identification
- ✅ **Complete Archive.org metadata:** All `item.metadata` fields captured — not a manual subset
- ✅ **License tracking:** `licenseurl` field documents original source license
- ✅ **Download timestamp:** ISO 8601 UTC format for audit trail
- ✅ **Media classification:** video|audio|image|text for organization

**Purpose:** Answer "where did this file come from?"
```bash
# Find all downloaded files by creator
grep "Leo McCarey" DARD/archive_org_public_domain/downloads.csv

# Check license compliance for all downloads
awk -F',' 'NR>1 {print $11}' DARD/archive_org_public_domain/downloads.csv | sort | uniq

# Find a specific download by UUID
grep "a3f8c9e2-1a2b-4c3d-5e6f-7a8b9c0d1e2f" DARD/archive_org_public_domain/downloads.csv
```

---

## 2. Clips Extraction Log (CSV)

**File:** `DARD/extracted_person_clips/clips_extraction.csv`

Real-time audit log that tracks every extracted person clip, linking it to its source video. This CSV is written **incrementally** as clips are extracted—each row is appended immediately to disk, ensuring the log survives if the extraction process is interrupted.

**Columns:**
```
uuid, archive_org_identifier, timestamp, source_video, fps,
start_frame, end_frame, start_seconds, duration_seconds,
max_persons_per_frame, detector_model, detector_confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `archive_org_identifier`: Archive.org item ID, linked from downloads.csv
- `max_persons_per_frame`: peak simultaneous person count across all frames in the clip
- `detector_confidence`: average confidence across all detections in all frames

**Example:**
```csv
uuid,archive_org_identifier,timestamp,source_video,fps,start_frame,end_frame,start_seconds,duration_seconds,max_persons_per_frame,detector_model,detector_confidence,output_path
a1b2c3d4-...,finger_man_1955,2026-05-09T10:22:00Z,Finger Man (1955).mp4,30.0,3270,3360,139.8,3.0,1,yolox-tiny,0.952,DARD/extracted_person_clips/Finger Man (1955)_02m09s-02m12s.mp4
```

**Key characteristics:**
- ✅ **UUID per row:** Stable identifier for cross-CSV linkage
- ✅ **Archive.org link:** `archive_org_identifier` connects to downloads.csv
- ✅ **Incremental writes:** One row appended per clip (immediate disk write)
- ✅ **Timestamped:** ISO 8601 UTC timestamps for every entry
- ✅ **Detection metadata:** FPS, frame range, peak person count, average confidence
- ✅ **Resilient:** Survives process interruptions—nothing is lost

**Typical usage:**
```bash
# Count total clips extracted
tail -n +2 clips_extraction.csv | wc -l

# Find all clips from a specific source video
grep "Finger Man (1955)" clips_extraction.csv

# Find all clips from a specific Archive.org item
grep "finger_man_1955" clips_extraction.csv
```

---

## 3. Frames Extraction Log (CSV)

**File:** `DARD/extracted_frames/frames_extraction.csv`

Logs individual frames extracted from person clips (typically for detailed face analysis, model training, etc.).

**Columns:**
```
uuid, clip_uuid, timestamp, source_clip_path, frame_number, timestamp_seconds, output_path
```

- `uuid`: row identifier (UUID4)
- `clip_uuid`: UUID of the parent row in `clips_extraction.csv`
- `source_clip_path`: full path to the source clip video

**Key characteristics:**
- ✅ `clip_uuid` links to clips_extraction.csv (direct UUID join, no filename matching needed)
- ✅ Frame index + timestamp in seconds (locate exact frame in source)
- ✅ Incremental writes (survive interruptions)

**Usage:**
```bash
# Find all frames from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_frames/frames_extraction.csv

# Count total frames extracted
tail -n +2 DARD/extracted_frames/frames_extraction.csv | wc -l
```

---

## 4. Face Crops Extraction Log (CSV)

**File:** `DARD/face_crops/face_crops_extraction.csv`

Tracks face regions extracted from person clips or images (critical for linking crops to original sources).

**Columns:**
```
uuid, parent_uuid, timestamp, crop_id, source_type, source_path, face_bbox, confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `parent_uuid`: UUID of the parent row — in `clips_extraction.csv` (when `source_type="person_clip"`) or in `downloads.csv` (when `source_type="image"`)
- `crop_id`: derived from output filename stem; kept in CSV because downstream loggers (`FilteredFaceCropsLogger`, `FaceQualityAnnotationLogger`) use it as a lookup key
- `source_type`: `"person_clip"` or `"image"`
- `face_bbox`: bounding box as `"x1,y1,x2,y2"` in source-frame coordinates

**Key characteristics:**
- ✅ **parent_uuid** links to upstream CSV (clips or downloads) without filename matching
- ✅ **source_type**: enables cross-media traceability
- ✅ **crop_id**: lookup key used by quality filter and annotation loggers

**Usage:**
```bash
# Find all face crops from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/face_crops/face_crops_extraction.csv

# Count crops by source type (clip vs. image)
awk -F',' 'NR>1 {print $6}' DARD/face_crops/face_crops_extraction.csv | sort | uniq -c
```

---

## 5. Transcriptions Extraction Log (CSV)

**File:** `DARD/extracted_person_clips/transcriptions_extraction.csv`

Logs transcriptions extracted from person clips (speech-to-text with language detection and confidence).

**Columns:**
```
uuid, clip_uuid, timestamp, source_clip_path, language_detected, confidence,
word_count, duration_seconds, model_version, output_path
```

- `uuid`: row identifier (UUID4)
- `clip_uuid`: UUID of the parent row in `clips_extraction.csv`

**Key characteristics:**
- ✅ `clip_uuid` links to clips_extraction.csv (direct UUID join)
- ✅ Language detection + confidence
- ✅ Word count + duration (quality metrics)
- ✅ Model version (reproducibility)

**Usage:**
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count transcriptions by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c
```

---

## 6. Filtered Face Crops Log (CSV)

**File:** `DARD/filtered_face_crops/filtered_face_crops.csv`

Tracks face crops that pass quality filtering (MagFace score ≥ threshold), linking back to original crops.

**Columns:**
```
uuid, crop_uuid, timestamp, source_crop_path, magface_score, filter_threshold, output_path
```

- `uuid`: row identifier (UUID4)
- `crop_uuid`: UUID of the parent row in `face_crops_extraction.csv`

**Key characteristics:**
- ✅ `crop_uuid` links to face_crops_extraction.csv (direct UUID join)
- ✅ MagFace score (quality metric, calibrated [0, 100])
- ✅ Filter threshold recorded for reproducibility

**Usage:**
```bash
# Count crops that passed filter
tail -n +2 DARD/filtered_face_crops/filtered_face_crops.csv | wc -l

# Average MagFace score of filtered crops
awk -F',' 'NR>1 {sum+=$5; count++} END {print "Avg MagFace: " sum/count}' \
  DARD/filtered_face_crops/filtered_face_crops.csv
```

---

## 7. Face Quality Annotation Log (CSV)

**File:** `DARD/filtered_face_crops/face_quality_annotation.csv`

Logs OFIQ (Open Face Image Quality) scores for face crops (7 dimensions: unified_score, sharpness, compression_artifacts, expression_neutrality, no_head_coverings, face_occlusion_prevention, head_pose).

**Columns:**
```
uuid, crop_uuid, timestamp, crop_id, crop_path,
sharpness, compression_artifacts, expression_neutrality,
no_head_coverings, face_occlusion_prevention, unified_score,
yaw_quality, pitch_quality, roll_quality, passed_filter
```

- `uuid`: row identifier (UUID4)
- `crop_uuid`: UUID of the parent row in `face_crops_extraction.csv`
- `crop_id`: derived from crop filename stem (kept for cross-referencing)
- All score columns are the **max over all sampled frames** for that crop
- `unified_score`: MagFace magnitude (same metric used in `filter_face_crops_by_quality.py`)
- `yaw/pitch/roll_quality`: OFIQ head-pose quality, `round(100 × cos²(angle))`
- `passed_filter`: always `True` (annotation runs on all crops in the input folder)

**Key characteristics:**
- ✅ `crop_uuid` links to face_crops_extraction.csv (direct UUID join)
- ✅ All 7 OFIQ scalar measures + 3 head-pose quality scores
- ✅ `unified_score` duplicates MagFace from `filtered_face_crops.csv` but with full per-crop stats in the `.quality.json` sidecar

**Usage:**
```bash
# Crops with high sharpness
awk -F',' 'NR>1 && $6 >= 90 {print $5}' DARD/filtered_face_crops/face_quality_annotation.csv

# Average unified_score
awk -F',' 'NR>1 {sum+=$11; count++} END {print "Avg unified_score: " sum/count}' \
  DARD/filtered_face_crops/face_quality_annotation.csv
```

---

## 8. How to Trace Artifacts Through the Complete Pipeline

### Scenario 1: Trace a Face Crop Back to Its Original Video

**You have:** `crop_00042.jpg` (a face crop file)  
**You want to know:** Which original Archive.org video did this come from?

**Steps:**
```bash
# Step 1: Find the crop in face_crops_extraction.csv
grep "crop_00042" DARD/face_crops/face_crops_extraction.csv
# Output: 2026-05-07T15:25:02Z,crop_00042,person_clip,Finger_Man_02m09s-02m12s,...
# → Source clip: "Finger_Man_02m09s-02m12s.mp4"

# Step 2: Find that clip in clips_extraction.csv
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/clips_extraction.csv
# Output: 2026-05-07T15:22:00Z,Finger_Man_02m09s-02m12s,Finger Man (1955).mp4,...
# → Source video: "Finger Man (1955).mp4"

# Step 3: Find the download record
grep "Finger Man (1955).mp4" DARD/archive_org_public_domain/downloads.csv
# Output: a3f8c9e2-...,movies__20200210@0_x264.mkv,...
# → Archive.org ID, creator, year, license, download timestamp
```

**Result:** Complete chain: Archive.org "Finger Man" (1955, Leo McCarey) → Downloaded → Extracted clip → Face crop

### Scenario 2: Trace a Transcription Back to Its Clip

**You have:** `The_Crooked_Web_00m57s-01m30s.transcription.json`  
**You want to know:** What clip did this come from, and what's its full provenance?

```bash
# The transcription filename stem matches the source clip stem directly.
# Confirm in transcriptions_extraction.csv (source_clip_path is col 4):
grep "The_Crooked_Web_00m57s-01m30s" DARD/extracted_person_clips/transcriptions_extraction.csv
# → source_clip_path shows the full clip path

# Step 2: Get clip metadata (fps, frame range, max_persons_per_frame, confidence)
grep "The_Crooked_Web_00m57s-01m30s" DARD/extracted_person_clips/clips_extraction.csv

# Step 3: Trace to original video
grep "The Crooked Web" DARD/archive_org_public_domain/downloads.csv
# → Full source metadata
```

### Scenario 3: Link a Quality Annotation to Its Crop and Original Clip

**You have:** A quality annotation with poor scores  
**You want to:** Understand where that crop came from

```bash
# Find crops with low sharpness (col 6 = sharpness, col 5 = crop_path)
awk -F',' 'NR>1 && $6 < 50 {print $5}' DARD/filtered_face_crops/face_quality_annotation.csv | head -5

# For a specific crop, find its source clip
# Crop names encode the parent clip: "Finger_Man_02m09s-02m12s_face_0" → clip "Finger_Man_02m09s-02m12s"
CROP_STEM="Finger_Man_02m09s-02m12s_face_0"
# source_path is col 6 in face_crops_extraction.csv
grep "$CROP_STEM" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f6
# → DARD/extracted_person_clips/Finger_Man_02m09s-02m12s.mp4

# Find that clip's source video (col 5 = source_video in clips_extraction.csv)
CLIP_STEM="Finger_Man_02m09s-02m12s"
grep "$CLIP_STEM" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f5
# → "Finger Man (1955).mp4"
```

---

## 9. FAIR Compliance

### 9.1 Findable
- **Starting point:** UUID for each download (downloads.csv)
- **Extraction tracking:** Timestamp + UUID for each extraction
- **Indexing:** CSV indices at both download and extraction stages
- **Searchable:** Source URL, media type, creator, date fields

### 9.2 Accessible
- **Multiple formats:** CSV (human-readable), JSON-LD (machine-readable)
- **Documentation:** Complete technical spec + quick reference guide
- **License tracking:** Original source license preserved in downloads.csv
- **No lock-in:** Open formats, no proprietary codecs

### 9.3 Interoperable
- **Standard formats:** CSV, JSON-LD, ISO 8601 timestamps
- **Linked data:** References Dublin Core, PROV-O ontologies
- **Metadata schemas:** Matches `person_clip_schema.json` + FAIR metadata
- **Cross-system links:** UUIDs enable integration with other databases

### 9.4 Reusable
- **Complete provenance:** From Archive.org ID through extraction to final product
- **Processing details:** Model names, versions, detector confidence documented
- **Source attribution:** Original creator, date, license always preserved
- **Reconstruction capability:** URLs stored, can regenerate data if needed

---

## 10. Provenance by Modality

Each modality has detailed provenance tracking:

### 10.1 VIDEO Modality
**Source:** Archive.org public domain films  
**Processing:** `pipeline/extract_person_clips_from_videos.py`

**Tracked metadata:**
- Original media identifier (Archive.org item ID)
- Download timestamp
- Source video resolution, FPS, duration
- Clip extraction parameters (temporal window, confidence thresholds)
- Face/pose detection models used
- Processing version

### 10.2 AUDIO Modality
**Source:** Transcoded from video or separate audio from Archive.org

**Tracked metadata:**
- Audio extraction date/time
- Sample rate, bit depth, channels
- Transcription models (Whisper version, language)
- Confidence scores
- Processing chain

### 10.3 FACE_CROPS Modality
**Source:** Extracted from person_clips using face detection

**Tracked metadata:**
- Source person_clip reference
- Face bounding box coordinates (frame-relative)
- Face detection model & confidence
- OFIQ quality scores
- Face embedding version (if applicable)

### 10.4 TRANSCRIPTIONS Modality
**Source:** Audio transcription

**Tracked metadata:**
- Source audio reference
- Speech recognition model (e.g., Whisper-small)
- Language detected
- Confidence per segment
- Alignment timestamps

---

## 11. Data Lineage Example

```
Archive.org [Finger Man (1955).mp4]
    ↓ download_media_from_archive.py
DARD/archive_org_public_domain/videos/eng/[Finger Man (1955).mp4]
    ↓ extract_person_clips_from_videos.py
        (detector: yolox-tiny, tracker: ocsort, poser: cigpose)
DARD/extracted_person_clips/[Finger Man_02m09s-02m12s.mp4]
    ├─ metadata: [Finger Man_02m09s-02m12s.json]
    ├─→ extract_face_crops_from_videos.py
    │   DARD/face_crops/[Finger_Man_02m09s-02m12s_face_0.mp4]
    └─→ transcribe_video_clips.py
        DARD/extracted_person_clips/[Finger_Man_02m09s-02m12s.transcription.json]
```

---

## 12. Integration into Your Scripts

### For All Pipeline Stages

Each script should initialize and use the appropriate logger(s):

```python
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.pipeline_loggers import (
    FramesExtractionLogger,
    FaceCropsExtractionLogger,
    TranscriptionsExtractionLogger,
    FaceQualityAnnotationLogger,
    FilteredFaceCropsLogger,
)

# Initialize loggers for your stage
logger = ExtractionLogger(output_dir="DARD/extracted_person_clips")  # For clips
frames_logger = FramesExtractionLogger(output_dir="DARD/extracted_person_clips")  # For frames
face_crops_logger = FaceCropsExtractionLogger(output_dir="DARD/extracted_person_clips")  # For face crops
trans_logger = TranscriptionsExtractionLogger(output_dir="DARD/extracted_person_clips")  # For transcriptions
quality_logger = FaceQualityAnnotationLogger(output_dir="DARD/filtered_face_crops")  # For OFIQ scores
filter_logger = FilteredFaceCropsLogger(output_dir="DARD/filtered_face_crops")  # For filtered crops
```

### Example: `extract_frames_from_videos.py`

```python
from dardcollect.pipeline_loggers import FramesExtractionLogger
from pathlib import Path

clips_csv = Path(cfg.input_dir) / "clips_extraction.csv"
frames_logger = FramesExtractionLogger(
    output_dir="DARD/extracted_frames",
    clips_csv_path=clips_csv,          # enables clip_uuid lookup
)

# For each frame extracted from a clip
frames_logger.log_frame_extraction(
    source_clip_path=str(clip_path),
    frame_number=frame_num,
    timestamp_seconds=frame_num / fps,
    output_path=str(output_frame_path),
)

frames_logger.print_summary()
```

### Example: `extract_face_crops_from_videos.py`

```python
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger

clips_csv = Path(face_config.input_dir) / "clips_extraction.csv"
face_crops_logger = FaceCropsExtractionLogger(
    output_dir="DARD/face_crops",
    clips_csv_path=clips_csv,          # enables parent_uuid lookup for person_clip sources
)

# For each face crop extracted from a person clip
face_crops_logger.log_face_crop_extraction(
    source_type="person_clip",         # or "image"
    source_path=str(clip_path),
    face_bbox=f"{x1},{y1},{x2},{y2}",
    confidence=detection_confidence,
    output_path=str(output_crop_path),
)

face_crops_logger.print_summary()
```

### Example: `transcribe_video_clips.py`

```python
from dardcollect.pipeline_loggers import TranscriptionsExtractionLogger

clips_csv = person_clips_dir / "clips_extraction.csv"
trans_logger = TranscriptionsExtractionLogger(
    output_dir=str(person_clips_dir),
    clips_csv_path=clips_csv,          # enables clip_uuid lookup
)

# After transcribing a clip
trans_logger.log_transcription(
    source_clip_path=str(clip_path),
    language_detected=language,
    confidence=avg_confidence,
    word_count=len(words),
    duration_seconds=clip_duration,
    output_path=str(output_json_path),
    model_version="whisper-small",
)

trans_logger.print_summary()
```

### Example: `filter_face_crops_by_quality.py`

```python
from dardcollect.pipeline_loggers import FilteredFaceCropsLogger

face_crops_csv = input_dir / "face_crops_extraction.csv"
filter_logger = FilteredFaceCropsLogger(
    output_dir="DARD/filtered_face_crops",
    face_crops_csv_path=face_crops_csv,  # enables crop_uuid lookup
)

# For each crop that passes the MagFace threshold
if magface_score >= threshold:
    filter_logger.log_filtered_crop(
        source_crop_path=str(input_crop_path),
        magface_score=magface_score,
        filter_threshold=threshold,
        output_path=str(output_crop_path),
    )

filter_logger.print_summary()
```

### Example: `annotate_face_quality.py`

```python
from dardcollect.pipeline_loggers import FaceQualityAnnotationLogger

face_crops_csv = Path(face_crop_cfg.output_dir) / "face_crops_extraction.csv"
quality_logger = FaceQualityAnnotationLogger(
    output_dir=str(input_dir),
    face_crops_csv_path=face_crops_csv,  # enables crop_uuid lookup
)

# After computing OFIQ scores
head_pose = quality_data.get("head_pose", {})
quality_logger.log_quality_annotation(
    crop_path=str(crop_path),
    sharpness=quality_data.get("sharpness", {}).get("max", 0.0),
    compression_artifacts=quality_data.get("compression_artifacts", {}).get("max", 0.0),
    expression_neutrality=quality_data.get("expression_neutrality", {}).get("max", 0.0),
    no_head_coverings=quality_data.get("no_head_coverings", {}).get("max", 0.0),
    face_occlusion_prevention=quality_data.get("face_occlusion_prevention", {}).get("max", 0.0),
    unified_score=quality_data.get("unified_score", {}).get("max", 0.0),
    yaw_quality=head_pose.get("yaw_quality", {}).get("max", 0.0),
    pitch_quality=head_pose.get("pitch_quality", {}).get("max", 0.0),
    roll_quality=head_pose.get("roll_quality", {}).get("max", 0.0),
    passed_filter=True,
)

quality_logger.print_summary()
```

---

## 13. Querying Traceability Data

### Basic Queries by Stage

**Person clips** — columns: `uuid(1) archive_org_identifier(2) timestamp(3) source_video(4) fps(5) start_frame(6) end_frame(7) start_seconds(8) duration_seconds(9) max_persons_per_frame(10) ...`
```bash
# Find all clips from a source video
grep "Finger Man" DARD/extracted_person_clips/clips_extraction.csv

# Count total clips
tail -n +2 DARD/extracted_person_clips/clips_extraction.csv | wc -l

# Average max_persons_per_frame across all clips
awk -F',' 'NR>1 {sum+=$11; count++} END {print sum/count}' \
  DARD/extracted_person_clips/clips_extraction.csv
```

**Frames** — columns: `uuid(1) clip_uuid(2) timestamp(3) source_clip_path(4) frame_number(5) timestamp_seconds(6) output_path(7)`
```bash
# Find all frames from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_frames/frames_extraction.csv

# Count frames extracted from all clips
tail -n +2 DARD/extracted_frames/frames_extraction.csv | wc -l
```

**Face crops** — columns: `uuid(1) parent_uuid(2) timestamp(3) crop_id(4) source_type(5) source_path(6) face_bbox(7) confidence(8) output_path(9)`
```bash
# Find all crops from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/face_crops/face_crops_extraction.csv

# Count crops by source type (clip vs. image)
awk -F',' 'NR>1 {print $5}' DARD/face_crops/face_crops_extraction.csv | sort | uniq -c

# Average detection confidence
awk -F',' 'NR>1 {sum+=$8; count++} END {print "Avg: " sum/count}' \
  DARD/face_crops/face_crops_extraction.csv
```

**Transcriptions** — columns: `uuid(1) clip_uuid(2) timestamp(3) source_clip_path(4) language_detected(5) confidence(6) word_count(7) ...`
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c
```

**Quality annotations** — columns: `uuid(1) crop_uuid(2) timestamp(3) crop_id(4) crop_path(5) sharpness(6) compression_artifacts(7) expression_neutrality(8) no_head_coverings(9) face_occlusion_prevention(10) unified_score(11) yaw_quality(12) pitch_quality(13) roll_quality(14) passed_filter(15)`
```bash
# Find low-quality crops (col 6 = sharpness, col 5 = crop_path)
awk -F',' 'NR>1 && $6 < 50 {print $5}' DARD/filtered_face_crops/face_quality_annotation.csv

# Count passed vs. failed (col 15 = passed_filter)
awk -F',' 'NR>1 {if ($15=="True") passed++; else failed++}
END {print "Passed: " passed ", Failed: " failed}' \
  DARD/filtered_face_crops/face_quality_annotation.csv
```

### Complex Tracing Queries

**Trace a face crop through all stages:**
```bash
# Given a crop_id, find its complete journey
CROP_ID="crop_00042"

echo "=== Face Crop Journey ==="
echo "1. Original crop extraction:"
grep "$CROP_ID" DARD/face_crops/face_crops_extraction.csv

echo -e "\n2. Source clip (col 6 = source_path):"
CLIP_STEM=$(grep "$CROP_ID" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f6 | xargs basename | sed 's/\..*//')
grep "$CLIP_STEM" DARD/extracted_person_clips/clips_extraction.csv

echo -e "\n3. Source video (col 5 = source_video):"
VIDEO=$(grep "$CLIP_STEM" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f5)
grep "$VIDEO" DARD/archive_org_public_domain/downloads.csv

echo -e "\n4. Quality annotation (if any):"
grep "$CROP_ID" DARD/filtered_face_crops/face_quality_annotation.csv

echo -e "\n5. Filtered status (if passed):"
grep "$CROP_ID" DARD/filtered_face_crops/filtered_face_crops.csv
```

**Find all outputs from a specific downloaded video:**
```bash
VIDEO_NAME="Finger Man (1955).mp4"

echo "=== All outputs from $VIDEO_NAME ==="

echo -e "\n📹 Person clips (output_path col 13):"
grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv | awk -F',' '{print $13}'

# Derive clip stems from output_path (col 13) for cross-CSV lookup
CLIPS=$(grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv \
  | awk -F',' '{gsub(/.*[\/\\]/, "", $13); gsub(/\.[^.]+$/, "", $13); printf "%s|", $13}' \
  | sed 's/|$//')

echo -e "\n👤 Face crops from those clips:"
grep -E "$CLIPS" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f1 | wc -l

echo -e "\n📸 Frames from those clips:"
grep -E "$CLIPS" DARD/extracted_frames/frames_extraction.csv | wc -l

echo -e "\n🎙️ Transcriptions from those clips:"
grep -E "$CLIPS" DARD/extracted_person_clips/transcriptions_extraction.csv | wc -l
```

**Quality analysis for a video series:**
```bash
# Find all crops from a video and their quality scores
VIDEO_NAME="Finger Man (1955).mp4"
CLIPS=$(grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f4 | tr '\n' '|')

echo "=== Quality Analysis for $VIDEO_NAME ==="
# col 5 = crop_path, col 6 = sharpness, col 11 = unified_score
awk -F',' -v pattern="$CLIPS" '$5 ~ pattern {
  print $5 ": sharpness=" $6 ", unified_score=" $11
}' DARD/filtered_face_crops/face_quality_annotation.csv | head -20
```

---

## 14. Additional Loggers: Images, Audio, Documents

### Image Person Detection Logger
**File:** `DARD/extracted_image_detections/image_person_detection.csv`

Tracks person detections extracted from static images (script: `extract_persons_from_images.py`).

**Columns:**
```
uuid, download_uuid, timestamp, source_image, source_image_path,
num_persons, detector_model, detector_confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `download_uuid`: UUID of the parent row in `downloads.csv`
- `source_image`: filename only — kept as a derived field because `ImageFaceCropsExtractionLogger` uses it as a lookup key

**Example Usage:**
```python
from dardcollect.pipeline_loggers import ImagePersonDetectionLogger

detection_logger = ImagePersonDetectionLogger(
    output_dir="DARD/extracted_image_detections",
    downloads_csv_path="DARD/archive_org_public_domain/downloads.csv",
)

detection_logger.log_image_detection(
    source_image_path="/path/to/photo.jpg",
    num_persons=3,
    detector_model="yolox_tiny",
    detector_confidence=0.875,
    output_path="/path/to/photo.json",
)

detection_logger.print_summary()
```

### Image Face Crops Extraction Logger
**File:** `DARD/face_crops/image_face_crops_extraction.csv`

Tracks 616×616 OFIQ-aligned face crop extraction from static images (script: `extract_face_crops_from_images.py`).

**Columns:**
```
uuid, detection_uuid, timestamp, source_image_path, face_bbox, confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `detection_uuid`: UUID of the parent row in `image_person_detection.csv`

**Example Usage:**
```python
from dardcollect.pipeline_loggers import ImageFaceCropsExtractionLogger

crop_logger = ImageFaceCropsExtractionLogger(
    output_dir="DARD/face_crops",
    image_detection_csv_path="DARD/extracted_image_detections/image_person_detection.csv",
)

crop_logger.log_face_crop_extraction(
    source_image_path="/path/to/photo.jpg",
    face_bbox="100,50,200,150",
    confidence=0.95,
    output_path="/path/to/photo_face_0.jpg",
)

crop_logger.print_summary()
```

### Audio Transcriptions Extraction Logger
**File:** `DARD/audio_transcriptions/audio_transcriptions_extraction.csv`

Tracks transcriptions extracted from standalone audio files (script: `transcribe_audio_files.py`).

**Columns:**
```
uuid, download_uuid, timestamp, source_audio_path,
language_detected, confidence, duration_seconds, model_version, output_path
```

- `uuid`: row identifier (UUID4)
- `download_uuid`: UUID of the parent row in `downloads.csv`

**Example Usage:**
```python
from dardcollect.pipeline_loggers import AudioTranscriptionsExtractionLogger

audio_logger = AudioTranscriptionsExtractionLogger(
    output_dir="DARD/audio_transcriptions",
    downloads_csv_path="DARD/archive_org_public_domain/downloads.csv",
)

audio_logger.log_audio_transcription(
    source_audio_path="/path/to/speech.mp3",
    language_detected="en",
    confidence=1.0,
    duration_seconds=125.5,
    model_version="small",
    output_path="/path/to/speech.transcription.json",
)

audio_logger.print_summary()
```

### Document Text Extraction Logger
**File:** `DARD/preprocessed_documents/document_text_extraction.csv`

Tracks text extraction from documents (PDF, TXT) (script: `extract_text_from_doc.py`).

**Columns:**
```
uuid, download_uuid, timestamp, source_document_path,
text_length, word_count, model_version, output_annotation_path, output_text_path
```

- `uuid`: row identifier (UUID4)
- `download_uuid`: UUID of the parent row in `downloads.csv`

**Example Usage:**
```python
from dardcollect.pipeline_loggers import DocumentTextExtractionLogger

doc_logger = DocumentTextExtractionLogger(
    output_dir="DARD/preprocessed_documents",
    downloads_csv_path="DARD/archive_org_public_domain/downloads.csv",
)

doc_logger.log_text_extraction(
    source_document_path="/path/to/manifest.pdf",
    text_length=5234,
    word_count=842,
    model_version="pdfplumber",
    output_annotation_path="/path/to/manifest.annotation.json",
    output_text_path="/path/to/manifest.text.txt",
)

doc_logger.print_summary()
```

---

## 15. Custom Data Sources (Non-Archive.org Workflows)

The traceability chain always starts at `downloads.csv` — the root that gives
every source file a UUID. All downstream loggers (`FaceCropsExtractionLogger`,
`AudioTranscriptionsExtractionLogger`, etc.) receive this CSV as
`downloads_csv_path` and use it to resolve `download_uuid` for each row they
write.

When your data does **not** come from Archive.org, use `register_source_files()`
to create an equivalent manifest CSV before running any pipeline stage:

```python
from dardcollect import register_source_files

# Creates (or appends to) a downloads.csv-compatible manifest
register_source_files(
    input_dir="my_dataset/videos/",
    output_csv="my_dataset/downloads.csv",
    media_type="video",
    extra_metadata={"dataset": "MyDataset2024", "license": "CC-BY-4.0"},
)
```

**CSV schema produced:**

```
uuid, archive_org_identifier, filename_downloaded, media_type, registered_at, source_path, [extra columns]
```

- `archive_org_identifier` is left empty (schema compatibility only).
- `filename_downloaded` is the lookup key used by all downstream loggers.
- `registered_at` is an ISO 8601 UTC timestamp.
- `source_path` is the absolute path to the original file.
- Any `extra_metadata` columns are appended after the fixed columns.

Pass the manifest path as `downloads_csv_path` to any logger that accepts it,
and the rest of the traceability chain works identically to the Archive.org
pipeline. See [docs/5-LIBRARY-API.md](5-LIBRARY-API.md) for a complete example.

---

## 16. References

- [FAIR Data Principles](https://www.go-fair.org/fair-principles/)
- [W3C PROV Ontology](https://www.w3.org/TR/prov-overview/)
- [Dublin Core Metadata Initiative](https://dublincore.org/)
- [JSON-LD Specification](https://json-ld.org/)

---

← [Back to README](../README.md)
