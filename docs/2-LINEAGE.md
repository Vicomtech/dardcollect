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
  - [Quality Annotations](#7-quality-annotation-sidecars-json)
  - [Colour Classification](#7b-colour-classification-log-csv--standalone-stage)
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
| **4. Face Crops (video)** | `video_face_crops/video_face_crops_extraction.csv` | Face regions from person clips | clips_extraction.csv |
| **4. Face Crops (image)** | `image_face_crops/image_face_crops_extraction.csv` | Face regions from static images | image_person_detection.csv |
| **5. Transcriptions (video)** | `extracted_person_clips/transcriptions_extraction.csv` | Speech transcribed from person clips | clips_extraction.csv |
| **5. Transcriptions (audio)** | `audio_transcriptions/audio_transcriptions_extraction.csv` | Speech transcribed from audio files | downloads.csv |
| **6. Image Detection** | `extracted_image_detections/image_person_detection.csv` | Person detections in static images | downloads.csv |
| **7. Quality Filter (video)** | `filtered_video_face_crops/video_filtered_face_crops.csv` | High-quality video crops after MagFace filtering | video_face_crops_extraction.csv |
| **7. Quality Filter (image)** | `filtered_image_face_crops/image_filtered_face_crops.csv` | High-quality image crops after MagFace filtering | image_face_crops_extraction.csv |
| **8. Quality Annotation (video)** | `video_face_crops/*_face_N.ofiq_attr.json` (JSON sidecar) | OFIQ 7-dimension quality scores for video crops | video_face_crops_extraction.csv |
| **8. Quality Annotation (image)** | `image_face_crops/*_face_N.ofiq_attr.json` (JSON sidecar) | OFIQ 7-dimension quality scores for image crops | image_face_crops_extraction.csv |
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
frames_extraction  video_face_crops_extraction  transcriptions_extraction
     │               ↓
     │    filter_face_crops_by_quality
     │               ↓
     │    video_filtered_face_crops.csv
     │               ↓
      │    annotate_face_quality
      │               ↓
      │    *_face_N.ofiq_attr.json (+ .magface.json)
      └→ (frames available for any downstream task)
```

**Trace any artifact to its source:**
```bash
# 1. Find a face crop's source clip
grep "crop_xyz" DARD/video_face_crops/video_face_crops_extraction.csv
# → source_clip: "Finger_Man_02m09s-02m12s.mp4"

# 2. Find that clip's source video
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/clips_extraction.csv
# → source_video: "Finger Man (1955).mp4"

# 3. Find the download record
grep "Finger Man (1955).mp4" DARD/archive_org_public_domain/downloads.csv
# → UUID + creator, license, download timestamp

# Complete chain: Source manifest (Archive.org item or custom-registered source) → Video → Clip → Face Crop → Quality Annotation
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

**Traceability Path:** Source manifest (Archive.org item ID or custom-registered source row) → Download UUID → Video filename → Extracted clips → Detection metadata

Each artifact is uniquely identifiable and can be traced back to its source through the CSV files and JSON sidecars.

---

## CSV Schemas

## 1. Download Manifest (downloads.csv)

**File:** `DARD/archive_org_public_domain/downloads.csv`

Records all media files downloaded from Archive.org. This is the **starting point** of the complete traceability chain.

**Schema:** dynamic — columns grow as new Archive.org metadata fields are encountered across items.

Fixed pipeline fields (always present, always first):
```
uuid, title, creator, date, license, archive_org_identifier, filename_downloaded,
media_type, downloaded_at, download_stage_script, download_stage_timestamp
```

`title`/`creator`/`date`/`license` are Dublin Core Terms columns (from the item's
`title`/`creator`/`date`/`licenseurl` metadata); the sidecars' shared JSON-LD
`@context` maps these same keys to `dct:title`/`dct:creator`/`dct:date`/`dct:license`
(see [FAIR Compliance](#9-fair-compliance)).

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
| `mediatype` | `media_type` | Archive.org's own type label vs. the pipeline's classification from `configs/config.archive_all.yaml` |

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

**File:** `DARD/video_face_crops/video_face_crops_extraction.csv`

Tracks face regions extracted from person clips or images (critical for linking crops to original sources).

**Columns:**
```
uuid, parent_uuid, timestamp, crop_id, source_type, source_path, face_bbox, confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `parent_uuid`: UUID of the parent row — in `clips_extraction.csv` (when `source_type="person_clip"`) or in `downloads.csv` (when `source_type="image"`)
- `crop_id`: derived from output filename stem; kept in CSV because downstream loggers (`FilteredFaceCropsLogger`) use it as a lookup key
- `source_type`: `"person_clip"` or `"image"`
- `face_bbox`: bounding box as `"x1,y1,x2,y2"` in source-frame coordinates

**Key characteristics:**
- ✅ **parent_uuid** links to upstream CSV (clips or downloads) without filename matching
- ✅ **source_type**: enables cross-media traceability
- ✅ **crop_id**: lookup key used by the quality-filter logger

**Usage:**
```bash
# Find all face crops from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/video_face_crops/video_face_crops_extraction.csv

# Count crops by source type (clip vs. image)
awk -F',' 'NR>1 {print $6}' DARD/video_face_crops/video_face_crops_extraction.csv | sort | uniq -c
```

---

## 5. Transcriptions Extraction Log (CSV)

**File:** `DARD/extracted_person_clips/transcriptions_extraction.csv`

Logs transcriptions extracted from person clips (speech-to-text with language detection). This CSV is a
**lean join index**: the authoritative payload (transcription text, per-segment timestamps) lives in the
`*.transcription.json` sidecar; the CSV holds only the linkage + lookup metrics.

**Columns:**
```
uuid, clip_uuid, timestamp, source_clip_path, language_detected,
word_count, model_version, output_path
```

- `uuid`: row identifier (UUID4)
- `clip_uuid`: UUID of the parent row in `clips_extraction.csv`

**Key characteristics:**
- ✅ `clip_uuid` links to clips_extraction.csv (direct UUID join)
- ✅ Language detection + word count (lookup metrics)
- ✅ Model version (reproducibility)
- ℹ️ No `confidence`/`duration_seconds` columns: Whisper provides no per-clip
  confidence, and duration is authoritative in the sidecar (`duration_seconds`) —
  the CSV never duplicates sidecar-owned values.

**Usage:**
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count transcriptions by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c
```

---

## 6. Filtered Face Crops Log (CSV)

**File:** `DARD/filtered_video_face_crops/video_filtered_face_crops.csv` (or `image_filtered_face_crops.csv` for images)

Tracks face crops that pass quality filtering (MagFace score ≥ threshold), linking back to original crops.

**Columns:**
```
uuid, crop_uuid, timestamp, source_crop_path, magface_score, filter_threshold, output_path
```

- `uuid`: row identifier (UUID4)
- `crop_uuid`: UUID of the parent row in `video_face_crops_extraction.csv`

**Key characteristics:**
- ✅ `crop_uuid` links to video_face_crops_extraction.csv (direct UUID join)
- ✅ MagFace score (quality metric, calibrated [0, 100])
- ✅ Filter threshold recorded for reproducibility

**Usage:**
```bash
# Count crops that passed filter
tail -n +2 DARD/filtered_video_face_crops/video_filtered_face_crops.csv | wc -l

# Average MagFace score of filtered crops
awk -F',' 'NR>1 {sum+=$5; count++} END {print "Avg MagFace: " sum/count}' \
  DARD/filtered_video_face_crops/video_filtered_face_crops.csv
```

---

## 7. Quality Annotation Sidecars (JSON)

**File:** `DARD/video_face_crops/VideoTitle_face_N.ofiq_attr.json` (or `image_face_crops/ImageName_face_N.ofiq_attr.json`), plus a sibling `.magface.json` with MagFace unified_score aggregates.

Quality annotations are stored as **JSON sidecars** next to each face crop (not as CSV rows). Each `.ofiq_attr.json` carries the OFIQ 7 dimensions (unified_score, sharpness, compression_artifacts, expression_neutrality, no_head_coverings, face_occlusion_prevention, head_pose) as aggregates plus a per-frame `frame_data` array. Schema: `schemas/quality_annotation_schema.json` (validated at write time).

**Provenance fields (inside the sidecar):**
```
uuid, schema_version, parent_crop {uuid, file},
face_crop_video, face_crop_json, source_video,
annotated_at, annotator, frame_stride, max_frames_sampled
```

- `uuid`: sidecar identifier (UUID4), schema-validated at write
- `parent_crop.uuid`: UUID of the parent face-crop sidecar (FAIR join)
- `source_video`: original person clip (or image) the crop came from
- `frame_stride` / `max_frames_sampled`: sampling parameters actually used

**Key characteristics:**
- ✅ `parent_crop.uuid` links to the face-crop sidecar (direct UUID join; the crop sidecar in turn links to its clip via `parent_clip`)
- ✅ All 7 OFIQ scalar measures + 3 head-pose quality scores
- ✅ `unified_score` matches the MagFace score used in `video_filtered_face_crops.csv` but with full per-crop stats and per-frame values in the sidecar

**Usage:**
```bash
# Crops with high sharpness (jq)
jq '.sharpness.max' DARD/video_face_crops/VideoTitle_face_0.ofiq_attr.json

# Average unified_score across all crops of one video
jq -s '[.[].unified_score.mean] | add/length' DARD/video_face_crops/VideoTitle_face_*.ofiq_attr.json
```

### 7b. Colour Classification Log (CSV) — standalone stage

**File:** `DARD/archive_org_public_domain/videos/color_classification.csv` (next to the classified videos)

Produced by the standalone stage `pipeline/filter_videos_by_color.py` (issue #11, not wired into the
orchestrator DAG). Classifies videos colour vs black-and-white by pixel content — archive.org's
`color` metadata tag is unreliable.

**Columns:**
```
uuid, timestamp, video_path, video_name, classification, mean_saturation, frames_sampled, moved, moved_to
```

- `classification`: `color` | `black_and_white` | `unreadable` (probe/decode failure)
- `mean_saturation`: mean HSV saturation over sampled frames (threshold 12.0, calibrate per corpus)
- `moved` / `moved_to`: set when `--move` relocated a B&W video — the recorded rows make the move reversible

**Key characteristics:**
- ✅ Resumable: one row per video; reruns skip already-classified videos
- ✅ `--move` is opt-in and reversible (undo replays the recorded `moved_to` → `video_path` rows)
- ✅ ffmpeg keyframe fast path (`-skip_frame nokey`) with logged OpenCV fallback

---

## 8. How to Trace Artifacts Through the Complete Pipeline

### Scenario 1: Trace a Face Crop Back to Its Original Video

**You have:** `crop_00042.jpg` (a face crop file)  
**You want to know:** Which original Archive.org video did this come from?

**Steps:**
```bash
# 1. Find the crop in video_face_crops_extraction.csv
grep "crop_00042" DARD/video_face_crops/video_face_crops_extraction.csv
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
# Find crops with low sharpness (sharpness.max inside each sidecar)
for f in DARD/video_face_crops/*_face_*.ofiq_attr.json; do
  jq -r --arg f "$f" 'select(.sharpness.max < 50) | $f' "$f"
done | head -5

# For a specific crop, find its source clip
# Crop names encode the parent clip: "Finger_Man_02m09s-02m12s_face_0" → clip "Finger_Man_02m09s-02m12s"
CROP_STEM="Finger_Man_02m09s-02m12s_face_0"
# source_path is col 6 in video_face_crops_extraction.csv
grep "$CROP_STEM" DARD/video_face_crops/video_face_crops_extraction.csv | cut -d',' -f6
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
- **Multiple formats:** CSV (traceability index), JSON sidecars (per-artifact records)
- **Documentation:** Complete technical spec + quick reference guide
- **License tracking:** Original source license preserved in downloads.csv (`licenseurl` + the Dublin Core `license` column)
- **No lock-in:** Open formats, no proprietary codecs

### 9.3 Interoperable
- **Standard formats:** CSV, JSON, ISO 8601 timestamps
- **JSON-LD context:** Every sidecar carries a shared `@context` mapping its keys
  to [Dublin Core Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/)
  (`uuid` → `dct:identifier`, `title` → `dct:title`, `creator` → `dct:creator`,
  `license` → `dct:license`, …) and the `parent_clip`/`parent_crop`/`parent_audio`
  links to [PROV-O](https://www.w3.org/TR/prov-o/) `prov:wasDerivedFrom` — so each
  sidecar parses as JSON-LD linked data with no transformation. (CSVs are plain
  tables: they carry the same DC-named columns, but no `@context`.)
- **Metadata schemas:** Every sidecar is validated at write time against its JSON
  Schema in `schemas/` (`person_clip_schema.json`, `face_crop_schema.json`,
  `transcription_schema.json`, `quality_annotation_schema.json`,
  `image_detection_schema.json`, `document_schema.json`)
- **Cross-system links:** UUIDs enable integration with other databases

### 9.4 Reusable
- **Complete provenance:** From Archive.org ID through extraction to final product
- **Processing details:** Model names, versions, detector confidence documented
- **Source attribution:** Original creator, date, license always preserved
- **Reconstruction capability:** URLs stored, can regenerate data if needed

**CSVs vs sidecars (why both exist):** each CSV is a lean *join index* — identity
(`uuid`), a parent link (`parent_uuid`/`clip_uuid`/`download_uuid`/`detection_uuid`/
`crop_uuid`), and a few stage-specific lookup keys/metrics, appended incrementally so
an interrupted stage resumes without loss. The full per-artifact payload (segments,
per-frame data, quality measures, provenance objects) lives in the schema-validated
JSON sidecars, which are the authoritative record. Downstream stages join on the CSV
index (one lookup per row) instead of scanning every sidecar.

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
    │   DARD/video_face_crops/[Finger_Man_02m09s-02m12s_face_0.mp4]
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
    FilteredFaceCropsLogger,
)

# Initialize loggers for your stage
logger = ExtractionLogger(output_dir="DARD/extracted_person_clips")  # For clips
frames_logger = FramesExtractionLogger(output_dir="DARD/extracted_person_clips")  # For frames
face_crops_logger = FaceCropsExtractionLogger(output_dir="DARD/extracted_person_clips")  # For face crops
trans_logger = TranscriptionsExtractionLogger(output_dir="DARD/extracted_person_clips")  # For transcriptions
# Quality scores: write *_face_N.ofiq_attr.json sidecars via dardcollect.quality (see docs/3-ANNOTATIONS.md §3)
filter_logger = FilteredFaceCropsLogger(output_dir="DARD/filtered_video_face_crops")  # For filtered crops
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
    output_dir="DARD/video_face_crops",
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
    word_count=len(words),
    output_path=str(output_json_path),
    model_version="whisper-small",
)

trans_logger.print_summary()
```

### Example: `filter_face_crops_by_quality.py`

```python
from dardcollect.pipeline_loggers import FilteredFaceCropsLogger

face_crops_csv = input_dir / "video_face_crops_extraction.csv"
filter_logger = FilteredFaceCropsLogger(
    output_dir="DARD/filtered_video_face_crops",
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

Quality annotation writes JSON sidecars (not CSV rows) — see
[docs/3-ANNOTATIONS.md §3](3-ANNOTATIONS.md#3-quality-annotations-face-quality-scores).
Each `*_face_N.ofiq_attr.json` is FAIR-validated at write time and links to its
parent crop via `parent_crop.uuid`, which resolves to the row keyed by `crop_id`
in `video_face_crops_extraction.csv` / `image_face_crops_extraction.csv`.

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
grep "Finger_Man_02m09s-02m12s" DARD/video_face_crops/video_face_crops_extraction.csv

# Count crops by source type (clip vs. image)
awk -F',' 'NR>1 {print $5}' DARD/video_face_crops/video_face_crops_extraction.csv | sort | uniq -c

# Average detection confidence
awk -F',' 'NR>1 {sum+=$8; count++} END {print "Avg: " sum/count}' \
  DARD/video_face_crops/video_face_crops_extraction.csv
```

**Transcriptions** — columns: `uuid(1) clip_uuid(2) timestamp(3) source_clip_path(4) language_detected(5) word_count(6) ...`
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c
```

**Quality annotations** — JSON sidecars `*_face_N.ofiq_attr.json` next to each crop
```bash
# Find low-quality crops (sharpness.max inside each sidecar)
for f in DARD/video_face_crops/*_face_*.ofiq_attr.json; do
  jq -r --arg f "$f" 'select(.sharpness.max < 50) | $f' "$f"
done

# Crops whose unified_score never crossed a threshold
for f in DARD/video_face_crops/*_face_*.ofiq_attr.json; do
  jq -r --arg f "$f" 'select(.unified_score.max < 15) | $f' "$f"
done
```

### Complex Tracing Queries

**Trace a face crop through all stages:**
```bash
# Given a crop_id, find its complete journey
CROP_ID="crop_00042"

echo "=== Face Crop Journey ==="
echo "1. Original crop extraction:"
grep "$CROP_ID" DARD/video_face_crops/video_face_crops_extraction.csv

echo -e "\n2. Source clip (col 6 = source_path):"
CLIP_STEM=$(grep "$CROP_ID" DARD/video_face_crops/video_face_crops_extraction.csv | cut -d',' -f6 | xargs basename | sed 's/\..*//')
grep "$CLIP_STEM" DARD/extracted_person_clips/clips_extraction.csv

echo -e "\n3. Source video (col 5 = source_video):"
VIDEO=$(grep "$CLIP_STEM" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f5)
grep "$VIDEO" DARD/archive_org_public_domain/downloads.csv

echo -e "\n4. Quality annotation (if any):"
cat "DARD/video_face_crops/${CROP_ID}.ofiq_attr.json"

echo -e "\n5. Filtered status (if passed):"
grep "$CROP_ID" DARD/filtered_video_face_crops/video_filtered_face_crops.csv
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
grep -E "$CLIPS" DARD/video_face_crops/video_face_crops_extraction.csv | cut -d',' -f1 | wc -l

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
# per-crop aggregates live in the sidecars next to each crop
for f in DARD/video_face_crops/*_face_*.ofiq_attr.json; do
  jq -r '"\(.face_crop_video): sharpness=\(.sharpness.max), unified_score=\(.unified_score.max)"' "$f"
done | head -20
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
- `source_image`: source image filename (derived from `source_image_path`, used as CSV join key)

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
**File:** `DARD/image_face_crops/image_face_crops_extraction.csv`

Tracks 616×616 OFIQ-aligned face crop extraction from static images (script: `extract_face_crops_from_images.py`).

**Columns:**
```
uuid, detection_uuid, timestamp, source_image_path, bbox_in_source, bbox_confidence, output_path
```

- `uuid`: row identifier (UUID4)
- `detection_uuid`: UUID of the parent row in `image_person_detection.csv`
- `bbox_in_source`: person bbox `"x1,y1,x2,y2"` in source-image coordinates — same
  values as the sidecar's `bbox_in_source` (the CSV column deliberately shares the
  sidecar field name so both formats agree)

**Example Usage:**
```python
from dardcollect.pipeline_loggers import ImageFaceCropsExtractionLogger

crop_logger = ImageFaceCropsExtractionLogger(
    output_dir="DARD/image_face_crops",
    image_detection_csv_path="DARD/extracted_image_detections/image_person_detection.csv",
)

crop_logger.log_face_crop_extraction(
    source_image_path="/path/to/photo.jpg",
    bbox_in_source="100,50,200,150",
    bbox_confidence=0.95,
    output_path="/path/to/photo_face_0.jpg",
)

crop_logger.print_summary()
```

### Audio Transcriptions Extraction Logger
**File:** `DARD/audio_transcriptions/audio_transcriptions_extraction.csv`

Tracks transcriptions extracted from standalone audio files (script: `transcribe_audio_files.py`).
Lean join index — text + segments live in the `.transcription.json` sidecar.

**Columns:**
```
uuid, download_uuid, timestamp, source_audio_path,
language_detected, model_version, output_path
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
uuid, title, creator, date, license, archive_org_identifier, filename_downloaded,
media_type, registered_at, source_path, [extra columns]
```

- `archive_org_identifier` is left empty (schema compatibility only).
- `title`/`creator`/`date`/`license` are Dublin Core Terms columns; `extra_metadata`
  values for those keys override the defaults (title defaults to the file stem).
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
