# 📊 DARDcollect Data Traceability & Lineage

**Version:** 1.0  
**Last Updated:** 2026-05-07  
**Compliance:** FAIR Data Principles (Findable, Accessible, Interoperable, Reusable)

---

## 🚀 Quick Start (5 min read)

**CSV files track everything through the entire extraction pipeline:**

| Stage | CSV (co-located with output) | Purpose | Links To |
|-------|------------------------------|---------|----------|
| **1. Download** | `archive_org_public_domain/downloads.csv` | Archive.org downloads with UUID + metadata | — |
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

**Key characteristics:**
- ✅ **Unique identifier (UUID):** Every download gets a uuid4 for permanent identification
- ✅ **Complete Archive.org metadata:** All `item.metadata` fields captured — not a manual subset
- ✅ **License tracking:** `licenseurl` field documents original source license
- ✅ **Download timestamp:** ISO 8601 UTC format for audit trail
- ✅ **Media classification:** video|audio|image|text for organization
- ✅ **No redundancy:** Archive.org's `identifier` field is stored as `archive_org_identifier`; the item URL is derivable from it and not stored separately

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
timestamp,clip_id,source_video,start_frame,end_frame,start_seconds,duration_seconds,num_persons,detector_model,detector_confidence,output_path
```

**Example:**
```csv
2026-05-07T15:22:00Z,Finger_Man_02m09s-02m12s,Finger Man (1955).mp4,3270,3360,139.8,3.0,1,yolox-tiny,0.952,DARD/extracted_person_clips/Finger Man (1955)_02m09s-02m12s.mp4
2026-05-07T15:40:21Z,The_Crooked_Web_00m57s-01m30s,The Crooked Web (1955).mp4,1368,2160,57.0,33.1,2,yolox-tiny,0.918,DARD/extracted_person_clips/The Crooked Web (1955)_00m57s-01m30s.mp4
```

**Key characteristics:**
- ✅ **Incremental writes:** One row appended per clip (immediate disk write)
- ✅ **Timestamped:** ISO 8601 UTC timestamps for every entry
- ✅ **Source traceability:** Links each clip to its source video (connect to downloads.csv via filename)
- ✅ **Detection metadata:** Includes confidence scores and person counts
- ✅ **Resilient:** Survives process interruptions—nothing is lost
- ✅ **Human-readable:** Simple CSV format, easy to query with grep/awk

**Typical usage:**
```bash
# Count total clips extracted (including incomplete runs)
tail -n +2 clips_extraction.csv | wc -l

# Find all clips from a specific source video
grep "Finger_Man_1955" clips_extraction.csv

# Get duration and person stats per source video
awk -F',' 'NR>1 {src=$3; sum[src]+=$7; count[src]++; persons[src]+=$8} 
END {for (s in sum) printf "%s: %d clips, %.0f sec, %.1f persons/clip\n", s, count[s], sum[s], persons[s]/count[s]}' \
  clips_extraction.csv
```

---

## 3. Frames Extraction Log (CSV)

**File:** `DARD/extracted_frames/frames_extraction.csv`

Logs individual frames extracted from person clips (typically for detailed face analysis, model training, etc.).

**Columns:**
```
timestamp,frame_id,source_clip,source_clip_path,frame_number,timestamp_seconds,output_path
2026-05-07T15:23:14Z,Finger_Man_02m09s-02m12s_frame_00042,Finger_Man_02m09s-02m12s.mp4,DARD/extracted_person_clips/Finger_Man_02m09s-02m12s.mp4,42,1.68,DARD/extracted_frames/Finger_Man_02m09s-02m12s_frame_00042.png
```

**Key characteristics:**
- ✅ Links to source clip (trace frame origin)
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
timestamp,crop_id,source_type,source_id,source_path,face_bbox,confidence,width,height,output_path
2026-05-07T15:25:02Z,crop_00001,person_clip,Finger_Man_02m09s-02m12s,DARD/extracted_person_clips/Finger_Man_02m09s-02m12s.mp4,"512,320,640,480",0.952,128,160,DARD/extracted_face_crops/crop_00001.jpg
2026-05-07T15:25:15Z,crop_00002,image,archive_image_xyz,DARD/archive_org_public_domain/images/eng/photo.jpg,"256,128,384,256",0.887,128,128,DARD/extracted_face_crops/crop_00002.jpg
```

**Key characteristics:**
- ✅ **source_type**: "person_clip" or "image" (enables cross-media traceability)
- ✅ **source_id** + **source_path**: Links to original artifact
- ✅ **face_bbox**: Coordinates for reproducibility
- ✅ **confidence**: Detection confidence score

**Usage:**
```bash
# Find all face crops from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/face_crops/face_crops_extraction.csv

# Find all crops from images vs. clips
awk -F',' 'NR>1 {print $3}' DARD/face_crops/face_crops_extraction.csv | sort | uniq -c

# Get average detection confidence
awk -F',' 'NR>1 {sum+=$7; count++} END {print "Avg confidence: " sum/count}' \
  DARD/face_crops/face_crops_extraction.csv
```

---

## 5. Transcriptions Extraction Log (CSV)

**File:** `DARD/extracted_person_clips/transcriptions_extraction.csv`

Logs transcriptions extracted from person clips (speech-to-text with language detection and confidence).

**Columns:**
```
timestamp,transcription_id,source_clip,source_clip_path,language_detected,confidence,word_count,duration_seconds,model_version,output_path
2026-05-07T15:30:21Z,trans_001,Finger_Man_02m09s-02m12s.mp4,DARD/extracted_person_clips/Finger_Man_02m09s-02m12s.mp4,en,0.94,156,3.0,whisper-small,DARD/extracted_transcriptions/trans_001.json
```

**Key characteristics:**
- ✅ Links to source clip (trace audio origin)
- ✅ Language detection + confidence
- ✅ Word count + duration (quality metrics)
- ✅ Model version (reproducibility)

**Usage:**
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count transcriptions by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c

# Total words transcribed
awk -F',' 'NR>1 {sum+=$7} END {print "Total words: " sum}' \
  DARD/extracted_person_clips/transcriptions_extraction.csv
```

---

## 6. Filtered Face Crops Log (CSV)

**File:** `DARD/filtered_face_crops/filtered_face_crops.csv`

Tracks face crops that pass quality filtering (MagFace score ≥ threshold), linking back to original crops.

**Columns:**
```
timestamp,crop_id,source_crop_path,magface_score,filter_threshold,output_path
2026-05-07T15:35:42Z,crop_00001,DARD/extracted_face_crops/crop_00001.jpg,42.5,30.0,DARD/filtered_face_crops/crop_00001.jpg
```

**Key characteristics:**
- ✅ Links to source crop (trace quality journey)
- ✅ MagFace score (quality metric)
- ✅ Filter threshold (reproducibility)

**Usage:**
```bash
# Find high-quality crops
grep "40\." DARD/filtered_face_crops/filtered_face_crops.csv  # MagFace ≥ 40

# Count crops that passed filter
tail -n +2 DARD/filtered_face_crops/filtered_face_crops.csv | wc -l

# Average MagFace score of filtered crops
awk -F',' 'NR>1 {sum+=$4; count++} END {print "Avg MagFace: " sum/count}' \
  DARD/filtered_face_crops/filtered_face_crops.csv
```

---

## 7. Face Quality Annotation Log (CSV)

**File:** `DARD/filtered_face_crops/face_quality_annotation.csv`

Logs OFIQ (Open Face Image Quality) scores for face crops (7 dimensions: sharpness, illumination, contrast, structure, completeness, eye_openness, mouth_openness).

**Columns:**
```
timestamp,crop_id,crop_path,sharpness,illumination,contrast,structure,completeness,eye_openness,mouth_openness,overall_score,passed_filter
2026-05-07T15:40:15Z,crop_00001,DARD/extracted_face_crops/crop_00001.jpg,92.3,78.5,81.2,88.9,95.1,86.4,92.1,88.21,true
```

**Key characteristics:**
- ✅ Links to source crop
- ✅ 7 OFIQ quality dimensions (comprehensive quality assessment)
- ✅ Overall score (average of all dimensions)
- ✅ passed_filter: Whether crop meets quality threshold

**Usage:**
```bash
# Find high-quality crops (sharpness ≥ 90)
awk -F',' '$4 >= 90' DARD/filtered_face_crops/face_quality_annotation.csv

# Crops that passed filter
grep "true" DARD/filtered_face_crops/face_quality_annotation.csv | wc -l

# Average quality by dimension
awk -F',' 'NR>1 {
  sharp+=$ 4; illum+=$5; contrast+=$6; struct+=$7; compl+=$8; eye+=$9; mouth+=$10; count++
} 
END {
  printf "Sharpness: %.2f\nIllumination: %.2f\nContrast: %.2f\nStructure: %.2f\nCompleteness: %.2f\nEye Openness: %.2f\nMouth Openness: %.2f\n",
  sharp/count, illum/count, contrast/count, struct/count, compl/count, eye/count, mouth/count
}' DARD/filtered_face_crops/face_quality_annotation.csv
```

---

## 8. Traceability Index (CSV)

**File:** `traceability_index.csv`

Maps every extracted artifact to its source with complete metadata chain:

```csv
artifact_id,artifact_type,artifact_path,source_identifier,source_url,source_media_type,processing_date,processing_version,processor,modality,uuid,file_hash_sha256,file_size_bytes,parents,related_artifacts
```

**Example:**
```csv
Finger_Man_1955_02m09s-02m12s,person_clip,DARD/extracted_person_clips/Finger Man (1955)_02m09s-02m12s.mp4,archive.org:movies/movies__20200210@0_x264.mkv/Finger Man (1955) d8888_512kb.mp4,https://archive.org/download/movies__20200210@0_x264.mkv/Finger%20Man%20%281955%29%20d8888_512kb.mp4,video,2026-05-07T15:22:00Z,1.0,extract_person_clips.py,video,6fb3cd43-809d-4e1d-a54a-2071d3b57805,a3f8c9e2...,2150000,archive.org:Finger_Man_1955,Finger_Man_1955_02m09s-02m12s.json|face_crops_...
```

**Key Advantages:**
- ✅ Unique artifact identification (UUID)
- ✅ Source traceability
- ✅ Processing metadata
- ✅ File integrity (SHA256)
- ✅ Dependency graph (parents/related)

---

## How to Trace Artifacts Through the Complete Pipeline

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

**You have:** `trans_042.json` (transcription)  
**You want to know:** What clip did this come from, and what's its full provenance?

```bash
# Step 1: Find in transcriptions_extraction.csv
grep "trans_042" DARD/extracted_person_clips/transcriptions_extraction.csv | cut -d',' -f3
# → Source clip: "The_Crooked_Web_00m57s-01m30s.mp4"

# Step 2: Get clip metadata
grep "The_Crooked_Web_00m57s-01m30s" DARD/extracted_person_clips/clips_extraction.csv
# → Shows: start frame, end frame, num_persons, detector_confidence

# Step 3: Trace to original video
grep "The Crooked Web" DARD/archive_org_public_domain/downloads.csv
# → Full source metadata
```

### Scenario 3: Link a Quality Annotation to Its Crop and Original Clip

**You have:** A quality annotation with poor scores  
**You want to:** Understand where that crop came from

```bash
# Find crops with low sharpness
awk -F',' '$4 < 50' DARD/filtered_face_crops/face_quality_annotation.csv | head -5

# For a specific crop, find its source clip
CROP_ID="crop_00042"
grep "$CROP_ID" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f4
# → Source clip ID

# Find that clip's source video
CLIP_ID=$(grep "$CROP_ID" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f4)
grep "$CLIP_ID" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f3
# → Source video name
```

---

## 9. FAIR Compliance

### 9.1 Findable
- **Starting point:** UUID for each download (downloads.csv)
- **Extraction tracking:** Timestamp + clip_id for each extraction
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
**Processing:** `scripts/extract_person_clips_from_videos.py`

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
    ├─→ extract_face_crops.py
    │   DARD/extracted_face_crops/[uuid_face_001.jpg]
    └─→ transcribe_clips.py
        DARD/extracted_transcriptions/[uuid_transcription_001.json]
```

---

## 12. File Integrity & Versioning

**Checksums:** All artifacts include SHA256 hash in `traceability_index.csv`  
**Processing versions:** Tracked in `data_lineage.json`  
**Schema versions:** Each artifact JSON includes `schema_version` field

**To verify integrity:**
```bash
sha256sum -c traceability_checksums.txt
```

---

## 13. Accessing Lineage

### Find all artifacts from a source video:
```bash
grep "archive.org:Finger_Man_1955" traceability_index.csv
```

### Find all face crops from a person clip:
```bash
grep "Finger_Man_1955_02m09s-02m12s" traceability_index.csv | grep face_crop
```

### Find processing history:
```bash
jq '.processing_history[] | select(.artifact_id=="...")' data_lineage.json
```

---

## 14. Metadata Schema (JSON-LD Context)

Every artifact should reference this context:

```json
{
  "@context": {
    "@vocab": "http://dard.vicomtech.org/schema/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "prov": "http://www.w3.org/ns/prov#",
    "uuid": "http://purl.org/net/uniqueID#uuid",
    "sha256": "http://purl.org/net/checksum#sha256",
    "source_url": "dcat:source",
    "derived_from": "prov:wasDerivedFrom",
    "generated_by": "prov:wasGeneratedBy"
  }
}
```

---

## 15. Regular Maintenance

- **Weekly:** Verify checksums in `traceability_index.csv`
- **Monthly:** Update `source_manifest.json` with new downloads
- **Per-release:** Version `data_lineage.json` with processing changes

---

## 16. Integration into Your Scripts

### For All Pipeline Stages

Each script should initialize and use the appropriate logger(s):

```python
from persondet.extraction_logger import ExtractionLogger
from persondet.pipeline_loggers import (
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

### Example: `extract_frames.py`

```python
from persondet.pipeline_loggers import FramesExtractionLogger
from pathlib import Path

frames_logger = FramesExtractionLogger(output_dir="DARD/extracted_person_clips")

# For each frame extracted from a clip
frames_logger.log_frame_extraction(
    frame_id=f"{clip_id}_frame_{frame_num:05d}",
    source_clip=clip_filename,
    source_clip_path=str(clip_path),
    frame_number=frame_num,
    timestamp_seconds=frame_num / fps,
    output_path=str(output_frame_path),
)

# At the end
frames_logger.print_summary()
```

### Example: `extract_face_crops.py`

```python
from persondet.pipeline_loggers import FaceCropsExtractionLogger

face_crops_logger = FaceCropsExtractionLogger(output_dir="DARD/extracted_person_clips")

# For each face crop extracted
face_crops_logger.log_face_crop_extraction(
    crop_id=f"crop_{crop_counter:05d}",
    source_type="person_clip",  # or "image"
    source_id=clip_id,
    source_path=str(clip_path),
    face_bbox=f"{x1},{y1},{x2},{y2}",
    confidence=detection_confidence,
    width=crop_width,
    height=crop_height,
    output_path=str(output_crop_path),
)

face_crops_logger.print_summary()
```

### Example: `transcribe_clips.py`

```python
from persondet.pipeline_loggers import TranscriptionsExtractionLogger

trans_logger = TranscriptionsExtractionLogger(output_dir="DARD/extracted_person_clips")

# After transcribing a clip
trans_logger.log_transcription(
    transcription_id=f"trans_{clip_id}",
    source_clip=clip_filename,
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
from persondet.pipeline_loggers import FilteredFaceCropsLogger

filter_logger = FilteredFaceCropsLogger(output_dir="DARD/filtered_face_crops")

# For each crop that passes the MagFace threshold
if magface_score >= threshold:
    filter_logger.log_filtered_crop(
        crop_id=crop_id,
        source_crop_path=str(input_crop_path),
        magface_score=magface_score,
        filter_threshold=threshold,
        output_path=str(output_crop_path),
    )

filter_logger.print_summary()
```

### Example: `annotate_face_quality.py`

```python
from persondet.pipeline_loggers import FaceQualityAnnotationLogger

quality_logger = FaceQualityAnnotationLogger(output_dir="DARD/filtered_face_crops")

# After computing OFIQ scores
overall = sum([sharpness, illumination, contrast, structure, 
               completeness, eye_openness, mouth_openness]) / 7

quality_logger.log_quality_annotation(
    crop_id=crop_id,
    crop_path=str(crop_path),
    sharpness=ofiq_scores["sharpness"],
    illumination=ofiq_scores["illumination"],
    contrast=ofiq_scores["contrast"],
    structure=ofiq_scores["structure"],
    completeness=ofiq_scores["completeness"],
    eye_openness=ofiq_scores["eye_openness"],
    mouth_openness=ofiq_scores["mouth_openness"],
    overall_score=overall,
    passed_filter=(overall >= quality_threshold),
)

quality_logger.print_summary()
```

---

## 17. Querying Traceability Data

### Basic Queries by Stage

**Person clips:**
```bash
# Find all clips from a source video
grep "Finger Man" DARD/extracted_person_clips/clips_extraction.csv

# Count total clips
tail -n +2 DARD/extracted_person_clips/clips_extraction.csv | wc -l

# Average person count per clip
awk -F',' 'NR>1 {sum+=$8; count++} END {print sum/count}' \
  DARD/extracted_person_clips/clips_extraction.csv
```

**Frames:**
```bash
# Find all frames from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_frames/frames_extraction.csv

# Count frames extracted from all clips
tail -n +2 DARD/extracted_frames/frames_extraction.csv | wc -l
```

**Face crops:**
```bash
# Find all crops from a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/face_crops/face_crops_extraction.csv

# Count crops by source type (clip vs. image)
awk -F',' 'NR>1 {print $3}' DARD/face_crops/face_crops_extraction.csv | sort | uniq -c

# Average detection confidence
awk -F',' 'NR>1 {sum+=$7; count++} END {print "Avg: " sum/count}' \
  DARD/face_crops/face_crops_extraction.csv
```

**Transcriptions:**
```bash
# Find transcriptions for a specific clip
grep "Finger_Man_02m09s-02m12s" DARD/extracted_person_clips/transcriptions_extraction.csv

# Count by language
awk -F',' 'NR>1 {print $5}' DARD/extracted_person_clips/transcriptions_extraction.csv | sort | uniq -c
```

**Quality annotations:**
```bash
# Find low-quality crops (sharpness < 50)
awk -F',' '$4 < 50 {print $2}' DARD/filtered_face_crops/face_quality_annotation.csv

# Count passed vs. failed
awk -F',' 'NR>1 {if ($12=="true") passed++; else failed++} 
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

echo -e "\n2. Source clip:"
CLIP_ID=$(grep "$CROP_ID" DARD/face_crops/face_crops_extraction.csv | cut -d',' -f4)
grep "$CLIP_ID" DARD/extracted_person_clips/clips_extraction.csv

echo -e "\n3. Source video:"
VIDEO=$(grep "$CLIP_ID" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f3)
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

echo -e "\n📹 Person clips:"
grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f2

CLIPS=$(grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f2 | tr '\n' '|')

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
CLIPS=$(grep "$VIDEO_NAME" DARD/extracted_person_clips/clips_extraction.csv | cut -d',' -f2 | tr '\n' '|')

echo "=== Quality Analysis for $VIDEO_NAME ==="
awk -F',' -v pattern="$CLIPS" '$2 ~ pattern {
  print $2 ": Sharpness=" $4 ", Illumination=" $5 ", Overall=" $11
}' DARD/filtered_face_crops/face_quality_annotation.csv | head -20
```

---

## 18. Additional Loggers: Images, Audio, Documents

### Image Person Detection Logger
**File:** `DARD/extracted_image_detections/image_person_detection.csv`

Tracks person detections extracted from static images (script: `extract_persons_from_images.py`).

**Columns:**
- `timestamp`: ISO 8601 UTC of detection
- `detection_id`: Unique identifier for this detection batch
- `source_image`: Source image filename
- `source_image_path`: Full path to source image
- `num_persons`: Number of persons detected
- `detector_model`: Model name (e.g., "yolox_tiny")
- `detector_confidence`: Average detection confidence
- `output_path`: Path to detection JSON sidecar

**Example Usage:**
```python
from persondet.pipeline_loggers import ImagePersonDetectionLogger

detection_logger = ImagePersonDetectionLogger(output_dir="DARD/extracted_image_detections")

# After detecting persons in an image
detection_logger.log_image_detection(
    detection_id="img_001",
    source_image="photo.jpg",
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
- `timestamp`: ISO 8601 UTC of extraction
- `crop_id`: Unique crop identifier
- `source_image`: Source image filename
- `source_image_path`: Full path to source image
- `face_bbox`: Bounding box as "x1,y1,x2,y2"
- `confidence`: Face detection confidence in source image
- `width`: Crop width (always 616)
- `height`: Crop height (always 616)
- `output_path`: Path to output .jpg crop

**Example Usage:**
```python
from persondet.pipeline_loggers import ImageFaceCropsExtractionLogger

crop_logger = ImageFaceCropsExtractionLogger(output_dir="DARD/extracted_person_clips")

# After extracting a face crop from an image
crop_logger.log_face_crop_extraction(
    crop_id="img_001_face_0",
    source_image="photo.jpg",
    source_image_path="/path/to/photo.jpg",
    face_bbox="100,50,200,150",
    confidence=0.95,
    width=616,
    height=616,
    output_path="/path/to/photo_face_0.jpg",
)

crop_logger.print_summary()
```

### Audio Transcriptions Extraction Logger
**File:** `DARD/audio_transcriptions/audio_transcriptions_extraction.csv`

Tracks transcriptions extracted from standalone audio files (script: `transcribe_audio_files.py`).

**Columns:**
- `timestamp`: ISO 8601 UTC of transcription
- `transcription_id`: Unique transcription identifier
- `source_audio`: Source audio filename
- `source_audio_path`: Full path to source audio
- `language_detected`: Detected language (e.g., "en", "es")
- `confidence`: Transcription confidence (typically 1.0 for Whisper)
- `duration_seconds`: Audio duration in seconds
- `model_version`: Whisper model size (e.g., "small")
- `output_path`: Path to output .transcription.json

**Example Usage:**
```python
from persondet.pipeline_loggers import AudioTranscriptionsExtractionLogger

audio_logger = AudioTranscriptionsExtractionLogger(output_dir="DARD/extracted_person_clips")

# After transcribing an audio file
audio_logger.log_audio_transcription(
    transcription_id="audio_001",
    source_audio="speech.mp3",
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
- `timestamp`: ISO 8601 UTC of extraction
- `extraction_id`: Unique extraction identifier
- `source_document`: Source document filename
- `source_document_path`: Full path to source document
- `text_length`: Number of characters extracted
- `word_count`: Number of words extracted
- `model_version`: Extraction method (e.g., "pdfplumber", "ocr")
- `output_annotation_path`: Path to output .annotation.json
- `output_text_path`: Path to output .text.txt

**Example Usage:**
```python
from persondet.pipeline_loggers import DocumentTextExtractionLogger

doc_logger = DocumentTextExtractionLogger(output_dir="DARD/extracted_person_clips")

# After extracting text from a document
doc_logger.log_text_extraction(
    extraction_id="doc_001",
    source_document="manifest.pdf",
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

## 19. References

- [FAIR Data Principles](https://www.go-fair.org/fair-principles/)
- [W3C PROV Ontology](https://www.w3.org/TR/prov-overview/)
- [Dublin Core Metadata Initiative](https://dublincore.org/)
- [JSON-LD Specification](https://json-ld.org/)

---

← [Back to README](../README.md)
