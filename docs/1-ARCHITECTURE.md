# DARDcollect — Architecture & Workflow

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Archive.org Public Domain                        │
│              (videos, images, audio, documents)                     │
└────────────────┬──────────────────┬──────────────┬──────────────────┘
                 │                  │              │
         ┌───────▼─────────┬────────▼─────┐  ┌────▼────────┐
         │  VIDEO PIPELINE │ IMAGE PIPELINE│  │ AUDIO + DOC │
         └─────┬───────────┴─────┬────────┘  │  PIPELINES  │
               │                 │           └──────┬──────┘
         ┌─────▼─────────────────▼──────┐           │
         │  Person Detection + Pose Kpts │           │
         │  (YOLOX-tiny + CigPose-m)    │  ┌────────▼────────┐
         └─────┬──────────────────┬─────┘  │  Transcriptions │
               │                  │        │  (OpenAI Whisper│
         ┌─────▼─────┐      ┌─────▼────┐  │   small model)  │
         │  CLIPS    │      │  IMAGES  │  └─────────┬────────┘
         │ (person   │      │(person   │            │
         │ detection)│      │detection)│            │
         └─────┬─────┘      └─────┬────┘            │
               │                  │                 │
         ┌─────▼──────────────────▼─────────────────▼────┐
         │   FACE CROP EXTRACTION (616×616 OFIQ)        │
         │  (Affine transform → normalized OFIQ crops)  │
         └─────┬──────────────────────────────────┬──────┘
               │                                  │
         ┌─────▼─────────────────────────┐  ┌────▼──────────┐
         │   QUALITY ANNOTATION (OFIQ)   │  │ TEXT EXTRACTION
         │  (7 dimensions + MagFace)     │  │ (PDF/TXT docs)
         └─────┬──────────────────┬──────┘  └────┬──────────┘
               │                  │              │
         ┌─────▼──────────────────▼──────────────▼────┐
         │         TRACEABILITY CSV SYSTEM (FAIR)     │
         │   Links every artifact to its source via   │
         │   10 incremental CSV files (see logging.md)│
         └──────────────────────────────────────────┘
```

## Key Components

### Modality Pipelines

| Modality | Input | Detection | Output | Logger |
|----------|-------|-----------|--------|--------|
| **Video** | MP4 from Archive.org | YOLOX (person), CigPose (pose) | Person clips (MP4) | `ExtractionLogger` |
| | Clips | Face detection, alignment | 616×616 OFIQ crops | `FaceCropsExtractionLogger` |
| | Clips | Whisper transcription | JSON sidecars | `TranscriptionsExtractionLogger` |
| **Image** | JPG/PNG from Archive.org | YOLOX (person), CigPose (pose) | Person detections (JSON) | `ImagePersonDetectionLogger` |
| | Images + detections | Face crop extraction | 616×616 OFIQ crops | `ImageFaceCropsExtractionLogger` |
| **Audio** | MP3/WAV files | Whisper transcription | JSON sidecars | `AudioTranscriptionsExtractionLogger` |
| **Document** | PDF/TXT files | pdfplumber/OCR | Text + annotation JSON | `DocumentTextExtractionLogger` |
| **Annotation** | All face crops | OFIQ 7-dim + MagFace | Quality JSON sidecars | `FaceQualityAnnotationLogger` |
| | Quality crops | MagFace threshold filter | Filtered crops | `FilteredFaceCropsLogger` |

### Detection Models

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **YOLOX-tiny** | Person bounding box detection | Image/frame | Bboxes + confidence |
| **CigPose-m (COCO-Wholebody)** | 133 keypoint pose estimation | Image/frame + bbox | Keypoints + scores (face, body, hands) |
| **OpenAI Whisper (small)** | Audio transcription | MP3/WAV or video audio | Text + language |
| **MagFace (IResNet50)** | Face quality/embedding | 112×112 crop (ArcFace) | Quality score ∈ [0,1] |
| **OFIQ (7D)** | ISO/IEC 29794-5 quality | 616×616 crop | unified_score, sharpness, compression_artifacts, expression_neutrality, no_head_coverings, face_occlusion_prevention, head_pose |

### FAIR Compliance Strategy: Findability, Accessibility, Interoperability, Reusability

Every artifact gets **FAIR metadata** enabling reproducibility and interoperability:

#### What's Tracked

| Aspect | How It's Implemented |
|--------|---------------------|
| **Findability** | UUID v4 for every artifact (clip, crop, transcription, quality annotation) — enables permanent linking and citation |
| **Accessibility** | All data in open formats (MP4, JSON, CSV) — no lock-in to proprietary tools or external registries |
| **Interoperability** | Standard formats (ISO 8601 timestamps, JSON schemas, Dublin Core metadata) — enables integration with other tools |
| **Reusability** | Complete provenance chain from Archive.org → download → clip → crop → quality scores; source attribution always preserved |
| **Schema Versioning** | Every sidecar includes `schema_version` (e.g., `"1.0"`) — enables format evolution without breaking existing tools |
| **Automatic Validation** | `jsonschema` validates all sidecars during write; invalid sidecars raise detailed errors immediately |

#### Example Sidecar Chain

```
Person Clip (UUID: 550e8400...)
  ├─ Source: Archive.org ID + URL + License
  ├─ [transcription parent ref] → Transcription (UUID: 550e8400..., parent_clip.uuid: 550e8400...)
  └─ [face crop parent ref] → Face Crop (UUID: 550e8401..., parent_clip.uuid: 550e8400...)
       └─ [quality parent ref] → Quality Annotation (UUID: 550e8402..., parent_crop.uuid: 550e8401...)
```

Complete chain of custody: **Archive.org → Download UUID → Video file → Clip UUID → Crop UUID → Quality scores**  
Every link preserved in CSV + JSON, no external registry needed.

See [docs/2-LOGGING.md](2-LOGGING.md) for CSV schemas and traceability queries. See [docs/3-ANNOTATIONS.md](3-ANNOTATIONS.md) for sidecar JSON structure details.

## Data Flow Example: "Finger Man" Video

```
1. Download from Archive.org
   └─> DARD/archive_org_public_domain/videos/fingerDance1956.mp4
       + dataset.csv entry (uuid, archive_id, title, creator, date, license)

2. Extract Person Clips
   └─> Detect persons, slice into clips
       DARD/extracted_person_clips/fingerDance1956/fingerDance_00m12s-00m15s.mp4
       + clips_extraction.csv (clip_id, source_video, num_persons, etc.)

3a. Extract Face Crops (from clips)
   └─> 616×616 OFIQ crops per person per clip
       DARD/extracted_face_crops/fingerDance_00m12s-00m15s_face_0.mp4
       + face_crops_extraction.csv (crop_id, source_clip, bbox, confidence)

3b. Transcribe Clips
   └─> Whisper transcription
       DARD/transcriptions/fingerDance_00m12s-00m15s.transcription.json
       + transcriptions_extraction.csv (trans_id, language, word_count)

4. Annotate Face Quality
   └─> OFIQ 7D scores for each crop
       DARD/extracted_face_crops/fingerDance_00m12s-00m15s_face_0.json
       + face_quality_annotation.csv (crop_id, sharpness, illumination, ...)

5. Filter High-Quality Crops
   └─> Keep crops with overall_score ≥ threshold
       DARD/filtered_face_crops/fingerDance_00m12s-00m15s_face_0.mp4
       + filtered_face_crops.csv (crop_id, magface_score, filter_threshold)
```

**Complete Chain of Custody:**
- Archive.org ID → Download UUID → Video file → Clip ID → Crop ID → Quality scores
- Each step recorded in CSV with timestamps

## Modality Specifics

### Video Workflow
1. Person clips extracted via sliding window + detection
2. Each clip can spawn multiple face crops (one per detected person)
3. Clips are transcribed independently
4. Crops undergo OFIQ + MagFace quality assessment

### Image Workflow
1. Person detection sidecar written next to image
2. Face crops extracted using pose keypoints (same OFIQ alignment as video)
3. Crops skip transcription (no audio in images)
4. Crops still undergo full quality assessment

### Audio Workflow
1. Transcription only (no face crops)
2. Language detected by Whisper
3. Metadata includes duration, word count, confidence

### Document Workflow
1. **PDF:** pdfplumber extracts the embedded text layer
   - If ≥ 100 characters extracted → done (`method=text_layer`)
   - If < 100 characters → OCR fallback: PyMuPDF renders each page to a numpy array (150 DPI, in-memory), PaddleOCR ONNX runs detection + recognition (`method=ocr_paddleocr`)
2. **TXT:** direct UTF-8 read (`method=native`)
3. Character + word count tracked; documents below `min_text_length` (50 chars) discarded
4. `.text.txt` + `.annotation.json` sidecar written per document

---

See [docs/2-LOGGING.md](2-LOGGING.md) for CSV schema details and traceability queries.

← [Back to README](../README.md)
