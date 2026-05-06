# Annotations Format

This document describes the structure of sidecars and annotations produced by the pipeline. It explains the hierarchy from person clips → face crops → quality annotations, and how data flows through each stage.

---

## FAIR Principles: Data Findability, Accessibility, Interoperability, Reusability

To support reproducibility and interoperability, all dataset sidecars embed FAIR metadata directly:

### Schema Versioning
Every sidecar includes `schema_version` (e.g., `"1.0"`) to document the structure version. Breaking changes increment the major version; backwards-compatible additions increment the minor version.

### Unique Identifiers (UUIDs)
Every data item gets a unique UUID v4 assigned at creation:
- **Person clips**: UUID assigned by `extract_person_clips.py`
- **Face crops**: UUID assigned by `extract_face_crops.py`; includes reference to parent clip's UUID
- **Quality annotations**: UUID assigned by `annotate_face_quality.py`; includes reference to parent crop's UUID

This enables permanent links, citation, and reproducibility across all datasets.

### Source Tracking
Person clip sidecars include Archive.org metadata:
```json
"source": {
  "archive_org_id": "titanic_1912",
  "archive_org_url": "https://archive.org/details/titanic_1912",
  "license": "public-domain"
}
```

Face crop and quality annotation sidecars include parent references:
```json
"parent_clip": {
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "file": "55-09-25  The Jack Benny Program  s06e01  Jack Goes To Dennis' House_01m07s-01m09s.mp4"
}
```

### JSON Schemas
Formal JSON Schemas validate all sidecars:
- `schemas/person_clip_schema.json` — Person clip structure
- `schemas/face_crop_schema.json` — Face crop structure
- `schemas/quality_annotation_schema.json` — Quality annotation structure

Validation is automatic during sidecar write operations via `jsonschema`.

---

## Overview: Data Flow

```
extracted_person_clips/
  VideoTitle.mp4                       ← Full-body clip with all detected persons
  VideoTitle.json                      ← Sidecar: bboxes, keypoints, per-frame data
  
face_crops/ (or filtered_face_crops/)
  VideoTitle_face_0.mp4                ← 616×616 OFIQ crop for person 0
  VideoTitle_face_0.json               ← Sidecar: same format as person clip (crop metadata)
  VideoTitle_face_0.quality.json       ← Quality scores (7 OFIQ measures)
  
  VideoTitle_face_1.mp4                ← 616×616 OFIQ crop for person 1
  VideoTitle_face_1.json
  VideoTitle_face_1.quality.json
```

**Key insight:** Quality annotations are **additions** to the face crop pipeline — they don't change the existing data structures, they just add `.quality.json` files alongside face crops.

---

## 1. Extracted Person Clips (Full-Body Videos)

**Location**: `extracted_person_clips/VideoTitle.mp4` + `VideoTitle.json`

**What it is**: A full-body video clip of one or more detected persons, with rich metadata about detections, pose, and transcription.

### Person Clip Sidecar Structure

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "schema_version": "1.0",
  "source": {
    "archive_org_id": "titanic_1912",
    "archive_org_url": "https://archive.org/details/titanic_1912",
    "license": "public-domain"
  },

  "start_frame": 0,
  "end_frame": 2500,
  "start_seconds": 0.0,
  "end_seconds": 100.0,
  "duration_seconds": 100.0,
  
  "source_video": "DARD/archive_org_public_domain/VideoTitle.mp4",
  "fps": 24.0,
  "video_info": {
    "width": 1280,
    "height": 720,
    "codec": "h264",
    "duration_seconds": 100.0
  },
  
  "track_ids": [0, 1, 3],
  
  "frame_data": {
    "0": [
      {
        "track_id": 0,
        "bbox": [150, 200, 300, 450],
        "score": 0.95,
        "keypoints": [[162, 210], [168, 215], ..., [290, 440]],
        "keypoint_scores": [0.98, 0.97, ..., 0.92],
        "face_visible": true,
        "frontal": true,
        "mouth_open": false,
        "face_crop_corners_ofiq": [[160, 180], [340, 180], [340, 560], [160, 560]],
        "face_crop_corners_arcface": [[200, 220], [280, 220], [280, 300], [200, 300]]
      },
      {
        "track_id": 1,
        "bbox": [800, 150, 950, 400],
        ...
      }
    ],
    "1": [...]
  },
  
  "transcription": "Well, hello there! How are you today?",
  
  "face_quality": {
    "0": {
      "unified_score": {"max": 90.5, "mean": 87.2, "p50": 88.1},
      "sharpness": {"max": 95.0, "mean": 92.3, "p50": 93.0},
      ...
    }
  }
}
```

### FAIR Metadata Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `uuid` | string | UUID v4 unique identifier for this person clip |
| `schema_version` | string | Schema version (e.g., `"1.0"`) for backwards compatibility |
| `source` | object | Archive.org source metadata and license tracking |
| `source.archive_org_id` | string | Archive.org identifier (e.g., `"titanic_1912"`) |
| `source.archive_org_url` | string | Full Archive.org item URL |
| `source.license` | string | License of the source content (e.g., `"public-domain"`) |

### Data Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `start_frame`, `end_frame` | int | Frame range of this clip within the source video |
| `start_seconds`, `end_seconds` | float | Time range in seconds |
| `duration_seconds` | float | Clip length |
| `source_video` | string | Full path (forward-slash) to original Internet Archive film |
| `fps` | float | Frames per second |
| `video_info` | object | Video codec, dimensions, duration metadata |
| `track_ids` | array[int] | List of unique person identifiers in this clip |
| `frame_data` | array | Per-frame detections and annotations (see below) |
| `transcription` | string | Speech transcription (filled by `transcribe_clips.py`) |

### Per-Frame Data

Each entry in `frame_data` describes all persons detected in one frame:

| Field | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | int | 0-based frame number in the clip |
| `timestamp` | float | Time in seconds within the clip |
| `persons` | array | List of detected person objects (one per `track_id`) |

### Per-Person Detection

For each tracked person in a frame:

| Field | Type | Description |
| :--- | :--- | :--- |
| `track_id` | int | Person identifier (consistent across frames in the clip) |
| `bbox` | [x1, y1, x2, y2] | Bounding box in source video pixel coordinates |
| `score` | float | Detection confidence [0, 1] |
| `keypoints` | array[[x, y], ...] | 133 COCO WholeBody pose keypoints (x, y per joint) |
| `keypoint_scores` | array[float] | Confidence per keypoint [0, 1] |
| `face_visible` | bool | True if face is clearly visible and frontal |
| `frontal` | bool | True if face is looking mostly toward camera |
| `mouth_open` | bool | True if mouth is detectably open |
| `face_crop_corners_ofiq` | [[x, y], [x, y], [x, y], [x, y]] | 4 corners of OFIQ-aligned face crop in source video coordinates (top-left, top-right, bottom-right, bottom-left) |
| `face_crop_corners_arcface` | [[x, y], [x, y], [x, y], [x, y]] | 4 corners of ArcFace-aligned region within the OFIQ crop (constant across all frames due to fixed landmark alignment) |

**Note**: `keypoints` are indexed by COCO WholeBody joint order; refer to `persondet/poser.py` for full joint list.

---

## 2. Face Crop Videos (Aligned Face Crops)

**Location**: `face_crops/VideoTitle_face_N.mp4` + `VideoTitle_face_N.json`

**What it is**: A 616×616 OFIQ-aligned face crop video extracted from one person's detections in a person clip. Used as input to face quality assessment.

### Face Crop Sidecar Structure

Face crop sidecars use the **same format as person clip sidecars**, but specialized for a single person:

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440001",
  "schema_version": "1.0",
  "parent_clip": {
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
    "file": "55-09-25  The Jack Benny Program  s06e01  Jack Goes To Dennis' House_01m07s-01m09s.mp4"
  },

  "start_frame": 0,
  "end_frame": 2500,
  "start_seconds": 0.0,
  "end_seconds": 100.0,
  "duration_seconds": 100.0,
  
  "source_video": "path/to/extracted_person_clips/VideoTitle.mp4",
  "track_id": 0,
  "crop_format": "ofiq",
  "output_size": 616,
  
  "fps": 24.0,
  "video_info": {
    "width": 616,
    "height": 616,
    "codec": "h264",
    "duration_seconds": 100.0
  },
  
  "frame_data": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "bbox": [50, 60, 560, 570],
      "score": 0.95,
      "keypoints": [[60, 75], [65, 80], ..., [550, 565]],
      "keypoint_scores": [0.98, 0.97, ..., 0.92],
      "face_crop_corners_arcface": [[180, 200], [250, 200], [250, 270], [180, 270]]
    },
    {
      "frame_index": 1,
      ...
    },
    ...
  ],
  
  "valid_face_frames": 2500
}
```

### FAIR Metadata Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `uuid` | string | UUID v4 unique identifier for this face crop |
| `schema_version` | string | Schema version (e.g., `"1.0"`) for backwards compatibility |
| `parent_clip` | object | Reference to the parent person clip |
| `parent_clip.uuid` | string | UUID of the parent person clip |
| `parent_clip.file` | string | Filename of the parent person clip |

### Key Differences from Person Clips

| Field | Meaning |
| :--- | :--- |
| `track_id` | Which person this crop came from (used to back-propagate quality) |
| `crop_format` | **"ofiq"** — signals that this is a 616×616 OFIQ-aligned crop |
| `output_size` | **616** — OFIQ canonical size (eyes at y≈272, nose at y≈336) |
| `frame_data` | Only contains `bbox`, `keypoints`, `keypoint_scores` (no need for multiple persons — it's just one person's face) |
| `valid_face_frames` | Count of frames where face was successfully detected and cropped |

**Note**: `face_crop_corners_arcface` is **constant across all frames** because both OFIQ and ArcFace align to fixed landmark positions. The 4 corners define the region within each 616×616 OFIQ frame where the 112×112 ArcFace crop is extracted.

---

## 3. Quality Annotations (Face Quality Scores)

**Location**: `face_crops/VideoTitle_face_N.quality.json` (alongside each face crop video)

**What it is**: A sidecar containing OFIQ face quality measurements computed from the face crop video.

### Quality JSON Structure

```json
{
  "uuid": "550e8400-e29b-41d4-a716-446655440002",
  "schema_version": "1.0",
  "parent_crop": {
    "uuid": "550e8400-e29b-41d4-a716-446655440001",
    "file": "55-09-25  The Jack Benny Program  s06e01  Jack Goes To Dennis' House_01m07s-01m09s_face_0.mp4"
  },

  "face_crop_video": "VideoTitle_face_1.mp4",
  "face_crop_json": "VideoTitle_face_1.json",
  "source_video": "path/to/extracted_person_clips/VideoTitle.mp4",
  "annotated_at": "2026-05-06T10:28:54Z",
  "annotator": "scripts/annotate_face_quality.py",
  "frame_stride": 1,
  "max_frames_sampled": 30,
  
  "unified_score": {...},
  "sharpness": {...},
  "compression_artifacts": {...},
  "expression_neutrality": {...},
  "no_head_coverings": {...},
  "face_occlusion_prevention": {...},
  "head_pose": {...},
  
  "frame_data": [...]
}
```

### FAIR Metadata Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `uuid` | string | UUID v4 unique identifier for this quality annotation |
| `schema_version` | string | Schema version (e.g., `"1.0"`) for backwards compatibility |
| `parent_crop` | object | Reference to the parent face crop being annotated |
| `parent_crop.uuid` | string | UUID of the parent face crop |
| `parent_crop.file` | string | Filename of the parent face crop video |

### Provenance Fields

| Field | Description |
| :--- | :--- |
| `face_crop_video` | Filename of the OFIQ face crop video this annotation is for |
| `face_crop_json` | Filename of the face crop sidecar (contains per-frame crop metadata) |
| `source_video` | Full path to the original person clip |
| `annotated_at` | ISO 8601 timestamp (UTC) when annotation was computed |
| `annotator` | Tool identifier: `dardcollect/annotate_face_quality.py` |
| `frame_stride` | Sampling interval (e.g., `1` = score every frame, `5` = score every 5th) |
| `max_frames_sampled` | Max frames to score; actual count may be less if video is shorter |

---

## 4. Quality Measures (OFIQ)

Each quality measure summarizes one aspect of face image quality across sampled frames. All measures except `head_pose` follow this aggregate format:

```json
{
  "max": 85.3,
  "mean": 72.4,
  "p10": 60.1,
  "p50": 75.2,
  "p90": 88.6
}
```

| Statistic | Meaning |
| :--- | :--- |
| `max` | Highest score across sampled frames |
| `mean` | Average across sampled frames |
| `p10` | 10th percentile (worst 10%) |
| `p50` | Median |
| `p90` | 90th percentile (best 10%) |

### The Seven Measures

All measures follow [ISO/IEC 29794-5 (OFIQ)](https://www.iso.org/standard/81694.html):

#### 1. Unified Score

```json
{
  "unified_score": {
    "max": 45.8,
    "mean": 38.2,
    "p10": 25.1,
    "p50": 40.5,
    "p90": 52.3
  }
}
```

**Component**: `UnifiedQualityScore`  
**Model**: MagFace IResNet50 magnitude  
**Range**: [0, 100] (higher = better)  
**Meaning**: Overall face image quality as measured by how confidently a face recognition model can embed the crop. This is the **primary quality metric** in OFIQ. Scores reflect biometric sample suitability — essential for face recognition tasks.

> **Note:** MagFace requires 112×112 ArcFace crops. The script extracts these on-the-fly from each 616×616 OFIQ frame using the constant region from `persondet/face_geometry.py`. If the sidecar lacks `crop_format: "ofiq"`, this measure is omitted.

#### 2. Sharpness

```json
{
  "sharpness": {
    "max": 95.2,
    "mean": 87.1,
    "p10": 72.3,
    "p50": 88.5,
    "p90": 92.8
  }
}
```

**Component**: `Sharpness`  
**Model**: Laplacian/Sobel random forest  
**Range**: [0, 100] (higher = better)  
**Meaning**: Image sharpness — higher indicates crisp, in-focus faces. Lower scores suggest blur or motion artifacts.

#### 3. Compression Artifacts

```json
{
  "compression_artifacts": {
    "max": 89.4,
    "mean": 81.2,
    "p10": 68.5,
    "p50": 82.1,
    "p90": 90.3
  }
}
```

**Component**: `CompressionArtifacts`  
**Model**: SSIM CNN  
**Range**: [0, 100] (higher = better)  
**Meaning**: Absence of compression artifacts (JPEG blocking, etc.). Higher scores indicate high-quality, lightly-compressed images.

#### 4. Expression Neutrality

```json
{
  "expression_neutrality": {
    "max": 78.5,
    "mean": 62.3,
    "p10": 45.2,
    "p50": 65.1,
    "p90": 80.4
  }
}
```

**Component**: `ExpressionNeutrality`  
**Models**: HSEmotion EfficientNet-B0/B2 + AdaBoost  
**Range**: [0, 100] (higher = better)  
**Meaning**: Facial expression neutrality. Higher scores = neutral faces (minimal emotion). Lower scores = strong expressions (smiling, frowning, etc.), which can degrade face recognition.

#### 5. No Head Coverings

```json
{
  "no_head_coverings": {
    "max": 100.0,
    "mean": 95.7,
    "p10": 85.3,
    "p50": 98.2,
    "p90": 100.0
  }
}
```

**Component**: `NoHeadCoverings`  
**Model**: BiSeNet face parsing  
**Range**: [0, 100] (higher = better)  
**Meaning**: Absence of head coverings (hats, sunglasses, scarves, etc.). Computed as: `100 * (1 - fraction_of_face_occluded_by_hat_or_cloth)`.

#### 6. Face Occlusion Prevention

```json
{
  "face_occlusion_prevention": {
    "max": 92.1,
    "mean": 88.4,
    "p10": 78.6,
    "p50": 89.5,
    "p90": 95.2
  }
}
```

**Component**: `FaceOcclusionPrevention`  
**Model**: Face occlusion segmentation CNN  
**Range**: [0, 100] (higher = better)  
**Meaning**: Absence of occlusion from any source (hands, hair, shadows, etc.), detected via pixel-level segmentation.

#### 7. Head Pose

```json
{
  "head_pose": {
    "yaw_deg": {
      "mean": -5.2,
      "abs_mean": 8.7
    },
    "pitch_deg": {
      "mean": 2.1,
      "abs_mean": 6.4
    },
    "roll_deg": {
      "mean": -0.8,
      "abs_mean": 3.2
    },
    "yaw_quality": {
      "max": 99.2,
      "mean": 91.5,
      "p10": 80.1,
      "p50": 93.2,
      "p90": 98.5
    },
    "pitch_quality": {...},
    "roll_quality": {...}
  }
}
```

**Component**: `HeadPose`  
**Model**: MobileNetV1 3DDFAV2  
**Range**: Angles in degrees (signed); quality scores in [0, 100]  
**Meaning**: Head pose angles and per-angle quality confidence.

- **Angles**: `mean` = average angle; `abs_mean` = average absolute deviation from frontal (frontal = 0° yaw, 0° pitch, 0° roll).
- **Quality scores**: Cosine² confidence per angle; higher = more confidence in estimate.

**Sign convention**:
- **Yaw**: Negative = head turned left, positive = head turned right
- **Pitch**: Positive = face looking down, negative = looking up
- **Roll**: Positive = head tilted right, negative = tilted left

---

## 5. Per-Frame Quality Data

## 5. Per-Frame Quality Data

The `.quality.json` file includes a `frame_data` array with individual frame scores. This enables **dynamic per-frame visualization** in the viewer:

```json
{
  "frame_data": [
    {
      "frame_index": 0,
      "unified_score": 42.3,
      "sharpness": 89.2,
      "compression_artifacts": 85.1,
      "expression_neutrality": 65.4,
      "no_head_coverings": 100.0,
      "face_occlusion_prevention": 91.2,
      "head_pose": {
        "yaw_deg": -3.5,
        "pitch_deg": 1.2,
        "roll_deg": -0.8,
        "yaw_quality": 95.3,
        "pitch_quality": 92.1,
        "roll_quality": 88.7
      }
    },
    {
      "frame_index": 1,
      "unified_score": 45.1,
      "sharpness": 91.5,
      ...
    },
    ...
  ]
}
```

**Structure**: Each entry represents one sampled frame:

| Field | Meaning |
| :--- | :--- |
| `frame_index` | 0-based index in the original face crop video (respects `frame_stride`) |
| All quality measures | Single floating-point values (not aggregates) |
| `head_pose` | Same structure as aggregate, but with single angle/quality values |

**Viewer usage**: As you play or scrub through a face crop video, the viewer looks up the current frame and displays its individual frame scores instead of aggregates. This shows **exactly what the model scored at that moment**.

---

## 6. Back-Propagated Quality in Person Clips

Quality summaries are automatically written into the **source person clip's sidecar JSON** under `face_quality`, indexed by `track_id`:

**File**: `extracted_person_clips/VideoTitle.json`

```json
{
  "start_frame": 0,
  "...": "...other fields...",
  
  "face_quality": {
    "0": {
      "unified_score": {
        "max": 52.3,
        "mean": 45.8,
        "p10": 38.1,
        "p50": 47.2,
        "p90": 58.9
      },
      "sharpness": {...},
      "compression_artifacts": {...},
      "expression_neutrality": {...},
      "no_head_coverings": {...},
      "face_occlusion_prevention": {...},
      "head_pose": {...},
      "face_crop": "VideoTitle_face_0.mp4"
    },
    "1": {
      "unified_score": {...},
      ...
      "face_crop": "VideoTitle_face_1.mp4"
    }
  }
}
```

**Structure**:
- **Key**: `track_id` as a string (e.g., `"0"`, `"1"`)
- **Value**: Quality summary for that person:
  - All aggregated quality measures (max, mean, p10, p50, p90)
  - `face_crop`: Filename of the extracted face crop video
  - **Note**: Only aggregates here (no `frame_data`); per-frame data stays in the `.quality.json` alongside the face crop

This enables the viewer to show per-track quality when browsing person clips — expand a track accordion to see its quality metrics.

---

## 7. Viewer Integration

### Face Crop Videos

When viewing a face crop video (from `face_crops/` or `filtered_face_crops/`):

1. Viewer loads the `.quality.json` sidecar
2. As you **play** the video:
   - Extracts current frame index
   - Looks up closest frame in `frame_data`
   - **Displays per-frame quality metrics dynamically**
3. As you **scrub** the timeline, quality metrics update in real-time

**Result**: You see **frame-by-frame variation** — exactly what the model scored for each frame.

### Person Clips

When viewing a person clip (from `extracted_person_clips/`):

1. Viewer reads the sidecar's `face_quality` field
2. For each `track_id`:
   - Creates an expandable accordion panel
   - Header shows person's **max unified_score** (color-coded bar)
   - Expanding shows all quality measures (aggregate stats)

**Result**: Quick assessment of each person's face quality without opening their extracted crop.

---

## 8. Example Workflow

1. Run `annotate_face_quality.py` on face crops in `filtered_face_crops/`
   ```bash
   python scripts/annotate_face_quality.py
   ```
   → Produces `VideoTitle_face_N.quality.json` files next to each video

2. Run the back-propagation step (automatic within `annotate_face_quality.py`):
   → Updates `extracted_person_clips/VideoTitle.json` with `face_quality[track_id]` entries

3. Open `viewer/detection_viewer.html` and drop in:
   - **`extracted_person_clips/`** → See person clips with quality accordions
   - **`filtered_face_crops/`** → See face crops with dynamic per-frame quality display

---

## Quality Score Interpretation

| Score Range | Interpretation |
| :--- | :--- |
| 0–33 | Poor quality; likely unsuitable for face recognition |
| 34–66 | Moderate quality; acceptable for many applications; may need preprocessing |
| 67–100 | High quality; suitable for demanding applications (face recognition, forensics) |

These ranges are approximate and task-dependent. The `filter_face_crops_by_quality.py` script uses `quality_threshold` (default: 75.0) to auto-filter by `unified_score`.

---

## File Location Reference

| File Type | Location | Produced By | Contains |
| :--- | :--- | :--- | :--- |
| Person clip video | `extracted_person_clips/VideoTitle.mp4` | `extract_person_clips.py` | Full-body video of 1+ persons |
| Person clip sidecar | `extracted_person_clips/VideoTitle.json` | `extract_person_clips.py` + `annotate_face_quality.py` | Bboxes, keypoints, per-frame data, transcription, `face_quality[track_id]` |
| Face crop video | `face_crops/VideoTitle_face_N.mp4` | `extract_face_crops.py` | 616×616 OFIQ-aligned crop of one person |
| Face crop sidecar | `face_crops/VideoTitle_face_N.json` | `extract_face_crops.py` | Crop metadata (same format as person clip, single person) |
| Quality annotation | `face_crops/VideoTitle_face_N.quality.json` | `annotate_face_quality.py` | 7 OFIQ quality measures + `frame_data` array |

---

## References

- **OFIQ Specification**: [ISO/IEC 29794-5](https://www.iso.org/standard/81694.html)
- **OFIQ Reference Implementation**: [BSI-OFIQ/OFIQ-Project](https://github.com/BSI-OFIQ/OFIQ-Project)
- **MagFace Paper**: [MagFace: A Universal Representation for Face Recognition and Meta-Face Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Meta-Face_Recognition_CVPR_2021_paper.pdf)
