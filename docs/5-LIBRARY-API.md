# DARDcollect Library API

DARDcollect is both a **complete extraction pipeline** and a **modular library** of reusable components. This guide covers using individual components in custom workflows.

---

## Overview

The library is organized into functional groups:

| Module | Purpose | Use Case |
|--------|---------|----------|
| **Detection & Tracking** | YOLOX person detection, OC-SORT tracking | Find people in video |
| **Pose Estimation** | CIGPose 133-keypoint estimation | Extract body + face keypoints |
| **Audio** | Whisper transcription | Transcribe speech in audio/video |
| **OCR** | PaddleOCR + pdfplumber | Extract text from PDFs and documents |
| **Face Crops** | OFIQ alignment (616×616) | Normalize and extract face regions |
| **Quality Scoring** | ISO/IEC 29794-5 (OFIQ) | Rate face quality (7 dimensions) |
| **Frame Extraction** | PNG frame export | Save video as individual frames |
| **Archive.org** | Mass download with metadata | Fetch historical media |
| **FAIR Metadata** | UUID + provenance tracking | Enable reproducibility |

---

## Quick Examples

### 1. Detect People in Your Video

```python
from dardcollect import PersonDetector, DetectorConfig
from pathlib import Path

# Create config (or load from config.yaml)
config = DetectorConfig(
    detection_threshold=0.5,
    tracking_score_threshold=0.5,
    tracking_min_hits=3,
    tracking_max_time_lost=30,
    pose_keypoint_threshold=0.3,
    gpu_id=0,
)

# Initialize detector
detector = PersonDetector(
    config=config,
    model_path="dardcollect/models/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
)

# Detect in an image
import cv2
image = cv2.imread("my_video_frame.jpg")
bboxes, scores = detector.get_detections(image, score_threshold=0.5)

print(f"Found {len(bboxes)} people")
for bbox, score in zip(bboxes, scores):
    x1, y1, x2, y2 = bbox
    print(f"  Person at ({x1}, {y1}) → ({x2}, {y2}), confidence: {score:.2f}")
```

### 2. Estimate Body Pose

```python
from dardcollect import PoseEstimator
import numpy as np

# Initialize pose estimator
poser = PoseEstimator(
    model_path="dardcollect/models/cigpose-m_coco-wholebody_256x192.onnx",
    gpu_id=0
)

# Estimate keypoints for detected person
bbox = np.array([x1, y1, x2, y2])  # From detector
keypoints, scores = poser.estimate(image, bbox)

# keypoints shape: (133, 2) — 133 points (face, body, hands, feet)
# scores shape: (133,) — confidence for each keypoint

print(f"Detected {np.sum(scores > 0.3)} keypoints with confidence > 0.3")
```

### 3. Transcribe Audio from a Video Clip

```python
from dardcollect import AudioTranscriber
from pathlib import Path

# Initialize transcriber (lazy-loads model on first use)
transcriber = AudioTranscriber(model_size="small")

# Transcribe a full video
text = transcriber.transcribe_file(Path("my_video.mp4"))
print(f"Transcription: {text}")

# Or transcribe a time segment
segment_text = transcriber.transcribe_segment(
    video_path=Path("my_video.mp4"),
    start_time=10.0,  # seconds
    end_time=20.0,
)
print(f"Segment (10-20s): {segment_text}")
```

### 4. Extract Text from a PDF

```python
from dardcollect import DocumentExtractor
from pathlib import Path

# Initialize OCR extractor
extractor = DocumentExtractor(
    models_dir=Path("dardcollect/models"),
    gpu_id=0
)

# Extract text (tries text layer first, falls back to OCR)
text, language = extractor.extract(Path("scanned_document.pdf"))
print(f"Extracted {len(text)} characters in {language}")
print(text[:500])  # First 500 chars
```

### 5. Extract Face Crops from Your Images

```python
from dardcollect import (
    PersonDetector,
    PoseEstimator,
    process_image,
)
from dardcollect.config import FaceCropConfig
from pathlib import Path
import json

# Setup
det_config = DetectorConfig(...)
face_config = FaceCropConfig(
    output_dir="my_face_crops/",
    min_face_size_percent=2.0,
    min_eye_distance_px=10.0,
)

detector = PersonDetector(config=det_config, model_path="...")
poser = PoseEstimator(model_path="...")

# Process one image
detection_json = Path("my_image.json")  # From your own detection pipeline
num_crops = process_image(
    image_path=Path("my_image.jpg"),
    detection_json_path=detection_json,
    face_config=face_config,
    output_dir=Path("my_face_crops/"),
)

print(f"Extracted {num_crops} face crops")

# Load the crop metadata
for crop_file in Path("my_face_crops/").glob("*_face_*.jpg"):
    json_file = crop_file.with_suffix(".json")
    with open(json_file) as f:
        metadata = json.load(f)
    print(f"Crop UUID: {metadata['uuid']}")
```

### 6. Score Face Quality (OFIQ)

```python
from dardcollect import load_models, score_video
from pathlib import Path

# Load all OFIQ quality models
models = load_models(
    models_dir=Path("dardcollect/models"),
    gpu_id=0
)

# Score a face crop video (616×616 OFIQ-aligned)
crop_path = Path("my_face_crop.mp4")
result = score_video(
    crop_path=crop_path,
    models=models,
)

# result contains detailed quality scores (7 OFIQ dimensions)
unified_score = result["unified_score"]
print(f"Unified quality score: {unified_score:.1f}/100")

if unified_score >= 70.0:
    print(f"✓ Face passes quality filter")
else:
    print(f"✗ Face does not meet quality threshold")
```

### 7. Download Media from Archive.org

```python
from dardcollect import download_item
from pathlib import Path

result = download_item(
    identifier="example_item_2020",
    dest_dir=Path("downloads/"),
    seen_titles=set(),
    history_file=Path("downloads.csv"),
    min_duration_mins=1.0,
    media_type="video",
)

if result["success"]:
    print(f"✓ Downloaded: {result['metadata']['title']}")
    print(f"  UUID: {result['metadata']['uuid']}")
    print(f"  License: {result['metadata']['license']}")
else:
    print(f"✗ Download failed: {result}")
```

### 8. Add FAIR Metadata to Your Data

```python
from dardcollect import add_fair_metadata, generate_uuid, reorganize_for_fair
from pathlib import Path
import json

# Create a data object
data = {
    "detection_result": {
        "bbox": [100, 150, 300, 450],
        "score": 0.95,
    }
}

# Add FAIR metadata (UUID, timestamps, schema version)
fair_data = add_fair_metadata(
    data=data,
    schema_type="person_clip",  # or "face_crop", "transcription", etc.
    source_url="https://archive.org/details/example",
    parent_uuid=None,
)

# Reorder keys so FAIR fields appear first (human-readable JSON)
reorganized = reorganize_for_fair(fair_data, schema_type="person_clip")

# Save with full provenance
with open("output_with_metadata.json", "w") as f:
    json.dump(reorganized, f, indent=2)
```

### 9. Check Face Visibility and Frontal Orientation

```python
from dardcollect import check_face_visibility, check_frontal_face
import numpy as np

# Assume you have keypoints from PoseEstimator
keypoints = np.array([...])  # (133, 2)
scores = np.array([...])     # (133,)

# Check if face is visible and large enough
is_visible = check_face_visibility(
    keypoints=keypoints,
    scores=scores,
    image_height=720,
    min_face_size_percent=2.0,
    score_threshold=0.3,
)

if is_visible:
    # Check if face is frontal (not in profile)
    is_frontal = check_frontal_face(
        keypoints=keypoints,
        scores=scores,
        symmetry_threshold=0.6,
        score_threshold=0.3,
    )
    
    if is_frontal:
        print("✓ Frontal face detected")
    else:
        print("✗ Face is in profile or rotated")
else:
    print("✗ Face not visible or too small")
```

### 10. Extract Individual Frames from a Video

```python
from dardcollect import extract_frames
from pathlib import Path

manifest = extract_frames(
    video_path=Path("my_video.mp4"),
    sidecar_path=Path("my_video.json"),  # Detection metadata sidecar
    output_dir=Path("extracted_frames/"),
    clip_type="person_clip",
    overwrite=False,
)

if manifest:
    print(f"Extracted {len(manifest['frames'])} frames")
    for frame_info in manifest['frames'][:3]:
        print(f"  {frame_info['filename']} — UUID: {frame_info['uuid']}")
else:
    print("Frame extraction failed")
```

---

## Configuration

Most components accept a configuration object. For consistency with the pipeline, use YAML:

```yaml
# config.yaml
person_extraction:
  detection_threshold: 0.5
  tracking_score_threshold: 0.5
  tracking_min_hits: 3
  tracking_max_time_lost: 30
  pose_keypoint_threshold: 0.3
  gpu_id: 0
```

Load programmatically:

```python
from dardcollect import DetectorConfig

config = DetectorConfig.from_yaml("config.yaml")
detector = PersonDetector(config=config, model_path="...")
```

---

## GPU & CPU Modes

All ONNX-based components (detection, pose, quality, OCR) automatically select the best execution provider:

1. **TensorRT** (NVIDIA RTX/A100) — fastest
2. **CUDA** (any NVIDIA GPU with CUDA 12+)
3. **CPU** (fallback)

To force CPU mode:

```python
import os
os.environ["DETECTOR_USE_GPU"] = "0"

# Then import components
from dardcollect import PersonDetector
```

---

## Error Handling

All components raise informative exceptions on failure:

```python
from dardcollect import PersonDetector
import logging

logging.basicConfig(level=logging.INFO)

try:
    detector = PersonDetector(config=config, model_path="missing.onnx")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except RuntimeError as e:
    print(f"GPU error: {e}")
```

---

## Combining Components

Build custom pipelines by chaining components:

```python
from dardcollect import (
    PersonDetector,
    PoseEstimator,
    process_image,
    check_face_visibility,
)
from dardcollect.config import FaceCropConfig
from pathlib import Path
import cv2

# 1. Detect people
detector = PersonDetector(config, model_path="...")
image = cv2.imread("photo.jpg")
bboxes, scores = detector.get_detections(image)

# 2. Estimate pose
poser = PoseEstimator(model_path="...")
for bbox in bboxes:
    kpts, kpt_scores = poser.estimate(image, bbox)
    
    # 3. Validate face
    if check_face_visibility(kpts, kpt_scores, image.shape[0], min_face_size_percent=2.0):
        # 4. Extract face crop
        det = {"keypoints": kpts, "keypoint_scores": kpt_scores}
        process_image(Path("photo.jpg"), Path("photo.json"), FaceCropConfig(...), ...)
```

---

## See Also

- **Complete Pipeline Example:** See [docs/0-GETTING-STARTED.md](0-GETTING-STARTED.md)
- **Architecture & Data Flow:** See [docs/1-ARCHITECTURE.md](1-ARCHITECTURE.md)
- **CSV Traceability & Provenance:** See [docs/2-LINEAGE.md](2-LINEAGE.md)
- **JSON Sidecar Formats:** See [docs/3-ANNOTATIONS.md](3-ANNOTATIONS.md)
- **Development Guide:** See [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md)

← [Back to README](../README.md)
