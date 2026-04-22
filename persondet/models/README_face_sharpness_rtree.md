# Model Card — Face Sharpness R-Tree (`face_sharpness_rtree.xml.gz`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Face sharpness / blur estimation** — predicting a sharpness score for a face image region, correlating with focus quality and motion blur.  
**Original authors/packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.  
**Format:** Gzip-compressed OpenCV `ml::RTrees` (random decision forest) XML model.

### 1b. Interaction with Hardware & Software
- Runtime: OpenCV `ml::RTrees::load()` (requires OpenCV ≥ 4.x).
- Intended downstream use (OFIQ): `Sharpness` quality measure — low sharpness scores indicate blur/defocus that degrades face recognition.
- No external network access at inference time.

### 1c. Software Versions
- Input: hand-crafted image features extracted from the face region (frequency-domain or gradient-based features; exact feature extraction defined in OFIQ C++ source).
- Output: scalar sharpness score.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project

### 1g. Interface for Deployers
Not currently used directly by any pipeline script. Intended for future use in the `Sharpness` quality measure via OpenCV.

### 1h. Usage Notes
- Unlike the ONNX models, this requires OpenCV's `ml` module (not ONNX Runtime).
- The raw score is mapped through a sigmoid calibration in the OFIQ pipeline (see `ofiq_config.jaxn`).
- `face_region_alpha: 0.0` in the OFIQ config means the entire face bounding box region is used (no erosion).

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Algorithm: **Random Decision Forest** (OpenCV RTrees) — ensemble of decision trees trained on image features to regress a sharpness score.
- Features: likely Laplacian variance, gradient magnitude statistics, or frequency spectrum metrics — standard hand-crafted blur indicators.

### 2d. Training Data
Training data not publicly disclosed by BSI/ITWM. Likely pairs of sharp and artificially blurred/defocused face images with perceptual sharpness labels.

### 2e. Human Oversight
Not currently wired into any automated pipeline stage.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Fast inference (random forest on hand-crafted features — no GPU needed).
- Interprets both defocus blur and motion blur.

### Limitations
- Hand-crafted features may not generalise well to film grain (present in archival footage), which can be mistaken for sharpness texture.
- Random forest model; requires OpenCV `ml` module (separate from ONNX Runtime infrastructure used by the rest of the pipeline).

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `Sharpness` measure.
- OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
