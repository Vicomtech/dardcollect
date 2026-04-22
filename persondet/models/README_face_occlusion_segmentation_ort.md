# Model Card — Face Occlusion Segmentation (`face_occlusion_segmentation_ort.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Face occlusion segmentation** — detecting and segmenting regions of the face that are occluded by foreign objects (hands, masks, sunglasses, scarves, etc.).  
**Original authors/packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.  
**Version:** ONNX Runtime–compatible variant (`_ort`).

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0.
- Intended downstream use (OFIQ): `FaceOcclusionPrevention` quality measure — penalises face images where significant portions of the face are occluded.
- No external network access at inference time.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project

### 1g. Interface for Deployers
Not currently used directly by any pipeline script. Intended for future use in the `FaceOcclusionPrevention` quality measure.

---

## 2. Development Elements & Process

### 2d. Training Data
Training data and architecture details are not publicly disclosed by BSI/ITWM beyond the OFIQ project documentation. Likely trained on face datasets augmented with synthetic and real occlusions.

### 2e. Human Oversight
Not currently wired into any automated pipeline stage. Future use would feed a human-reviewable quality score.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Detects occlusions that are not captured by face parsing alone (e.g., hands covering part of the face).

### Limitations
- Architecture and training data details not publicly disclosed; generalisation outside OFIQ's tested conditions is uncertain.
- May not generalise well to historical footage where period accessories (veils, hats) could be misclassified as occlusions.

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `FaceOcclusionPrevention` measure.
- OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
