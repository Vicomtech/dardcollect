# Model Card — MobileNet Head Pose (`mb1_120x120.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Head pose estimation** — predicting yaw, pitch, and roll angles (in degrees) of a face in an image.  
**Original authors:** WHENet / 6DRepNet family of head pose models. ONNX packaging: BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.  
**Version:** MobileNetV1 backbone, 120×120 input resolution.

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0.
- Intended downstream use (OFIQ): `HeadPose` quality measure — scores images by how close head pose is to frontal (yaw ≈ 0°, pitch ≈ 0°).
- No external network access at inference time.

### 1c. Software Versions
- Input: float32, shape `[batch, 3, 120, 120]`.
- Output: yaw, pitch, roll angles in degrees.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project

### 1e. Hardware Requirements
- Model file size: ~13 MB (MobileNetV1, lightweight).
- CPU inference: fast (< 5 ms per image).

### 1g. Interface for Deployers
Not currently used directly by any pipeline script. Intended for future use in the `HeadPose` quality measure.

### 1h. Usage Notes
- Input should be a face-cropped image resized to 120×120.
- Yaw angle is the most discriminative for frontal face assessment; pitch and roll indicate tilt.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Architecture: **MobileNetV1** — depthwise separable convolutions for efficient inference.
- Regression head: predicts Euler angles directly.

### 2d. Training Data
Typically trained on **300W-LP** (synthesised multi-pose face images from 300W) and **AFLW2000-3D** (3D annotated faces). Exact OFIQ training data not publicly disclosed.

### 2e. Human Oversight
Not currently wired into any automated pipeline stage.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Lightweight model suitable for real-time head pose estimation on CPU.
- Covers full yaw range (±90°) and moderate pitch/roll.

### Limitations
- MobileNetV1 accuracy is lower than larger models (ResNet-50 based approaches).
- Performance degrades significantly beyond ±60° yaw.
- Historical footage with unusual lighting, resolution, or film grain is out-of-distribution.

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `HeadPose` measure.
- OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
