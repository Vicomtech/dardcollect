# Model Card — Compression Artifacts SSIM Model (`ssim_248_model.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Compression artifact detection** — estimating the degree of JPEG/compression distortion in a face image, expressed as an SSIM-derived score.  
**Original authors/packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.  
**Version:** CNN trained to predict SSIM degradation; 248×248 input resolution (centre-cropped to 184×184 internally per the OFIQ config).

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0.
- Intended downstream use (OFIQ): `CompressionArtifacts` quality measure — penalises images with visible JPEG block artefacts or other compression noise.
- No external network access at inference time.

### 1c. Software Versions
- Input: float32, shape `[batch, 3, 248, 248]` (centre-cropped to 184×184 per OFIQ config before passing to the head).
- Output: scalar SSIM-correlated score.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project

### 1g. Interface for Deployers
Not currently used directly by any pipeline script. Intended for future use in the `CompressionArtifacts` quality measure.

### 1h. Usage Notes
- The raw model output is mapped through a sigmoid calibration in the OFIQ pipeline (see `ofiq_config.jaxn`) to produce a [0, 100] score.
- Particularly relevant for datasets derived from re-encoded or heavily compressed video sources.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- CNN trained to predict SSIM between a compressed image and its uncompressed reference.
- SSIM (Structural Similarity Index) is a perceptual image quality metric that correlates well with human perception of compression artefacts.

### 2d. Training Data
Training data not publicly disclosed by BSI/ITWM. Likely synthetic: clean face images compressed at varying JPEG quality levels, with SSIM between compressed and original used as the regression target.

### 2e. Human Oversight
Not currently wired into any automated pipeline stage.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Detects compression artefacts that affect face recognition model performance (blocking, ringing, blurring at DCT boundaries).

### Limitations
- May conflate film grain (common in archival footage) with compression noise — both are high-frequency texture distortions.
- Training on synthetic JPEG compression may not generalise to video codec artefacts (H.264 blocking, motion blur).

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `CompressionArtifacts` measure.
- SSIM: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004.
- OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
