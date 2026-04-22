# Model Card — BiSeNet Face Parsing (`bisenet_400.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Face region parsing / semantic segmentation** — labelling each pixel of a face image as belonging to one of several facial components (skin, hair, eyes, nose, lips, background, head coverings, etc.).  
**Original authors:** BiSeNet — Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, Nong Sang (ECCV 2018). Face-specific training: OFIQ project (BSI/Fraunhofer ITWM).  
**ONNX packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation.  
**Version:** BiSeNet, 400×400 input resolution.

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0.
- Intended downstream use (OFIQ): `NoHeadCoverings` and `FaceOcclusionPrevention` quality measures — proportion of face pixels classified as head covering or occlusion.
- No external network access at inference time.

### 1c. Software Versions
- Input: float32, shape `[batch, 3, 400, 400]`.
- Output: per-pixel class logits or segmentation map.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project  
BiSeNet source: https://github.com/CoinCheung/BiSeNet

### 1e. Hardware Requirements
- Model file size: ~50 MB.
- Recommended: NVIDIA GPU with CUDA 12.x.

### 1g. Interface for Deployers
Not currently used directly by any pipeline script. Intended for future OFIQ-based quality measures (`NoHeadCoverings`, `FaceOcclusionPrevention`) that require face region segmentation.

### 1h. Usage Notes
- Input must be a face image resized to 400×400, normalised with ImageNet mean/std.
- Output class indices follow the OFIQ face parsing label scheme (defined in the OFIQ project documentation).

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Architecture: **BiSeNet** (Bilateral Segmentation Network) — two-path network combining a spatial path (preserving spatial detail) and a context path (capturing receptive field). Fused via a Feature Fusion Module.
- Training: fine-tuned by OFIQ project on face parsing datasets (CelebAMask-HQ or similar).

### 2b. Design Specifications & Algorithm
- **Spatial path:** 3 convolutional layers with stride 2; captures low-level spatial detail.
- **Context path:** Lightweight backbone (Xception-style) + global average pooling for large receptive field.
- **Feature Fusion Module:** Concatenates both paths, applies channel attention, produces final feature map.
- **Output head:** Per-pixel classification into facial region classes.

### 2d. Training Data
Face parsing typically trained on **CelebAMask-HQ** (Lee et al., 2020) — 30,000 high-resolution celebrity face images with 19-class pixel annotations (skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, earring, necklace, neck, cloth). Exact OFIQ training data not publicly disclosed.

### 2e. Human Oversight
Not currently wired into any automated pipeline stage. Future use in quality measures would feed a human-reviewable quality score.

### 2h. Cybersecurity
Weights loaded read-only. File obtained from the OFIQ project distribution.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Pixel-level face region segmentation enabling detection of occlusions and head coverings.
- Bilateral architecture balances spatial precision and semantic context efficiently.

### Limitations
- Trained on modern, high-resolution celebrity photographs; performance degrades on low-resolution or archival footage.
- 400×400 fixed input may lose fine detail when upscaling small face crops.

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5 (OFIQ): `NoHeadCoverings`, `FaceOcclusionPrevention` measures.
- BiSeNet: Yu et al., "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation", ECCV 2018.

---

## Citation
```bibtex
@inproceedings{yu2018bisenet,
  title     = {BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation},
  author    = {Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2018}
}
```
BiSeNet paper: https://arxiv.org/abs/1808.00897  
OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
