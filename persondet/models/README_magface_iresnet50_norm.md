# Model Card — MagFace IResNet50 (`magface_iresnet50_norm.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** **Face image quality assessment** — computing a scalar quality score that measures how confidently a face recognition model can embed a given face crop. Used as the unified quality score defined in ISO/IEC 29794-5 (OFIQ).  
**Original authors:** Qiang Liu, Anwei Luo, Huaibo Huang, Ran He — Institute of Automation, Chinese Academy of Sciences (CASIA).  
**ONNX packaging:** BSI/Fraunhofer ITWM as part of the OFIQ reference implementation (https://github.com/BSI-OFIQ/OFIQ-Project).  
**Version:** IResNet50 (ArcFace backbone), trained with MagFace loss; normalised output variant (`_norm`).

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0 (CPU or CUDA execution provider).
- Upstream: receives 112×112 RGB face crops, normalised to [-1, 1].
- Downstream: scalar output used as quality threshold in `scripts/filter_face_crops_by_quality.py` to decide whether a face crop video passes filtering.
- No external network access at inference time.

### 1c. Software Versions
- ONNX opset embedded in file.
- Input tensor: `input`, dtype float32, shape `[batch_size, 3, 112, 112]`, values in [-1, 1].
- Output tensor: `output`, dtype float32, shape `[batch_size]` — scalar quality score per image.

### 1d. Distribution Form
Distributed as part of the OFIQ reference implementation model bundle:  
https://github.com/BSI-OFIQ/OFIQ-Project  
OFIQ project (BSI): https://www.bsi.bund.de/OFIQ  
MagFace source repository: https://github.com/IrvingMeng/MagFace

### 1e. Hardware Requirements
- CPU inference: functional, approximately 5–15 ms per image.
- Recommended: NVIDIA GPU with CUDA 12.x (< 2 ms per image).
- Model file size: ~92 MB (IResNet50).

### 1g. Interface for Deployers
Used via `scripts/filter_face_crops_by_quality.py`:
- Input: BGR uint8 numpy array of shape `(H, W, 3)` (any square size — resized internally to 112×112).
- Preprocessing: BGR→RGB, resize to 112×112, normalise `(pixel − 127.5) / 128.0`.
- Output: positive float quality score (higher = better). Not sigmoid-calibrated to [0, 100].

### 1h. Usage Notes
- Face crops must be aligned (eyes horizontal, face centred) before passing to this model. The upstream `extract_face_crops.py` stage performs CIGPose-based alignment, making the crops compatible.
- The score is the direct ONNX model output — it is the MagFace embedding magnitude, which is monotonically related to face recognition quality. It is **not** the OFIQ sigmoid-calibrated [0, 100] score.
- Set `quality_threshold` empirically by inspecting score distributions on your data.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Architecture: **IResNet50** — a residual network adapted for face recognition (He et al., 2016; Deng et al., 2019 for ArcFace adaptation).
- Training loss: **MagFace** — extends ArcFace by making the angular margin a function of feature magnitude. Low-quality faces (blurry, occluded, extreme pose) produce low-magnitude embeddings; high-quality faces produce high-magnitude embeddings.
- Training framework: PyTorch.

### 2b. Design Specifications & Algorithm
- **Backbone:** IResNet50 — 50-layer identity-mapping residual network adapted for 112×112 face input.
- **MagFace loss:** `L = −log(exp(s·(cos(θ_yi) − m(a_i))) / Σ_j exp(s·cos(θ_j)))` where `m(a_i)` is the margin as a function of the feature magnitude `a_i = ||f_i||`. Higher magnitude → larger margin → harder training → model assigns high magnitude only to high-quality faces.
- **Quality score:** `||f_i||` — the L2 norm of the 512-dimensional face embedding.
- The `_norm` variant outputs the scalar magnitude directly (no embedding vector exposed).

### 2c. System Architecture & Compute
- Model file: ~92 MB ONNX.
- Parameters: ~43 M (IResNet50 standard).
- Batch size 1 at inference in this pipeline.

### 2d. Training Data
Training data composition for the OFIQ-distributed checkpoint is not fully disclosed by BSI/ITWM. The original MagFace paper used:

| Dataset | Images | Identities |
|---------|--------|-----------|
| MS-Celeb-1M (cleaned) | ~3.8 M | ~85,000 |

MS-Celeb-1M (Guo et al., 2016) is a large-scale celebrity face dataset scraped from the web. It has known demographic representation skews (predominantly Western celebrities) and has been partially retracted due to consent concerns in some jurisdictions. The OFIQ-specific training split and any additional data are not publicly documented.

### 2e. Human Oversight
The quality score is used as a binary filter threshold. The threshold is set by the operator and all filtered results are available for human review. No consequential decision is automated solely on this score.

### 2g. Validation & Testing
MagFace was evaluated on standard face recognition benchmarks (LFW, CFP-FP, AgeDB-30) and on quality-correlated rank correlation against human quality labels. In the OFIQ context, UnifiedQualityScore correlation with overall operator assessment is reported in the OFIQ evaluation report (BSI TR-03166).

| Benchmark | Score |
|-----------|-------|
| LFW (1:1 verification) | 99.83% |
| CFP-FP (frontal-profile) | 98.46% |
| AgeDB-30 | 98.17% |

No formal benchmark on archival or historical film footage has been conducted.

### 2h. Cybersecurity
- Weights loaded read-only at startup via ONNX Runtime.
- File obtained from the OFIQ project distribution; SHA integrity should be verified against the OFIQ release manifest in security-sensitive deployments.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Produces a quality score that correlates with face recognition system performance — directly relevant to biometric dataset curation.
- Operates directly on aligned face crops without requiring face re-detection.
- Fast inference on GPU (< 2 ms per image).

### Limitations
- **Training data bias:** MS-Celeb-1M is skewed toward Western celebrities; quality scores may be less reliable for other demographic groups.
- **Alignment sensitivity:** Score degrades if face crops are not aligned (eyes horizontal). Our pipeline's CIGPose alignment mitigates this.
- **Calibration:** Raw output is not calibrated to [0, 100]; threshold must be set empirically.
- **Historical footage:** Model trained on modern photographs; archival film grain, low resolution, and period appearance are out-of-distribution.

### Foreseeable Unintended Outcomes
- High-quality but demographically under-represented faces may receive systematically lower scores due to training data bias.
- Very blurry or low-resolution frames return 0.0 (ONNX error caught), which correctly causes the video to fail — but silently if all frames error.

### Input Data Specifications
- Input: RGB uint8 numpy array, shape `(112, 112, 3)`, normalised to float32 `[-1, 1]`.
- NCHW layout after transpose: `(1, 3, 112, 112)`.

---

## 4. Performance Metrics Rationale
Rank correlation between embedding magnitude and human-assigned quality labels is the standard MagFace quality evaluation metric. It directly measures whether the model correctly orders faces by quality, which is appropriate for a threshold-based filter.

---

## 5. Risk Management
Used for **non-high-risk** archival data curation. Risks mitigated by:
- Operator-configurable threshold (`quality_threshold`).
- Videos failing quality filtering remain on disk and can be reconsidered.
- Human review of filtered output before downstream use.

---

## 6. Known Lifecycle Changes
Checkpoint distributed with OFIQ v1.1.2. Updated OFIQ releases may include a retrained or replaced quality model. No fine-tuning on this project's data has been performed.

---

## 7. Standards & Specifications Applied
- ISO/IEC 29794-5: Biometric sample quality — Part 5: Face image data (OFIQ unified quality score definition).
- OFIQ reference implementation: https://github.com/BSI-OFIQ/OFIQ-Project
- MagFace paper: Liu et al., "MagFace: A Universal Representation for Face Recognition and Quality Assessment", CVPR 2021.
- ArcFace/IResNet: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.

---

## Citation
```bibtex
@inproceedings{meng2021magface,
  title     = {MagFace: A Universal Representation for Face Recognition and Quality Assessment},
  author    = {Meng, Qiang and Zhao, Shichao and Huang, Zhida and Zhou, Feng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2021}
}
```
MagFace paper: https://arxiv.org/abs/2103.06627  
OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project
