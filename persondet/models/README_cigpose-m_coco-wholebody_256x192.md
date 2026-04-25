# Model Card — CIGPose-M Wholebody (`cigpose-m_coco-wholebody_256x192.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Top-down **whole-body pose estimation** — predicting 133 keypoints (body, feet, face, and hands) for a pre-cropped person region.  
**Original authors:** Bohao Li, Zhicheng Cao, Huixian Li, Yangming Guo (CVPR 2026).  
**ONNX packaging:** Namas Bhandari (`namas191297`) via the `cigpose-onnx` repository; model weights converted to ONNX with embedded metadata (`cigpose_meta` field).  
**Version:** CIGPose-M, COCO-WholeBody 256×192, released March 2025.

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0 (CPU or CUDA execution provider).
- Upstream: receives person bounding boxes from RT-DETRv4-S detector + ByteTrack tracker.
- Downstream: keypoint coordinates and scores used for (a) mouth-open detection (`check_mouth_open`), (b) facial symmetry checks for frontal-face filtering, and (c) 5-point ArcFace face alignment in `persondet/face_geometry.py` — specifically body indices 0 (nose), 1 (left eye), 2 (right eye) and face indices 71 (dlib 48, viewer-left mouth corner) and 77 (dlib 54, viewer-right mouth corner).
- No external network access at inference time.

### 1c. Software Versions
- ONNX opset embedded in file; split_ratio and input dimensions read from `cigpose_meta` embedded metadata (fallback: 256×192, split_ratio=2.0).
- SimCC (Spatial Coordinate Classification) decoding applied in `persondet/postprocessing.py`.

### 1d. Distribution Form
Single ONNX file downloaded from the GitHub Releases of `namas191297/cigpose-onnx`:  
https://github.com/namas191297/cigpose-onnx/releases/latest/download/cigpose_models.zip  
Source paper repository: https://github.com/53mins/CIGPose

### 1e. Hardware Requirements
- CPU inference supported but slow for real-time use.
- Recommended: NVIDIA GPU with CUDA 12.x, ≥ 4 GB VRAM.
- Model file size: 74 MB.
- Approximate GFLOPs: 2.3 (M variant).

### 1g. Interface for Deployers
Entry points:
- `PoseEstimator.get_keypoints(image, bbox)` → `(keypoints, scores)` where `keypoints` is `(133, 2)` float32 in original image pixel coordinates, `scores` is `(133,)` float32.
- `PoseEstimator.check_mouth_open(keypoints, scores)` → `bool`.

### 1h. Usage Notes
- Requires a person bounding box as input; the model is **top-down** (not applicable to full-scene inference).
- Keypoint indexing follows COCO-133 layout: indices 0–16 are body, 17–22 feet, 23–90 face, 91–132 hands.
- Scores below `pose_keypoint_threshold` (default 0.4) should be treated as unreliable.
- Input is preprocessed as: aspect-ratio-preserving crop with 1.25× padding → resize to 256×192 → BGR→RGB → ImageNet normalisation.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Built on the **MMPose** framework (OpenMMLab).
- Employs a **Structural Causal Model (SCM)** to identify and remove spurious visual context ("backdoor paths") that corrupt keypoint predictions.
- Core modules: Causal Intervention Module (CIM) + Hierarchical Graph Neural Network (HGNN) for anatomical plausibility.
- SimCC (Spatial Coordinate Classification) output head: encodes keypoint locations as 1-D classification distributions rather than heatmaps, enabling fast argmax decoding.

### 2b. Design Specifications & Algorithm
- **Encoder:** Lightweight CNN backbone (Medium variant).
- **CIM:** Detects keypoint embeddings with high causal confusion score; replaces them with learned context-invariant prototypes from a maintained prototype bank.
- **HGNN:** Propagates corrected embeddings along the human skeleton graph to enforce anatomical consistency.
- **SimCC head:** Outputs separate x- and y-axis distributions per keypoint. Decoded via `argmax / split_ratio`.
- **Confidence score:** `min(max_val_x, max_val_y)` across the SimCC distributions.

### 2c. System Architecture & Compute
- Model file: 74 MB ONNX.
- Approx. 2.3 GFLOPs per forward pass at 256×192.
- Processes one person crop per call (batch size 1 at inference).

### 2d. Training Data
| Dataset | Split | Description |
|---------|-------|-------------|
| COCO-WholeBody v1.0 train | ~118,000 images | 133-keypoint whole-body annotations on COCO images |
| COCO-WholeBody v1.0 val | 5,000 images | Evaluation |
| UBody (optional) | ~1 M frames | Upper-body internet video dataset; used in larger variants |

**COCO-WholeBody** extends COCO 2017 with face, hand, and foot keypoint annotations crowd-sourced by the WholeBody consortium (Jin et al., 2020).  
**UBody** (Lin et al., 2023) is collected from internet videos; demographic representation is not formally documented.  
Training batch: 8 × 64 images, 420 epochs at 256×192.

### 2e. Human Oversight
Keypoint outputs feed into secondary heuristics (`check_mouth_open`, symmetry ratio) with configurable thresholds. All results feed a downstream human review process for clip selection. No legal or consequential automated decision relies solely on pose output.

### 2g. Validation & Testing
Evaluated on COCO-WholeBody v1.0 val2017:

| Model variant | Whole AP | Body AP | Face AP | Hand AP | Foot AP |
|---------------|----------|---------|---------|---------|---------|
| CIGPose-x (paper best) | 67.5 | — | — | — | — |
| CIGPose-m (this model) | Not separately published; expected lower than CIGPose-x | — | — | — | — |

Note: the authors publish aggregate whole-AP figures; per-part breakdown for the M variant is not available in the paper.

### 2h. Cybersecurity
- Weights are loaded read-only at startup.
- ONNX Runtime sandboxes graph execution.
- File downloaded from a public GitHub Release artifact; SHA integrity is not verified at load time — users in security-sensitive deployments should verify the file hash against the release.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Predicts 133 anatomical landmarks per person, covering the entire body.
- Robust to context-induced spurious correlations due to the causal intervention mechanism.
- Operates in real time on GPU for single-person crops.

### Limitations
- **Top-down dependency:** Quality depends entirely on the upstream detector bounding box. Poor localisation (too tight, off-centre) significantly degrades keypoint accuracy.
- **Historical footage:** Trained on modern internet imagery. Grayscale film, low frame rates, film grain, and non-contemporary clothing and body proportions are out-of-distribution.
- **Hand and face keypoints:** These sub-tasks have substantially lower AP than body keypoints. Face landmark indices 85/89 (upper/lower inner lip, mouth-open detection) and 71/77 (outer mouth corners, face alignment) may be unreliable when faces are < ~40 px. Alignment degrades gracefully to 2-eye-only when mouth/nose keypoints fall below `pose_keypoint_threshold`.
- **Occlusion:** Heavily occluded or partially visible persons produce unreliable keypoints for the hidden regions.
- **Small crops:** Performance degrades below ~100×200 px crop size.
- **Demographic fairness:** No formal evaluation of keypoint accuracy disparity across skin tones, age groups, or body types has been published for this model.

### Foreseeable Unintended Outcomes
- False mouth-open detection due to low-confidence lip keypoints in low-resolution or blurry frames.
- Anatomically incorrect but high-confidence keypoints in heavily occluded scenes not encountered in training.

### Input Data Specifications
- Input: BGR uint8 video frame + float32 bounding box `[x1, y1, x2, y2]`.
- Minimum useful crop: approximately 80×160 px before internal resize.

---

## 4. Performance Metrics Rationale
Whole-AP (IoU-based over keypoint visibility) is the standard MMPose/COCO-WholeBody evaluation metric. It captures both localisation precision and keypoint visibility classification simultaneously, which is appropriate for a model used to measure anatomical feature positions (lip distance, eye distance).

---

## 5. Risk Management
Within this pipeline the model is used for **non-high-risk** video archival analysis. Pose keypoints are only one signal among several (face size, symmetry, tracking duration) used to classify clips. Risks are mitigated by:
- Score thresholding (`pose_keypoint_threshold: 0.4`).
- Mouth-open detection requires both lip keypoints to exceed the minimum score before returning `True`.
- Symmetry-based frontal face check provides a redundant gating signal.

---

## 6. Known Lifecycle Changes
The ONNX file corresponds to the March 2025 release of `cigpose-onnx`. The CIGPose paper was accepted to CVPR 2026; a revised checkpoint may be released. No fine-tuning on this project's data has been performed.

---

## 7. Standards & Specifications Applied
- COCO-WholeBody evaluation protocol (https://github.com/jin-s13/COCO-WholeBody).
- MMPose framework conventions (https://github.com/open-mmlab/mmpose).
- ONNX specification (https://onnx.ai).
- SimCC decoding: Li et al., "SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation", ECCV 2022.

---

## Citation
```bibtex
@inproceedings{li2026cigpose,
  title     = {CIGPose: Causal Intervention Graph Neural Network
               for Whole-Body Pose Estimation},
  author    = {Li, Bohao and Cao, Zhicheng and Li, Huixian and Guo, Yangming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
arXiv preprint: https://arxiv.org/abs/2603.09418
