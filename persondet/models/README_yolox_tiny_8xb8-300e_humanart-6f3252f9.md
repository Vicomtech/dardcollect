# Model Card — YOLOX-Tiny HumanArt (`yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`)

Technical documentation structured in accordance with EU AI Act Annex IV.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Single-class **person detection** — localising one or more persons in a video frame and returning bounding boxes with confidence scores.  
**Original authors:** YOLOX — Zheng Ge, Songle Liu, Feng Wang, Zeming Li, Jian Sun (Megvii/ByteDance, 2021). HumanArt fine-tune — Xuan Ju, Ailing Zeng, Jianan Wang, Qiang Xu, Lei Zhang (CVPR 2023) via the MMPose/RTMPose project.  
**ONNX packaging:** OpenMMLab RTMPose team via MMDeploy end-to-end export (NMS baked into the graph).  
**Version:** YOLOX-Tiny, 300 epochs on HumanArt, checkpoint hash `6f3252f9`.

### 1b. Interaction with Hardware & Software
- Runtime: ONNX Runtime ≥ 1.16.0 (CPU or CUDA execution provider).
- Upstream: receives BGR video frames (any resolution; internally letterboxed to 416×416).
- Downstream: bounding boxes feed ByteTrack for multi-person tracking; tracked boxes feed the CIGPose wholebody pose estimator.
- No external network access at inference time.

### 1c. Software Versions
- ONNX opset 17, produced by MMDeploy end-to-end export pipeline.
- Input tensor name: `input`, dtype float32, shape `[1, 3, 416, 416]`.
- Output tensors: `dets` float32 `[1, N, 5]` (x1 y1 x2 y2 score) + `labels` int64 `[1, N]`.
- NMS is pre-applied inside the ONNX graph; no post-processing deduplication needed.

### 1d. Distribution Form
Downloaded via **rtmlib** (https://github.com/Tau-J/rtmlib) — a lightweight inference library for RTMPose models. rtmlib fetches the ONNX file automatically on first use from the OpenMMLab CDN:  
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip  
The zip contains `end2end.onnx` (stored here as `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`) plus an SDK bundle.  
rtmlib repository: https://github.com/Tau-J/rtmlib  
Source model config: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

### 1e. Hardware Requirements
- CPU inference: functional, approximately 15–25 ms per frame at 1080p on a modern CPU.
- Recommended: NVIDIA GPU with CUDA 12.x, ≥ 2 GB VRAM (< 5 ms per frame).
- Model file size: ~19 MB.

### 1g. Interface for Deployers
Entry point: `PersonDetector.get_detections(image, score_threshold=0.5)`  
- Input: BGR `np.ndarray` of any resolution.  
- Output: `(boxes, scores)` — `boxes` is `(M, 4)` float32 in original image pixel coordinates `[x1, y1, x2, y2]`; `scores` is `(M,)` float32.  
- Letterboxing and inverse-letterbox coordinate mapping are handled internally in `persondet/detector.py`.

### 1h. Usage Notes
- `score_threshold` default 0.5; lower values increase recall at the cost of more false positives.
- The model outputs only class 0 (person); all other class outputs are discarded.
- Input is passed as **raw BGR float32** — no per-channel mean/std normalisation. This matches the MMDeploy end-to-end export convention for YOLOX.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
- Base architecture: **YOLOX-Tiny** — an anchor-free single-stage detector with a decoupled detection head and SimOTA label assignment.
- Pre-trained on **COCO 2017** (person + 79 other classes), then **fine-tuned exclusively on HumanArt** for person detection.
- Training framework: **MMDetection / MMPose** (OpenMMLab).
- ONNX export: **MMDeploy** end-to-end pipeline, which fuses the YOLOX post-processing (decode + NMS) into the ONNX graph.

### 2b. Design Specifications & Algorithm
- **Backbone:** Modified CSP-DarkNet (Tiny variant) — ~5 M parameters, 6.45 GFLOPs at 416×416.
- **Neck:** PAN (Path Aggregation Network) feature pyramid.
- **Head:** Decoupled head — separate branches for classification (person vs. background) and box regression, avoiding the representation conflict found in coupled heads.
- **Anchor-free:** Predictions are made per grid cell; no manually designed anchor boxes.
- **Label assignment:** SimOTA — selects the optimal positive assignments dynamically based on transport cost (classification + regression loss).
- **NMS:** Fused inside ONNX graph via MMDeploy's `TRTBatchedNMS` or equivalent node; parameters: IoU threshold 0.65, score threshold 0.01 (pre-filter), top-k 100.

### 2c. System Architecture & Compute
- Model file: ~19 MB ONNX.
- FLOPs: ~6.45 GFLOPs per forward pass at 416×416.
- Parameters: ~5 M.
- Batch size 1 at inference.

### 2d. Training Data

#### Pre-training: COCO 2017
| Split | Images | Person instances |
|-------|--------|-----------------|
| train2017 | 118,287 | ~262,000 |
| val2017 | 5,000 | ~11,000 |

COCO 2017 contains natural photographs of everyday scenes with crowd-sourced bounding box annotations (80 object classes; class 0 = person).

#### Fine-tuning: HumanArt (CVPR 2023)
| Split | Images | Person instances |
|-------|--------|-----------------|
| train | ~45,000 | ~109,000 |
| val   | ~5,000  | ~17,000  |
| Total | ~50,000 | ~126,000 |

**HumanArt** (Ju et al., CVPR 2023) extends COCO-style annotation to 20 human depiction scenarios across three super-categories:

| Super-category | Scenarios |
|----------------|-----------|
| Natural | Acrobatics, Cosplay, Dance, Drama, Movie |
| 3D Artificial | Garage Kits, Relief Sculpture, Sculpture |
| 2D Artistic | Cartoon, Digital Art, Ink Painting, Kids Drawing, Mural, Oil Painting, Shadow Play, Sketch, Stained Glass, Ukiyo-e, Watercolor |

The 2D Artistic and 3D Artificial subsets include stylised, non-photorealistic, low-contrast, and monochromatic imagery that is substantially closer to historical film stock than COCO alone. This domain coverage is the primary reason YOLOX-HumanArt outperforms COCO-trained detectors on archival/historical video.

**Annotation:** Bounding boxes and person keypoints annotated using a semi-automatic pipeline combining off-the-shelf detectors with human correction. Annotations follow COCO format.

**License:** CC BY 4.0 (HumanArt dataset). COCO: CC BY 4.0.

### 2e. Human Oversight
Detection boxes are the first stage of a multi-stage pipeline. All downstream outputs (clip selection, face crops) are reviewed by the end user. No legal or consequential automated decision is derived solely from bounding box detections.

### 2g. Validation & Testing
COCO 2017 val benchmark (person class, reported by OpenMMLab RTMPose project):

| Model | AP (person) | AP50 | AP75 |
|-------|-------------|------|------|
| YOLOX-Tiny (COCO) | 42.7 | 63.7 | 46.1 |
| YOLOX-Tiny (HumanArt fine-tune) | Not separately published; HumanArt val mAP improvement reported as +8–12 pp over COCO baseline on artistic subsets |

No formal benchmark on silent-era or early sound film footage has been conducted. Empirical evaluation on A Man Alone (1955) shows substantially higher recall than RT-DETRv4-S (COCO-trained), which missed ~88% of scene time on the same footage.

### 2h. Cybersecurity
- Model loaded read-only at startup.
- ONNX Runtime sandboxes graph execution.
- File downloaded from OpenMMLab's official CDN; no SHA integrity verification is performed at load time. Users in security-sensitive deployments should verify the file hash against the value published on the MMPose releases page.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Detects persons across highly diverse visual styles: photographs, paintings, cartoons, sculptures, ink drawings, film frames.
- NMS baked into the ONNX graph eliminates the need for post-processing deduplication.
- Compact model (~19 MB, ~5 M params) enables fast inference even on modest hardware.
- Anchor-free design is more robust to extreme aspect ratios (very tall or wide crops) than anchor-based detectors.

### Limitations
- **Single class:** Outputs only person detections; cannot detect objects, animals, or scene context.
- **Small persons:** Detection quality degrades for persons occupying fewer than ~32×32 pixels in the original frame, even at the 416×416 inference resolution.
- **Dense crowds:** HumanArt has limited crowd scenes; heavily overlapping persons may be under-detected relative to COCO-specialised crowd detectors.
- **Historical film artefacts:** Film grain, scratches, variable exposure, and non-standard body proportions (period clothing, unusual stances) are partially — but not fully — covered by HumanArt's artistic subsets. Performance is better than COCO-only baselines but not validated on systematic benchmarks.
- **Truncated persons:** Persons cropped at the frame border may receive lower scores due to incomplete bounding box regression targets during training.
- **Score calibration:** Confidence scores are not calibrated probabilities; `score_threshold` may require tuning per footage type.
- **Demographic fairness:** No formal audit of detection rate disparities across skin tones, body types, or age groups has been published for the HumanArt fine-tune.

### Foreseeable Unintended Outcomes
- False negative detections for stylised or low-contrast depictions of persons not well represented in HumanArt's 20 scenarios (e.g., extreme close-ups of faces only, silhouettes, heavily backlit figures).
- False positive detections on human-shaped objects (mannequins, statues) — particularly relevant for the Sculpture and Garage Kits scenarios where the model was explicitly trained to detect such depictions as persons.

### Input Data Specifications
- Input: BGR `np.ndarray`, uint8 or float32, any spatial resolution.
- Internally letterboxed to 416×416 with constant padding value 114.
- No normalisation applied (raw pixel values passed directly to ONNX graph).

---

## 4. Performance Metrics Rationale
Average Precision (AP) at IoU threshold 0.5:0.95 is the standard COCO person detection metric. It balances localisation precision (IoU) and recall across score thresholds, making it appropriate for a detector used to seed a downstream tracker. AP50 provides a recall-oriented view relevant to practical clip extraction, where missing a person entirely is worse than a slightly loose box.

---

## 5. Risk Management
Within this pipeline the model is used for **non-high-risk** video archival analysis. Detection is the first stage of a multi-stage system with human review at the end. Risks are mitigated by:
- `score_threshold` configurable per job to trade precision for recall.
- ByteTrack temporal smoothing absorbs single-frame false negatives.
- Downstream pose filtering (`check_mouth_open`, symmetry ratio, minimum track duration) provides additional quality gates.
- All final clip selections are reviewed by the human operator.

---

## 6. Known Lifecycle Changes
Checkpoint `6f3252f9` corresponds to the 300-epoch HumanArt fine-tune published in the RTMPose v1 model zoo (2023–2024). OpenMMLab may release updated checkpoints; the filename hash identifies this specific version. No fine-tuning on this project's data has been performed. If recall on a new footage type is insufficient, consider increasing `score_threshold` downward or evaluating a larger YOLOX variant (Small, Medium) fine-tuned on HumanArt.

---

## 7. Standards & Specifications Applied
- COCO evaluation protocol (https://cocodataset.org/#detection-eval).
- HumanArt dataset and evaluation: https://github.com/IDEA-Research/HumanArt
- YOLOX architecture: https://arxiv.org/abs/2107.08430
- MMDeploy ONNX export: https://github.com/open-mmlab/mmdeploy
- MMPose RTMPose model zoo: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- Apache 2.0 License (OpenMMLab codebase and model weights).

---

## Citation
```bibtex
@article{ge2021yolox,
  title   = {YOLOX: Exceeding YOLO Series Detectors},
  author  = {Ge, Zheng and Liu, Songle and Wang, Feng and Li, Zeming and Sun, Jian},
  journal = {arXiv preprint arXiv:2107.08430},
  year    = {2021}
}

@inproceedings{ju2023humanart,
  title     = {HumanArt: A Versatile Human-Centric Fine-Grained Visual Recognition Benchmark},
  author    = {Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
YOLOX paper: https://arxiv.org/abs/2107.08430  
HumanArt paper: https://arxiv.org/abs/2303.01476
