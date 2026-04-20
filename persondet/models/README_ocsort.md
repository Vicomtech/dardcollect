# System Card — OC-SORT Tracker

Technical documentation structured in accordance with EU AI Act Annex IV.

> **Note:** OC-SORT is a model-free algorithm. It contains no learned weights and requires no downloaded artifacts. This card documents the algorithm as a system rather than a model file.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Multi-object tracking — associating person bounding boxes across consecutive video frames, maintaining stable identities (track IDs) through occlusions and brief missed detections.  
**Algorithm authors:** Jiangmiao Pang, Linlu Qiu, Xia Li, Haofei Chen, Qi Liu, Jiaqi Wang, Trevor Darrell (UC Berkeley / Shanghai AI Laboratory, CVPR 2023).  
**Implementation:** Original code in `persondet/tracker.py`, written for this project. Not a copy of the reference implementation.  
**Version:** OC-SORT as described in the CVPR 2023 paper; no modifications to the core algorithm.

### 1b. Interaction with Hardware & Software
- Runs on CPU only (pure NumPy / SciPy). No GPU required.
- Optional dependency: `lap` (linear assignment; falls back to greedy matching if unavailable), `cython_bbox` (IoU computation; falls back to NumPy).
- Upstream: receives `(boxes, scores)` from `PersonDetector.get_detections()` each frame.
- Downstream: supplies tracked `Tracklet` objects (with stable `track_id`) to the keypoint estimator and clip segmentation logic.
- No external network access.

### 1c. Software Versions
- Pure Python / NumPy implementation; no ONNX or PyTorch dependency.
- `scipy.linalg` used for Cholesky-based Kalman update (numerical stability).

### 1d. Distribution Form
Algorithm only — no binary artifact. The full implementation is `persondet/tracker.py` in this repository.  
Reference implementation (not used directly): https://github.com/noahcao/OC_SORT  
Paper: https://arxiv.org/abs/2203.14360

### 1e. Hardware Requirements
- CPU-only. Negligible compute relative to detection and pose estimation.
- Memory: O(T) where T is the number of active + lost tracks (typically < 20 per scene).

### 1g. Interface for Deployers
```python
tracker = PersonTracker()
tracker.init_tracker()
active_tracks = tracker.update(det_bboxes, det_scores, params=TrackingParams(...))
# active_tracks: List[Tracklet], each with .track_id, .tlbr, .det_score
```
`TrackingParams` fields: `score_threshold` (IoU match threshold), `min_hits` (frames before a track is confirmed), `max_time_lost` (frames before a lost track is removed), `inertia` (velocity direction penalty weight).

### 1h. Usage Notes
- `inertia=0` disables velocity direction consistency, reducing to near-SORT behaviour.
- A track must survive `min_hits` frames before appearing in output, suppressing single-frame false positives.
- On scene change, `init_tracker()` must be called to reset state and the track ID counter.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
OC-SORT extends the SORT family of trackers (Bewley et al., 2016) with two observation-centric corrections to the Kalman filter. No training is involved; the algorithm is deterministic given its input.

### 2b. Design Specifications & Algorithm

**State representation:** Each track maintains an 8-dimensional Kalman state `[cx, cy, a, h, vcx, vcy, va, vh]` — centre position, aspect ratio, height, and their velocities.

**Per-frame update cycle:**
1. Kalman predict — all active and lost tracks advance one time step.
2. **Stage 1** — Hungarian matching of active tracks to all detections using IoU cost.
3. **Stage 2** — Hungarian matching of lost tracks to remaining detections using IoU cost **plus velocity direction cost** (OC-SORT addition).
4. **Stage 3** — Hungarian matching of unconfirmed tracks to remaining detections using IoU cost.
5. Unmatched active tracks → lost pool. Unmatched detections → new tracks.
6. Lost tracks exceeding `max_time_lost` → removed.

**Velocity direction consistency (OCM):**  
For each lost track `i` and candidate detection `j`, a penalty is added to the IoU cost:

```
direction_cost[i,j] = inertia × (1 − cos_sim(velocity_i, direction(last_obs_i → center_j))) / 2
```

This penalises assignments where the candidate detection lies in the opposite direction of the track's estimated motion. `inertia` controls the strength (default 0.2).

**Observation-Centric Re-Update (ORU):**  
When a lost track is re-matched after a gap of G frames, the Kalman state is corrected by:
1. Re-initialising the filter from the last *actual* observation (not the drifted prediction).
2. Predicting forward G−1 steps.
3. Applying the new measurement.

This corrects velocity drift that accumulates during the gap and prevents the Kalman filter from "snapping" to a wrong position on re-association.

### 2c. System Architecture & Compute
- No model file; all state is in-memory Python objects.
- Per-frame cost: O(T × D) IoU computations where T = active tracks, D = detections. Typically < 1 ms per frame.

### 2d. Training Data
Not applicable. OC-SORT is a deterministic algorithm; it contains no learned parameters.

### 2e. Human Oversight
Track assignments are deterministic given detector outputs. All downstream clip selection is reviewed by the human operator.

### 2g. Validation & Testing
Published benchmark results (from the OC-SORT paper, MOT17 test set):

| Metric | OC-SORT | ByteTrack |
|--------|---------|-----------|
| HOTA ↑ | 63.9 | 63.1 |
| MOTA ↑ | 78.0 | 80.3 |
| IDF1 ↑ | 77.5 | 77.3 |
| ID Sw. ↓ | 1950 | 2196 |

OC-SORT trades a small MOTA loss for fewer ID switches, which is the relevant metric for this pipeline (stable clip-length track IDs matter more than frame-level accuracy).

No formal evaluation on silent-era or early sound film footage has been conducted.

### 2h. Cybersecurity
Pure algorithm; no model loading surface. No external network access.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Maintains stable track IDs through brief occlusions (up to `max_time_lost` frames).
- Velocity-consistent matching reduces ID switches when people cross paths.
- ORU corrects Kalman drift on re-association, producing more accurate bounding boxes after a gap.
- No ReID model required — no domain mismatch with historical film appearance.

### Limitations
- **Pure motion tracker:** relies entirely on bounding box IoU and velocity. Two people who swap positions while occluded will produce an ID switch.
- **No appearance cues:** cannot re-identify a person who leaves and re-enters the frame (a new track ID will be assigned).
- **Initialisation instability:** velocity estimates are unavailable for the first two observations; direction consistency cannot help during this window.
- **Detector dependency:** tracking quality is bounded by detection quality. Oscillating bounding box sizes (e.g. partial vs full body detection) reduce IoU and can break association even when the person is continuously visible.
- **Camera motion:** OC-SORT has no camera motion compensation. Fast pans can cause track drift; BoT-SORT would be more appropriate for such footage.
- **Historical film:** no published evaluation exists on pre-1960 film stock.

### Foreseeable Unintended Outcomes
- ID switches during occlusions in crowded scenes (multiple people overlapping).
- Short spurious tracks from background false-positive detections passing `min_hits`.

### Input Data Specifications
- `det_bboxes`: list of `[x1, y1, x2, y2]` float bounding boxes in original image pixel coordinates.
- `det_scores`: list of float confidence scores in `[0, 1]`.

---

## 4. Performance Metrics Rationale
HOTA (Higher Order Tracking Accuracy) jointly measures detection and association quality, making it the most representative single metric for this pipeline where both correct detection and correct ID assignment matter. IDF1 specifically measures ID consistency over time, which directly corresponds to stable track IDs for clip extraction. ID Switches is the most actionable diagnostic metric.

---

## 5. Risk Management
Within this pipeline the tracker is used for non-high-risk archival video analysis. Risks are mitigated by:
- `min_hits` suppresses single-frame spurious tracks.
- `max_time_lost` prevents indefinite track persistence.
- Post-processing (`suppress_overlapping_tracklets`, `suppress_by_keypoints`) removes duplicate tracks produced by the tracker.
- All clip selections are reviewed by the human operator.

---

## 6. Known Lifecycle Changes
This implementation follows the CVPR 2023 paper. The reference repository (github.com/noahcao/OC_SORT) may receive updates; this implementation would need manual review to incorporate them. No fine-tuning or learned components are present, so there is no model update cycle.

---

## 7. Standards & Specifications Applied
- MOT Challenge evaluation protocol (https://motchallenge.net).
- HOTA metric: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking", IJCV 2021.
- SORT: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.

---

## Citation
```bibtex
@inproceedings{pang2023ocsort,
  title     = {Observation-Centric SORT: Rethinking SORT for Robust
               Multi-Object Tracking},
  author    = {Pang, Jiangmiao and Qiu, Linlu and Li, Xia and Chen, Haofei
               and Liu, Qi and Wang, Jiaqi and Darrell, Trevor},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
Paper: https://arxiv.org/abs/2203.14360
