# System Card — Scene Change Detector

Technical documentation structured in accordance with EU AI Act Annex IV.

> **Note:** The scene change detector is a rule-based algorithm. It contains no learned weights and requires no downloaded artifacts. This card documents the algorithm as a system rather than a model file.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Hard cut detection — identifying frame boundaries where a film editor's cut transitions from one camera shot to another, so that the tracker can be reset and clip segments can be closed cleanly.  
**Implementation:** Original code in `scripts/extract_person_clips.py` (`scene_changed()` function), written for this project.  
**Version:** Two-signal design (luminance histogram + bounding box area ratio).

### 1b. Interaction with Hardware & Software
- Runs on CPU only (OpenCV + NumPy). No GPU required.
- Upstream: receives consecutive BGR video frames and the current-frame detection bounding boxes from `PersonDetector`.
- Downstream: when a cut is detected, the active `Segment` is closed and `PersonTracker.init_tracker()` is called, resetting track IDs for the new shot.
- No external network access.

### 1c. Software Versions
- OpenCV (`cv2`) for image resizing, colour conversion, histogram computation, and histogram comparison.
- NumPy for bounding box area arithmetic.

### 1d. Distribution Form
Algorithm only — no binary artifact. Full implementation in `scripts/extract_person_clips.py`.

### 1e. Hardware Requirements
- CPU-only. Each call processes two 128×72 thumbnail images; negligible compute.

### 1g. Interface for Deployers
```python
is_cut = scene_changed(
    prev_frame, curr_frame,
    hist_threshold,          # float, from config scene_change_threshold
    prev_det_bboxes,         # np.ndarray (N, 4) or empty
    curr_det_bboxes,         # np.ndarray (M, 4) or empty
    bbox_area_ratio_threshold,  # float, from config scene_change_bbox_area_ratio
)
```
Returns `True` if a hard cut is detected. Configured via `config.yaml` under `person_extraction`.

### 1h. Usage Notes
- `scene_change_threshold: 0.75` — lower values are more permissive (fewer false positives, more missed cuts). Raise toward 0.9 for footage with highly variable lighting within a single shot.
- `scene_change_bbox_area_ratio: 4.0` — only fires when detections are present in both frames. Has no effect on frames with no detected persons.
- The detector is called on every consecutive frame pair, not just when motion is detected. It adds ~0.2 ms per frame.

---

## 2. Development Elements & Process

### 2a. Development Methods & Third-Party Tools
Rule-based signal processing. No machine learning, no training data, no learned parameters.

### 2b. Design Specifications & Algorithm

Two independent signals are evaluated; a cut is declared when **either** fires.

---

**Signal 1 — Luminance histogram correlation**

Each frame is downscaled to 128×72 pixels and converted to greyscale. A 64-bin intensity histogram is computed and min-max normalised. The Pearson correlation between the previous and current frame histograms is computed using `cv2.HISTCMP_CORREL`.

```
correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
fires if correlation < scene_change_threshold  (default 0.75)
```

Within a single shot, consecutive frames share a very similar luminance distribution (correlation typically > 0.85). A hard cut introduces a completely different scene with a different distribution, dropping the correlation abruptly. Using luminance rather than colour channels ensures correct operation on both colour and greyscale footage.

**What it catches:** Any cut where the two shots differ in overall brightness distribution — the most common case.  
**What it misses:** Cuts between two shots that happen to share a similar brightness distribution (e.g., two outdoor daylight scenes with similar exposure). Signal 2 provides partial backup for this case when persons are detected.

---

**Signal 2 — Bounding box area ratio**

Only evaluated when both the previous and current frames contain at least one person detection. The area of the largest detection in each frame is compared:

```
ratio = max(area_prev / area_curr, area_curr / area_prev)
fires if ratio >= scene_change_bbox_area_ratio  (default 4.0)
```

A cut from a wide shot (person at ~10% of frame height) to a close-up (person at ~40% of frame height) produces an area ratio of approximately 16×, well above the threshold. A continuous zoom or the person walking towards the camera produces a gradual area change that stays below the threshold.

**What it catches:** Wide-to-close-up cuts (and vice versa) where the luminance distribution of the two shots may be similar.  
**What it misses:** Any cut where persons are not detected in one or both frames, or where both shots are at a similar distance.

---

### 2c. System Architecture & Compute
- No model file; stateless between calls.
- Per-call cost: two `cv2.resize` + two `cv2.cvtColor` + two `cv2.calcHist` + one `cv2.compareHist` + optional area comparison.
- Approximately 0.1–0.2 ms per frame pair on CPU.

### 2d. Training Data
Not applicable. The algorithm has no learned parameters.

### 2e. Human Oversight
Scene change decisions feed directly into clip segmentation. False positives split one shot into multiple clips (recoverable by inspection). False negatives merge two shots into one clip (detectable by reviewing output). All clips are reviewed by the human operator.

### 2g. Validation & Testing
No formal benchmark evaluation has been conducted on this specific implementation. Empirical observation on *A Man Alone* (1955) shows correct detection of hard cuts, including cuts between colour-similar indoor shots (Signal 1) and cuts between wide shots and close-ups (Signal 2).

Known failure modes observed:
- **Gradual fades / dissolves:** Signal 1 correlation drops slowly; the threshold may not be crossed in any single frame pair.
- **Flash frames:** A single white or black flash frame (common in damaged archive prints) will trigger Signal 1 and immediately re-trigger on the following frame, producing two spurious cuts. Mitigated by `merge_gap_frames: 5` in the segmentation logic.

### 2h. Cybersecurity
Pure algorithm; no model loading. No external network access. Input is locally-sourced video frames only.

---

## 3. Capabilities, Limitations & Risks

### Capabilities
- Correctly detects hard cuts on both colour and greyscale footage.
- Signal 2 specifically targets the wide/close-up transition common in classical Hollywood editing, independent of colour or luminance.
- Stateless and fast — negligible overhead relative to detection and pose estimation.
- No domain mismatch: the algorithm is not trained on any specific footage type.

### Limitations
- **Dissolves and fades:** Cross-fades (common in scene transitions in 1950s film) are not reliably detected. The luminance correlation changes gradually rather than abruptly.
- **Flash frames:** Damaged frames can trigger false cuts. Downstream `merge_gap_frames` partially mitigates this.
- **Threshold sensitivity:** A single threshold applies to all shots. Very dark scenes or high-contrast lighting (e.g., night-for-night filming) may cause false positives even within a continuous shot.
- **Signal 2 blind spot:** If the person leaves frame just before a cut and re-enters after it, Signal 2 cannot fire (no detections in one or both frames). Relies entirely on Signal 1 in this case.
- **No audio signal:** Audio-based cut detection (silence, music change) is not used and would be complementary.

### Foreseeable Unintended Outcomes
- A false positive cut mid-shot causes one valid clip to be split into two shorter clips, potentially dropping one below `min_clip_duration_seconds`.
- A false negative (missed cut) allows track IDs from one shot to persist into the next shot, producing a clip that spans two unrelated shots.

### Input Data Specifications
- Frames: BGR `np.ndarray` of any resolution (internally downscaled to 128×72).
- Bounding boxes: `np.ndarray` of shape `(N, 4)` in `[x1, y1, x2, y2]` pixel coordinates, or an empty array.

---

## 4. Performance Metrics Rationale
For a scene change detector in this pipeline, the relevant metrics are precision (false positive rate — spurious splits) and recall (false negative rate — missed cuts that bleed shot boundaries). No standardised benchmark dataset covering 1950s film stock has been identified; evaluation is currently qualitative. A useful future evaluation would be to manually annotate cut points in a representative 10-minute segment and compute precision/recall at the configured thresholds.

---

## 5. Risk Management
Within this pipeline the detector is used for non-high-risk archival video segmentation. Risks are mitigated by:
- `merge_gap_frames: 5` merges segments separated by very short gaps, absorbing flash-frame false positives.
- `min_clip_duration_seconds: 2.0` filters out trivially short clips produced by false positives.
- All output clips are reviewed by the human operator before use.

---

## 6. Known Lifecycle Changes
The two-signal design replaced an earlier three-signal design (luminance + bbox area + gradient histogram). The gradient histogram signal was removed as redundant with the luminance histogram. No further algorithmic changes are planned unless new failure modes are observed on additional footage.

---

## 7. Standards & Specifications Applied
- OpenCV histogram comparison methods: https://docs.opencv.org/4.x/d8/dc8/tutorial_histogram_comparison.html
- No published standard for scene change detection thresholds; thresholds are empirically tuned on the target footage.
