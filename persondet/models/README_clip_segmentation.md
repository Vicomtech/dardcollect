# System Card — Clip Segmentation Algorithm

Technical documentation structured in accordance with EU AI Act Annex IV.

> **Note:** The clip segmentation algorithm is rule-based. It contains no learned weights. This card documents the full decision logic that determines when a clip starts, when it ends, and whether it is kept or discarded.

---

## 1. General Description

### 1a. Intended Purpose & Provider
**Task:** Segment a continuous video file into individual person clips — discrete time intervals containing at least one tracked person with a visible, sufficiently frontal face. Each accepted clip is saved as an `.mp4` file with a companion `.json` sidecar.  
**Implementation:** `scripts/extract_person_clips.py` (`process_video()` and `flush_segments()`), written for this project.

### 1b. Interaction with Hardware & Software
- Upstream inputs: raw video frames from OpenCV `VideoCapture`, detection boxes from `PersonDetector`, tracked boxes from `PersonTracker`, pose keypoints from `PoseEstimator`, scene change signal from `scene_changed()`.
- Downstream outputs: `.mp4` clip files (encoded with libx264/aac via MoviePy) and `.json` sidecar metadata files containing per-frame bounding boxes, keypoints, and summary statistics.
- No external network access.

### 1c. Software Versions
- OpenCV for frame reading.
- MoviePy (`VideoFileClip.subclipped`) for clip extraction and re-encoding.
- SciPy `savgol_filter` for keypoint smoothing.

### 1d. Distribution Form
Algorithm only — no binary artifact. Full implementation in `scripts/extract_person_clips.py`.

### 1e. Hardware Requirements
- CPU for segmentation logic. GPU used by upstream detector and pose estimator.
- Clip re-encoding (libx264) is CPU-bound; 4 threads used by default.

### 1g. Interface for Deployers
```bash
python scripts/extract_person_clips.py
```
All parameters are read from `config.yaml` under `person_extraction`. No command-line arguments.

### 1h. Usage Notes
- `resume: true` skips videos whose progress file already exists, enabling safe interruption and restart.
- `dry_run: true` runs all logic but writes no files — useful for parameter tuning.
- Disk space is checked before every write; the process exits cleanly if free space drops below `min_free_disk_gb`.

---

## 2. Design Specifications & Algorithm

The algorithm operates as a streaming pipeline over video frames. At each frame, a sequence of gates is evaluated. Failing any gate terminates the current clip or prevents a new one from starting.

---

### Stage 0 — Detection pre-filtering (per detection, before tracking)

Before detections reach the tracker, two geometric filters remove obvious false positives:

| Filter | Parameter | Default | Rationale |
|--------|-----------|---------|-----------|
| Max bbox area | `max_bbox_area_percent` | 60% of frame | Rejects detections covering most of the frame (title cards, full-frame graphics) |
| Max aspect ratio | `max_detection_aspect_ratio` | 2.0 (width/height) | People are taller than wide; landscape-ratio boxes are furniture, animals, or artifacts |

---

### Stage 1 — Tracking

Surviving detections are passed to OC-SORT. Tracking parameters:

| Parameter | Config key | Default |
|-----------|-----------|---------|
| IoU match threshold | `tracking_score_threshold` | 0.4 |
| Frames before confirmed | `tracking_min_hits` | 3 |
| Frames before track removed | `tracking_max_time_lost` | 10 |
| Velocity direction penalty | `inertia` (hardcoded) | 0.2 |

A track must be seen in at least `tracking_min_hits` frames before it appears in output. This suppresses single-frame false positives from the detector.

---

### Stage 2 — Duplicate suppression (per frame, after tracking)

Two independent duplicate filters are applied to the set of active tracks each frame:

**IoU-based suppression** (`suppress_overlapping_tracklets`):  
If two track bounding boxes overlap above `max_track_overlap_iou` (default 0.5) or one box is substantially contained within the other (IoMin threshold), the lower-confidence track is removed. Fixes the common case of the detector producing two boxes for the same person.

**Keypoint-based suppression** (`suppress_by_keypoints`):  
After pose estimation, if two tracks have mean keypoint distance (averaged over mutually-visible keypoints, normalised by person height) below 0.15, they are considered the same person and the lower-confidence one is removed. Catches cases where the IoU filter fails because the boxes differ in size.

---

### Stage 3 — Scene change gate

On every consecutive frame pair, before updating the tracker, `scene_changed()` is evaluated. If a cut is detected:
- The current active segment is closed immediately and moved to the pending queue.
- The tracker is reset (`init_tracker()`), clearing all track IDs.
- Pending segments are flushed to disk.

This ensures track IDs never bleed across shots and that each clip corresponds to a single camera take. See [README_scene_change_detector.md](README_scene_change_detector.md) for the detection algorithm.

---

### Stage 4 — Segment open/close logic

A **segment** is a contiguous sequence of frames in which at least one tracked person is present.

- **Open:** A new segment is created on the first frame where at least one track is active.
- **Extend:** Each subsequent frame with at least one active track extends the segment's `end_frame`.
- **Close:** The segment is closed on the first frame where **no** active tracks are present. It is moved to a pending queue.

The segment accumulates per-frame counters as it grows:
- `face_visible_frames`: incremented each frame where at least one person passes the face visibility check (Stage 5).
- `mouth_open_frames`: incremented each frame where at least one visible-face person has their mouth detected as open.

---

### Stage 5 — Per-frame face visibility check

For each tracked person, pose keypoints are used to determine whether a usable face is visible. This check runs every frame and feeds the segment counters; it does **not** close a clip mid-stream if a face disappears.

**Step 5a — Minimum keypoints present**  
Requires nose score ≥ `pose_keypoint_threshold` AND at least one eye (left or right) score ≥ threshold.  
→ Rejects persons facing away from camera or with no face region detected.

**Step 5b — Anti-hallucination eye distance check**  
If both eyes are visible, their pixel distance must be ≥ 1% of image height.  
→ Rejects cases where the pose model places both eye keypoints on the torso (common on small/distant persons).

**Step 5c — Minimum face size**  
The bounding box of the visible face keypoints (nose, eyes, ears) is computed. Its height, scaled by 2.5× to approximate full face height, must be ≥ `min_face_size_percent` × image height (default 10%).  
→ Rejects persons who are too far from camera to be useful.

**Step 5d — Frontal face check** (if `require_frontal_face: true`)  
Requires nose and both eyes to score ≥ threshold (eyes are almost never detected on the back of a head, so this already rules out rear-facing persons). Ear handling is adaptive:

- **Both ears visible:** computes nose-to-ear symmetry ratio as before.
- **One ear visible:** passes if the visible ear's horizontal distance from the nose is at least `(1 − frontal_symmetry_threshold)` × estimated face width. A sharply profile-facing person places the visible ear very close to the nose; this check catches that case.
- **No ears visible:** passes (eyes + nose already confirm a roughly forward-facing pose; ears may simply be obscured by hair).

Full-symmetry formula when both ears are present:

```
ratio = min(|left_ear_x − nose_x|, |right_ear_x − nose_x|)
      / max(|left_ear_x − nose_x|, |right_ear_x − nose_x|)
```

The face is considered frontal if `ratio ≥ frontal_symmetry_threshold` (default 0.65). A profile view gives a ratio near 0; a frontal view gives a ratio near 1.

**Rationale for relaxing the ear requirement:** requiring both ears detects at most moderately-turned faces but incorrectly rejects women whose hair conceals one or both ears — a systematic bias on period film material.

→ Rejects persons shown in profile or at a large yaw angle.

**Step 5e — Mouth open detection** (if `enable_visual_speaking: true`)  
Uses CIGPose COCO-133 keypoints: upper lip (index 85) and lower lip (index 89). Both must score ≥ threshold. The mouth-open ratio is computed as:

```
mouth_dist = |upper_lip_y − lower_lip_y|
eye_dist   = |left_eye_y − right_eye_y| (or fallback distance)
ratio      = mouth_dist / eye_dist
```

Mouth is considered open if `ratio > open_threshold` (default 0.05). This is a proxy for speech activity; see [README_cigpose-m_coco-wholebody_256x192.md](README_cigpose-m_coco-wholebody_256x192.md) for lip keypoint reliability limitations.

---

### Stage 6 — Segment merging

Before filtering, adjacent segments in the pending queue are merged if the gap between them is ≤ `merge_gap_frames` (default 5 frames, ≈ 0.2 s at 24 fps).  
→ Absorbs brief detection gaps caused by flash frames, slight occlusion, or borderline-threshold frames within a continuous take.

---

### Stage 7 — Segment-level filters (applied at flush time)

These filters operate on complete segments, not individual frames:

| Filter | Parameter | Default | Condition |
|--------|-----------|---------|-----------|
| Minimum consecutive frames | `min_consecutive_frames` | 10 | `segment.frame_count >= 10` |
| Minimum duration | `min_clip_duration_seconds` | 2.0 s | `segment.duration >= 2.0 s` |
| Minimum face-visible frames | `min_face_visible_frames` | 15 | `segment.face_visible_frames >= 15` (only if `require_face_visibility: true`) |
| Minimum consecutive face-visible frames | `min_consecutive_face_frames` | 5 | `segment.max_consecutive_face_frames >= 5` (only if `require_face_visibility: true`) |

The two face filters are complementary: `min_face_visible_frames` ensures enough total coverage; `min_consecutive_face_frames` ensures those frames are not isolated single-frame blips scattered across the clip.

A segment failing any of these is silently discarded — no file is written.

---

### Stage 8 — Maximum duration splitting

If a segment exceeds `max_clip_duration_seconds` (default 60 s), it is split into consecutive sub-clips of at most 60 s each. Face and mouth frame counts are proportionally distributed across splits.

---

### Stage 9 — Keypoint smoothing

Before the sidecar JSON is written, each track's keypoint trajectory within the segment is smoothed using a **Savitzky-Golay filter** (window 0.25 s, polynomial order 2). This removes frame-to-frame jitter from the pose estimator without smearing motion onsets.  
Smoothing is skipped for segments shorter than 5 frames, or for individual keypoints that are consistently below the confidence threshold across the segment (noise/hallucination suppression).

---

### Stage 10 — Output

For each accepted segment:
- `.mp4` clip extracted from the source video using `start_frame` / `end_frame` boundaries (re-encoded with libx264/aac).
- `.json` sidecar written with:
  - Clip metadata: `source_video`, `start_frame`, `end_frame`, `start_seconds`, `end_seconds`, `duration_seconds`, `fps`, `video_info`, `max_persons`, `unique_tracks`, `face_visible_frames`, `mouth_open_frames`.
  - Per-frame data (`frame_data`): for each frame, a list of `{track_id, bbox, score, keypoints, keypoint_scores}`.

---

## 3. Full Decision Flow (Summary)

```
For each frame:
  ├─ Pre-filter detections (area, aspect ratio)
  ├─ Scene change check → if cut: close segment, reset tracker, flush
  ├─ Track detections (OC-SORT)
  ├─ IoU duplicate suppression
  ├─ Run pose estimation on all tracks
  ├─ Keypoint duplicate suppression
  ├─ For each track: face visibility check (size, anti-hallucination, frontal)
  ├─ For each visible-face track: mouth open check
  ├─ If tracks present: open/extend segment, accumulate counters
  └─ If no tracks: close segment → pending queue

At flush:
  ├─ Merge segments with gap ≤ merge_gap_frames
  ├─ Filter: min frames, min duration, min face-visible frames, min consecutive face frames
  ├─ Split segments exceeding max duration
  ├─ Smooth keypoints (Savitzky-Golay)
  └─ Write .mp4 + .json for each accepted segment
```

---

## 4. Configuration Reference

All parameters are in `config.yaml` under `person_extraction`:

| Parameter | Default | Stage | Effect |
|-----------|---------|-------|--------|
| `detection_threshold` | 0.4 | 0 | Min detector confidence |
| `max_bbox_area_percent` | 60% | 0 | Max detection size |
| `max_detection_aspect_ratio` | 2.0 | 0 | Max width/height ratio |
| `max_track_overlap_iou` | 0.5 | 2 | IoU duplicate suppression |
| `tracking_score_threshold` | 0.4 | 1 | IoU match threshold |
| `tracking_min_hits` | 3 | 1 | Frames to confirm a track |
| `tracking_max_time_lost` | 10 | 1 | Frames before track removed |
| `pose_keypoint_threshold` | 0.4 | 5 | Min keypoint confidence |
| `require_face_visibility` | true | 5/7 | Enable face checks |
| `min_face_size_percent` | 10% | 5c | Min face height |
| `require_frontal_face` | true | 5d | Enable frontal check |
| `frontal_symmetry_threshold` | 0.65 | 5d | Min ear-nose symmetry |
| `enable_visual_speaking` | true | 5e | Enable mouth open check |
| `scene_change_detection` | true | 3 | Enable cut detection |
| `scene_change_threshold` | 0.75 | 3 | Luminance correlation threshold |
| `scene_change_bbox_area_ratio` | 4.0 | 3 | Bbox area ratio threshold |
| `merge_gap_frames` | 10 | 6 | Max gap to merge segments (matches `tracking_max_time_lost`) |
| `min_consecutive_frames` | 10 | 7 | Min frames per clip |
| `min_clip_duration_seconds` | 2.0 s | 7 | Min clip duration |
| `min_face_visible_frames` | 15 | 7 | Min total frames with visible face |
| `min_consecutive_face_frames` | 5 | 7 | Min unbroken run of face-visible frames |
| `max_clip_duration_seconds` | 60 s | 8 | Max clip duration before split |

---

## 5. Capabilities, Limitations & Risks

### Capabilities
- Handles arbitrary-length videos; processes frames in a streaming fashion with bounded memory.
- Resumable: progress is checkpointed every 30 s of video; interrupted runs continue from the last checkpoint.
- All face checks are pose-based (keypoints), not pixel-based face detection — robust to grainy or low-resolution footage where CNN face detectors struggle.

### Limitations
- **Frontal check is yaw-only:** The symmetry check measures horizontal rotation (yaw). Pitch (looking up/down) and roll (head tilt) are not assessed. A person looking sharply downward may pass the frontal filter.
- **Mouth open proxy:** `mouth_open_frames` is a detection count, not a speaking duration. Silent mouth movements (chewing, breathing) may increment the counter; closed-mouth speech does not.
- **Merge gap is global:** `merge_gap_frames` (default 10, matching `tracking_max_time_lost`) applies uniformly. A genuine scene cut followed immediately by the same person re-entering the frame within 10 frames would be incorrectly merged — though the scene change detector would have closed and flushed the segment first if it fired.
- **Split segments lose context:** When a long clip is split at Stage 8, the face and mouth frame counts are only approximately preserved (proportional split). The sidecar of a sub-clip does not reference the other sub-clips.
- **Historical film:** Pose keypoint reliability on pre-1960 film stock is below published benchmarks. The face checks depend entirely on CIGPose keypoint quality. See [README_cigpose-m_coco-wholebody_256x192.md](README_cigpose-m_coco-wholebody_256x192.md).

### Foreseeable Unintended Outcomes
- A person in partial profile whose opposite ear is hallucinated at high confidence may pass the frontal filter.
- Very small or very distant persons may generate noise keypoints that pass individual confidence thresholds but fail the anti-hallucination eye-distance check, correctly rejecting the frame.
- A clip split at the maximum duration boundary may cut mid-utterance.

---

## 6. Risk Management
All output clips are reviewed by the human operator via the detection viewer before use. The sidecar JSON preserves per-frame keypoints and scores, allowing the operator to verify the basis for each face-visibility decision. No automated selection decision downstream relies solely on this algorithm without human review.

---

## 7. Related System Cards
- Detection: [README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md](README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md)
- Tracking: [README_ocsort.md](README_ocsort.md)
- Pose estimation: [README_cigpose-m_coco-wholebody_256x192.md](README_cigpose-m_coco-wholebody_256x192.md)
- Scene change detection: [README_scene_change_detector.md](README_scene_change_detector.md)
