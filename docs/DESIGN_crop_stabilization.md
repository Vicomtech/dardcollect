# Design Doc — Corner-Only Face-Crop Stabilization (issue #9)

**Status:** approved (queue execution 2026-09-09; default ON since 2026-09-30 — user decision: crops must never wobble; originally opt-in, default OFF for golden-neutrality)
**Issue:** #9 (agkanlis, feat/crop-stabilization reference implementation)
**Modality:** video · **Stage:** face_crop_extraction · **CSVs/sidecars:** none new
**CPU/GPU:** CPU-only · **Resumability:** unchanged (existing `.done` sentinels)

## 1. Problem statement (what + why)

Face crops are warped per frame from the detected face-crop quad; residual
sub-keypoint jitter (the tracker's Savitzky-Golay smoothing already removes most
frame-to-frame noise, but corner recomputation amplifies what remains) makes the
OFIQ crop background wobble. The reporter measured: corner-only stabilization
with per-track constants improves MagFace unified_score (mean +0.90) and
increases pass count at threshold 15 (750 vs 987→750 pass-drop recovered).

## 2. Architecture (which stages, new vs existing)

Integration into the existing `face_crop_extraction` stage — no new stage, no
new outputs:

- Helpers in `dardcollect/face_geometry.py`:
  - `compute_track_mean_corners()`: per track, median of the per-frame OFIQ
    corners (robust vs outliers) across all frames where corner computation
    succeeds; returns None when fewer than `stabilization_min_frames` stable
    corners exist (per-frame fallback).
  - `_valid_crop_corners()`: per-detection inclusion rule (corners available,
    no bbox overlap beyond `max_overlap_iou`) shared by the per-frame rendering
    path and the stabilization plan so both apply the exact same rule.
  - `plan_stabilized_track_crops()` (pass 1): corner-only plan over the clip's
    sidecar JSON — no decode, no pixels held; records per-frame corners and
    computes the per-track median quad.
  - `render_stabilized_track_frames()` (pass 2): re-decodes the clip once and
    renders each detection through its track-median quad (per-frame corners for
    fallback tracks). Holds only the current source frame — O(1) memory.
- Applied in `face_crops.py` `process_video()` when
  `face_config.stabilize_face_crops` is true: each output frame's warp uses the
  track-median corners instead of the per-frame corners. The face-crop
  sidecar's `frame_data` keypoints/bbox use that **same render warp** (median
  when engaged, per-frame otherwise), so overlaid annotations coincide with
  the rendered pixels — storing a per-frame re-estimated alignment instead
  misaligned overlays by ~2–14 px on a 616 px canvas (fixed 2026-10-02; the
  person-clip sidecar corners upstream stay raw per-frame, the plan input).
- Images (`process_image`) are single-frame — unaffected.

**Why 2-pass (2026-09-16):** the original single-pass design retained every
full-resolution source frame in memory while stabilizing at write time
(`n_frames × W × H × 3` — ~11 GB for a 60 s 1080p clip, the repo's
`max_clip_duration_seconds` default). The 2-pass redesign costs one extra clip
decode (cheap vs the upstream GPU detection) and is memory-bounded by one frame.
It also fixes a latent fallback bug: with the flag on but fewer than
`stabilization_min_frames` stable corners, the old code wrote the full-resolution
source frames as "616×616" crops; the fallback track is now rendered per-frame
through its own corners like the default path.

## 3. FAIR impact

None: no new CSVs, no new sidecar fields (corners recorded in the sidecar are
the raw per-frame values; stabilization is a rendering-time parameter).
Provenance unchanged; the config key itself is the provenance of the rendering
choice.

## 4. Semantics decision (the #9 design question, revised 2026-10-02)

Person-clip sidecar `face_crop_corners_ofiq` stays **raw per-frame** (what was
measured) — it is the stabilization plan's input. But the **face-crop**
sidecar's `frame_data` keypoints/bbox live in output-crop pixel space, so
they must use the **render warp** (track-median when stabilization engaged,
per-frame otherwise); anything else draws misaligned overlays. The sidecar
records the choice explicitly: `stabilized` (bool) + `render_quad_median`
(the median quad, when engaged). The original "sidecar stays per-frame"
invariant was wrong for output-space annotations and caused the reported
filtered-crop keypoint misalignment.

## 5. Resumability

Unchanged: same `.done` sentinels, same per-track skip. Toggling the flag
requires deleting the affected crops' `.done` sentinels to re-render (documented
in docs/0-GETTING-STARTED.md).

## 6. Config (default ON since 2026-09-30; OFF = legacy per-frame rendering)

```yaml
face_crop_extraction:
  stabilize_face_crops: true              # corner-only, per-track median
  stabilization_min_frames: 5             # min stable-corner frames to engage
```

## 7. Test plan

- Unit (synthetic, CPU-only): jittered corners around a fixed rect → median
  corners are constant, output frames identical to the un-jittered render;
  short tracks (< min_frames) fall back to per-frame corners; default OFF
  reproduces current output byte-for-byte (golden-neutral).
- Fixture gate: run with default OFF — no drift beyond GPU noise.

## 8. MagFace recalibration note (deferred)

The reporter measured MagFace gains at threshold 15 on their corpus. Recalibrating
`quality_threshold` is a dataset-level decision recorded in the issue, not in
this chunk (thresholds stay user-owned config).