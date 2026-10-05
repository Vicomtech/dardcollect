# Design Doc — Face-Crop Corner-Trajectory Stabilization (issue #9)

**Status:** approved (queue execution 2026-09-09; default ON since 2026-09-30 — user decision: crops must never wobble). **Revised 2026-10-05:** smoothing the corner *trajectory* instead of freezing the per-track *median* — the median removed wobble but left the eyes off the canonical OFIQ positions in moving tracks (user: "que estén centrados en los ojos pero sin ruido").
**Issue:** #9 (agkanlis, feat/crop-stabilization reference implementation)
**Modality:** video · **Stage:** face_crop_extraction · **CSVs/sidecars:** none new
**CPU/GPU:** CPU-only (Savitzky-Golay over corner series) · **Resumability:** unchanged (existing `.done` sentinels)

## 1. Problem statement (what + why)

Face crops are warped per frame from the detected face-crop quad; residual
sub-keypoint jitter (the tracker's Savitzky-Golay smoothing already removes most
frame-to-frame noise, but corner recomputation amplifies what remains) makes the
OFIQ crop background wobble.

The first fix (median per track) killed the wobble by rendering every frame
through one constant quad. That freezes the crop's position: in any track where
the face actually moves/grows/turns relative to the camera, frames far from the
median leave the real eyes off the canonical OFIQ landmarks (x=251/364, y=272),
degrading the quality measures and the MagFace identity crop. The stabilization
had **no bound on that misalignment** — the "tolerance" was effectively
unlimited for moving tracks.

The correct target is both properties at once: **eyes centred on their real
landmarks** (follow the face) **and no frame-to-frame wobble** (kill the noise).

## 2. Architecture (which stages, new vs existing)

Integration into the existing `face_crop_extraction` stage — no new stage, no
new outputs:

- Helpers in `dardcollect/face_stabilization.py` (split out of
  `face_geometry.py` 2026-10-05 to keep that file under the god-file cap):
  - `smooth_track_corners()`: per track, Savitzky-Golay low-pass filter of the
    corner trajectory (window = `stabilization_window_seconds * fps`), returning
    the quad to render *each frame* with. Gap frames are linearly interpolated
    for the filter input; the output stays `None` on interior gaps (no crop
    that frame). Returns all-`None` when fewer than `stabilization_min_frames`
    valid corners exist (per-frame fallback).
  - `compute_track_mean_corners()`: the track's median quad — retained as the
    smoothed trajectory's reference position (robust to outlier frames).
  - `plan_track_stabilization()`: pure over the corner series — smooths each
    track and computes the residual (max/mean px) of the raw corners vs the
    median, for log/sidecar observability.
  - `_valid_crop_corners()`: per-detection inclusion rule (corners available,
    no bbox overlap beyond `max_overlap_iou`) shared by the per-frame rendering
    path and the stabilization plan so both apply the exact same rule.
  - `plan_stabilized_track_crops()` (pass 1): corner-only plan over the clip's
    sidecar JSON — no decode, no pixels held; records per-frame corners (indexed
    by relative frame id) and the per-track smoothed series.
  - `render_stabilized_track_frames()` (pass 2): re-decodes the clip once and
    renders each detection through its **own smoothed quad** (per-frame corners
    for fallback tracks). Holds only the current source frame — O(1) memory.
- Applied in `face_crops.py` `process_video()` when
  `face_config.stabilize_face_crops` is true: each output frame's warp uses its
  smoothed quad. The face-crop sidecar's `frame_data` keypoints/bbox use that
  **same render warp** (smoothed per-frame when engaged, raw per-frame
  otherwise), so overlaid annotations coincide with the rendered pixels —
  storing a per-frame re-estimated alignment instead misaligned overlays by
  ~2–14 px on a 616 px canvas (fixed 2026-10-02). The person-clip sidecar
  corners upstream stay raw per-frame (the plan input).
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

**Why smoothing, not median (2026-10-05):** the median is a single constant
quad per track — zero wobble, but it cannot follow real motion. Smoothing the
trajectory keeps the zero-wobble property (high-frequency jitter is filtered)
while each frame keeps its own slowly-moving quad, so the eyes stay on their
real landmarks. On a static track the smoothed series converges to ~the median,
so nothing is lost there.

## 3. FAIR impact

None: no new CSVs, no new required sidecar fields. The face-crop sidecar
records the rendering choice for provenance: `stabilized` (bool),
`render_quad_median` (reference quad), `render_quad_residual_px`
(max/mean deviation vs the median, observability) and
`stabilization_window_seconds`. Provenance chain unchanged.

## 4. Semantics decision (the #9 design question, revised 2026-10-05)

Person-clip sidecar `face_crop_corners_ofiq` stays **raw per-frame** (what was
measured) — it is the stabilization plan's input. The **face-crop** sidecar's
`frame_data` keypoints/bbox live in output-crop pixel space, so they use the
**render warp** (smoothed per-frame when stabilization engaged, raw per-frame
otherwise); anything else draws misaligned overlays. The sidecar records the
choice explicitly: `stabilized` (bool) + `render_quad_median` +
`render_quad_residual_px` + `stabilization_window_seconds`.

## 5. Resumability

Unchanged: same `.done` sentinels, same per-track skip. Toggling the flag (or
regenerating after changing the smoothing window) requires deleting the affected
crops' outputs (`.mp4` + `.json`) to re-render (documented in
docs/0-GETTING-STARTED.md).

## 6. Config (default ON since 2026-09-30; OFF = legacy per-frame rendering)

```yaml
face_crop_extraction:
  stabilize_face_crops: true              # corner-trajectory Savitzky-Golay
  stabilization_min_frames: 5             # min stable-corner frames to engage
  stabilization_window_seconds: 0.4       # SavGol window; larger = smoother,
                                          # slower to follow genuine head motion
```

## 7. Test plan

- Unit (synthetic, CPU-only): jittered corners → smoothed trajectory wobble
  (frame-to-frame std) far below raw while the mean stays on centre; a slow
  translation is followed to its endpoint (unlike the old median); interior
  gaps stay `None`; short tracks (< min_frames) fall back to per-frame corners.
- Fixture gate: run with default ON — no drift beyond GPU noise.

## 8. MagFace recalibration note (deferred)

The reporter measured MagFace gains at threshold 15 on their corpus. Recalibrating
`quality_threshold` is a dataset-level decision recorded in the issue, not in
this chunk (thresholds stay user-owned config).
