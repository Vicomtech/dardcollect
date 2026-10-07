# Design Doc — Face-Crop Corner-Trajectory Stabilization (issue #9)

**Status:** approved (queue execution 2026-09-09; default ON since 2026-09-30 — user decision: crops must never wobble). **Revised 2026-10-05:** smoothing the corner *trajectory* instead of freezing the per-track *median* — the median removed wobble but left the eyes off the canonical OFIQ positions in moving tracks (user: "que estén centrados en los ojos pero sin ruido"). **Revised 2026-10-06:** the single Savitzky-Golay pass left a visible frame-to-frame wobble; the filter is now applied as a **two-pass cascade**, plus a **robust rate limit** and a **tolerance band** for the extreme cases (calibrated on RAVDESSfake).
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

**Why two passes, not one (2026-10-06):** the user still saw frame-to-frame
wobble after the 2026-10-06 reprocess. Measured on 519 RAVDESS tracks (motion
decomposed in output-crop pixels and validated against the rendered videos), the
single pass at the shipped 0.4 s window left 1.49 px/frame of high-frequency
crop acceleration. Two candidates were measured against the single pass:

| Variant | Wobble accel (px/frm) | Eye tracking error vs raw (mean / max, px) |
| :--- | :--- | :--- |
| single pass, 0.4 s (current) | 1.49 | 0.7 / 4.0 |
| single pass, 0.8 s | 0.62 | 1.7 / 6.6 |
| single pass, 1.2 s | 0.38 | 2.4 / 8.3 |
| One-Euro adaptive (zero-phase, swept) | 0.46–1.20 | 3.5 / 10.2 best |
| **two-pass cascade, 0.4 s** | **0.46** | **0.8 / 4.1** |
| two-pass cascade, 0.5 s | 0.22 | 1.2 / 5.2 |
| two-pass cascade, 0.6 s | 0.17 | 1.4 / 5.6 |

A velocity-adaptive (One-Euro) filter did **not** beat a single SavGol pass on
this data. The two-pass cascade does: applying the same low-pass twice squares
its magnitude response (sharper roll-off), so it removes ~3.2x more jitter than
one pass at the same 0.4 s window **with the same eye-tracking error**; a single
pass would need ~1.2 s to reach that jitter, paying >2x the tracking error. The
cascade is still linear and zero-phase (no lag), and the window remains the only
knob — the pass count is a fixed implementation constant, not a mode.

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
                                          # (applied twice: 2-pass cascade)
  stabilization_min_frames: 5             # min stable-corner frames to engage
  stabilization_window_seconds: 0.4       # SavGol window; larger = smoother,
                                          # slower to follow genuine head motion
  stabilization_max_step_median_factor: 5.0  # clip corner steps to N x track
                                             # median (extreme jumps; 0 = off)
  stabilization_band_tolerance_px: 0.0    # tolerance band on crop translation
                                          # (eye budget; 0 = off)
  stabilization_band_activate_px: 0.9     # apply the band only above this wobble
```

## 7. Extreme-case handling — robust rate limit + tolerance band (2026-10-06)

**Evidence.** Across 519 RAVDESSfake tracks the wobble is heavily tailed
(p50 0.65, p90 1.48, p95 2.14, p99 7.1, max 17.4 px/frame). The extreme cases
(cascade accel > 2 px/frame, 5.8 %) are **detection jumps**, not continuous
jitter: their largest raw corner step is ~76 px vs ~33 px on normal tracks.
Neither a tolerance band nor a median/Hampel pre-filter fixes them; a
**rate limit** does.

**Stages** (all in `plan_track_stabilization`, before the render):

1. `rate_limit_corners` — clip each frame's corner step to
   `stabilization_max_step_median_factor` x the track's median step (5.0).
   Cuts the extreme wobble ~3x; normal tracks are essentially unchanged (their
   steps are below the limit). Default ON, 0 disables.
2. `smooth_track_corners` — the two-pass SavGol cascade (§2).
3. `band_crop_trajectory` — the min-curvature **tolerance band** on the crop
   translation (user idea: allow the eyes to sit up to `tol` off canonical so
   the window can stay still). Solved with bound-constrained least squares
   (`lsq_linear`: `min ||D2 r||^2 s.t. |r-P| <= tol`). A switch/hysteresis
   controller was measured to make the wobble **worse** (velocity jumps at the
   recenter); the smooth band solve is what works. Applied only when the
   cascade still wobbles more than `stabilization_band_activate_px`, so stable
   tracks keep their exact alignment. Rotation/scale stay from the cascade so
   the quad remains a similarity. Default OFF; the RAVDESSfake config sets
   `tolerance=6 px`, `activate=0.9 px/frame`.

**Calibration (519 RAVDESSfake tracks, output px/frame; eye = max deviation):**

| Case | tracks | cascade | +rate-limit(5) | +band(6) |
| :--- | :--- | :--- | :--- | :--- |
| normal (<1.0) | 75 % | 0.51 · eye 4.1 | 0.42 · eye 4.9 | band gated off |
| moderate (1–2) | 20 % | 1.32 · eye 4.1 | 0.80 · eye 13.6 | 0.27 · eye 13.9 |
| extreme (>2) | 5.8 % | 4.05 · eye 4.1 | 1.34 · eye 19.3 | 0.78 · eye 18.2 |

The gated band leaves the 75 % stable tracks with their exact (cascade)
alignment and targets only the tail. `stabilization_band_px` records the
tolerance actually applied per track in the sidecar.

## 8. Test plan

- Unit (synthetic, CPU-only): jittered corners → smoothed trajectory wobble
  (frame-to-frame std) far below raw while the mean stays on centre; a slow
  translation is followed to its endpoint (unlike the old median); interior
  gaps stay `None`; short tracks (< min_frames) fall back to per-frame corners;
  the 2-pass cascade removes more jitter than one pass at the same window;
  the rate limiter caps detection jumps; the band stays within tolerance,
  preserves the similarity quad, and honours its activation gate.
- Fixture gate: run with default ON — no drift beyond GPU noise.

## 9. MagFace recalibration note (deferred)

The reporter measured MagFace gains at threshold 15 on their corpus. Recalibrating
`quality_threshold` is a dataset-level decision recorded in the issue, not in
this chunk (thresholds stay user-owned config).
