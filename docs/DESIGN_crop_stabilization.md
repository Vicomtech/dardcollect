# Design Doc — Corner-Only Face-Crop Stabilization (issue #9)

**Status:** approved (queue execution 2026-09-09; opt-in, default OFF)
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

- New helper `_compute_track_mean_corners()` in `dardcollect/face_geometry.py`:
  per track, average the per-frame OFIQ corners (median for robustness) across
  all frames where corner computation succeeds; falls back to per-frame corners
  when fewer than `stabilization_min_frames` stable corners exist.
- Applied in `face_crops.py` `process_video()` when
  `face_config.stabilize_face_crops` is true: each output frame's warp uses the
  track-median corners instead of the per-frame corners. `frame_data`
  keypoints/corners in the sidecar keep describing the raw per-frame alignment
  (documented invariant — the crop is stabilized; the sidecar stays honest about
  what was measured).
- Images (`process_image`) are single-frame — unaffected.

## 3. FAIR impact

None: no new CSVs, no new sidecar fields (corners recorded in the sidecar are
the raw per-frame values; stabilization is a rendering-time parameter).
Provenance unchanged; the config key itself is the provenance of the rendering
choice.

## 4. Semantics decision (the #9 design question)

Sidecar `face_crop_corners_ofiq` stays **raw per-frame** (what was measured).
The stabilization is a rendering-time aggregation of already-stored corners —
no sidecar schema change, no tracker change, no golden surface change when OFF.

## 5. Resumability

Unchanged: same `.done` sentinels, same per-track skip. Toggling the flag
requires deleting the affected crops' `.done` sentinels to re-render (documented
in docs/0-GETTING-STARTED.md).

## 6. Config (opt-in, default OFF = zero behavior change)

```yaml
face_crop_extraction:
  stabilization: false                    # corner-only, per-track median
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