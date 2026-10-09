# Design — Landmark-stable OFIQ face crops

**Status:** implemented and evaluated on four RAVDESSfake clips plus a 150-track sample (2026-10-08); full-dataset reprocessing is out of scope. **Modality/stage:** video / existing `face_crop_extraction` stage.

## Problem

The exact per-frame OFIQ crop uses noisy pose landmarks and visibly trembles. Existing stabilization smooths/limits/locks the already-derived crop quad, so the rendered quad can differ from the original landmark-derived OFIQ quad. This creates two geometric records: raw source alignment and effective render alignment. The previous face-crop sidecar passed the render quad to pixels and keypoints but did not record it per output frame; raw source quads therefore remained a separate alignment record. The desired contract is one auditable OFIQ transform per output frame, stable facial anchors, and no independent transform for annotations.

## Design

1. Estimate each eye center from its six CIGPose/dlib-68 eye-contour landmarks (indices 59–64 and 65–70 in COCO-133 order), confidence-weighted. Fit frame angle and source/output scale from the two eye centers and canonical OFIQ interocular distance (113 px). Smooth angle/log-scale with a zero-phase two-pass Savitzky-Golay filter (0.8 s).
2. Smooth the eye-midpoint path by minimizing second-difference energy subject to a per-frame anchor-displacement bound in OFIQ output pixels (`stabilization_anchor_tolerance_px`, default 2.5). This chooses the smoothest trajectory that keeps detected eye centers within the alignment budget; it is not a median lock or free-moving band. Build the OFIQ quad from that path so its filtered midpoint maps to (307.5,272) and eye line/distance follow the smoothed fit. Solver failure is loud. Keep the canonical OFIQ destination and crop size.
3. Use that exact final quad for both source-pixel warping and keypoint/bbox transformation. Record the final source-space render quad and source frame index in each face-crop frame sidecar entry. In keep-all mode, a repeated gap image repeats its source index, quad, and annotations. Keep original detector landmarks and source clip provenance intact.
4. Validate representative calm, medium-motion and high-motion tracks across Actors 05, 09, 23 and 24. Compare 0.65/0.8 s windows and report eye-anchor error plus frame-to-frame jitter. Do not reprocess the RAVDESSfake collection without a separate request.

If the prototype cannot meet both anchor stability and noise targets without an unapproved runtime fallback, stop and report the blocking evidence rather than adding a silent fallback.

## FAIR, resumability, and scope

No CSV or UUID/provenance-chain changes. The face-crop JSON sidecar gains per-frame source frame index + render quad and the per-track anchor tolerance (schema + annotation docs update); the exact rendered transform becomes explicit provenance. Existing `.done`/skip behavior is unchanged; changing the renderer requires deleting affected crop MP4/JSON outputs before re-running that stage. Images remain on their existing single-frame OFIQ path.

## Verification

- CPU synthetic test: a moving face with injected landmark noise must keep canonical anchor positions stable after applying the estimated transform, while filtered transform jitter is lower than raw.
- Identity regression tests: rendered pixels and transformed annotations use the same per-frame quad; keep-all gaps repeat matching pixel/annotation records; the schema rejects missing source index or render quad.
- Initial eye-anchored render on the real 3-second example: output-warp acceleration 11.16→1.28 px/frame; scale-step std 0.696%→0.277%; rotation-step std 0.619°→0.218°; raw measured eye midpoint exactly anchored.
- Dense-eye product render on Actor_09: warp acceleration raw 11.16→0.392 px/frame; scale-step std 0.696→0.081%; rotation-step std 0.619→0.139°. Dense eye-center residual p95 2.33/max 2.71 px; eye-line vertical residual p95 0.87 px; interocular-distance error p95 1.16 px. A 120-clip/7,538-frame RAVDESS sample had all six contour landmarks per eye above 0.2 confidence.
- Multi-clip constrained-smoother prototype (150 tracks / 17,271 valid frames): with a 2.5 px output anchor bound, median/p90 warp acceleration .267/.996 px/frame and median/p90 per-track eye p95 error 2.38/2.46 px, max 2.50 px. Fixed 0.65 s has .279/.697 acceleration but p90 per-track eye error p95 6.57 px (max 13.21). Product-code multi-video renders confirm the 0.8 s pose window: Actor05 warp accel .066→.045 (eye p95 2.18→2.18 px), Actor09 .340→.282 (2.38→2.38), Actor23 .346→.285 (2.62→2.62), Actor24 9.99→9.08 (2.71→2.71). The bounded trajectory uses the weighted-eye anchor; reported visual checks use unweighted dense-eye centers.
- CPU gates: Ruff check/format, ty, 278 pytest, quality gates, import-linter and validate_harness passed. Fresh objective gate passed (pipeline exit 0; 0 schema-invalid / 0 hard-fail). Linux tested; Windows not exercised. No full-dataset reprocess.
