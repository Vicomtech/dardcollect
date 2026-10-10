# OFIQ Face-Crop Alignment and Stabilization System Card

**System type:** deterministic landmark geometry and temporal signal processing; no learned model in this component. **Implementation:** `dardcollect/face_geometry.py`, `dardcollect/face_stabilization.py`.

## Purpose

Produce 616×616 OFIQ-aligned video face crops with the eye midpoint at the canonical OFIQ location while suppressing frame-to-frame scale and rotation jitter. CIGPose supplies per-frame COCO-133 landmarks, including six face-contour points for each eye; the crop uses confidence-weighted eye centers instead of the two coarse whole-body eye points.

## Processing

For each track, confidence-weighted centers of the two six-point eye contours define the eye line, interocular scale, and midpoint. A contour landmark counts as evidence at or above `stabilization_eye_min_confidence` (0.2 by default), and each eye needs at least `stabilization_eye_min_landmarks` (3) of them. Angle/log-scale are smoothed with a zero-phase two-pass Savitzky-Golay filter. Translation uses the minimum-curvature eye-midpoint path constrained to the configured `stabilization_anchor_tolerance_px` (2.5 output px by default), then maps that path to the canonical OFIQ midpoint. The resulting final source-space quad is the single transform used for the frame pixels and all sidecar keypoints/bboxes; it is stored as `frame_data[].render_quad_source` with its input `source_frame_index`. Keep-all gap frames repeat the preceding pixels and the associated source index, quad, and annotations.

## Intended use and limitations

The system stabilizes the crop geometry and bounds the eye-midpoint deviation from the canonical grid; it does not freeze expressions or guarantee that every non-rigid facial feature (mouth, nose, hair) is stationary. Landmark errors remain observable in the transformed annotations. A frame whose eye contours do not clear the evidence thresholds (`stabilization_eye_min_confidence`, `stabilization_eye_min_landmarks`) counts as no observation: it does not contribute to the anchor fit, and the count is logged once per track. No path in this system discards a video. A solver that cannot honour the anchor budget is likewise non-fatal: the budget is relaxed (1×→16×) and, failing that, the raw detected eye path is used, always logged.

## Verification

CPU tests inject eye-position and pose noise into a moving synthetic face, verify reduced crop acceleration and bounded canonical eye-center error, and assert that pixel and annotation geometry share the same serialized quad. Real-video validation is reported in the session record and design note; the full RAVDESSfake dataset is not implicitly reprocessed when the algorithm changes.

See `docs/DESIGN_landmark_stabilization.md` for the design, metrics, and verification plan.
