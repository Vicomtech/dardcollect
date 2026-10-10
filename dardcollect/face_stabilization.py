"""Eye-anchored OFIQ crop stabilization for video face tracks.

Pass 1 plans a final source-space OFIQ quad per frame from sidecar landmarks,
without retaining pixels. Pass 2 re-decodes once and uses that exact quad to
render each crop, keeping memory bounded to the current source frame.

The library never imports pipeline stage scripts; this is the allowed direction
(pipeline -> library).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dardcollect.face_geometry import (
    _ALIGN_OFIQ_DST,
    OFIQ_SIZE,
    _bbox_iou,
    _corners_to_warp,
    _get_or_compute_corners,
)
from dardcollect.pipeline_utils import make_tqdm

if TYPE_CHECKING:
    from dardcollect.config import FaceCropConfig

logger = logging.getLogger(__name__)
_SAVGOL_PASSES = 2
_MIN_CURVATURE_DIAGONAL = (1.0, -2.0, 1.0)

# Anchor budgets tried in order before falling back to the raw path. The first is
# the configured budget; the wider ones engage only when the solver cannot honour
# it, so no track is ever dropped (user decision 2026-10-10) and the relaxation is
# logged instead of being silent.
_ANCHOR_BUDGET_RELAXATIONS = (1.0, 2.0, 4.0, 8.0, 16.0)
_SOLVER_MAX_ITER = 10000
_EYE_LANDMARK_GROUPS = (slice(59, 65), slice(65, 71))
_EYE_LANDMARK_GROUPS = (slice(59, 65), slice(65, 71))
_OFIQ_EYE_DISTANCE = float(np.linalg.norm(_ALIGN_OFIQ_DST[1] - _ALIGN_OFIQ_DST[0]))


@dataclass(frozen=True)
class EyeAnchorSmoothing:
    """Per-track smoothing inputs used to construct final OFIQ render quads."""

    fps: float
    window_seconds: float
    anchor_tolerance_px: float
    min_frames: int


@dataclass
class TrackStabilization:
    """Per-track render quads and anchor bound consumed by writer paths."""

    per_frame: list[np.ndarray | None]
    anchor_tolerance_px: float | None


def _interpolate_series(values: list[np.ndarray | None]) -> list[np.ndarray]:
    """Linearly fill missing vectors/quads; clamp leading/trailing gaps."""
    valid_idx = [i for i, value in enumerate(values) if value is not None]
    valid_values = [value for value in values if value is not None]
    stacked = np.stack(valid_values)
    first, last = valid_idx[0], valid_idx[-1]
    filled: list[np.ndarray] = []
    for i, value in enumerate(values):
        if value is not None:
            filled.append(value)
        elif i < first:
            filled.append(stacked[0])
        elif i > last:
            filled.append(stacked[-1])
        else:
            j = np.searchsorted(valid_idx, i)
            lo, hi = valid_idx[j - 1], valid_idx[j]
            t = (i - lo) / (hi - lo)
            lo_value, hi_value = values[lo], values[hi]
            assert lo_value is not None and hi_value is not None
            filled.append((1.0 - t) * lo_value + t * hi_value)
    return filled


def _savgol_window(n_frames: int, fps: float, window_seconds: float) -> int | None:
    """Odd Savitzky-Golay window that fits *n_frames*, or None if too short."""
    win = int(window_seconds * fps) | 1
    win = min(win, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return win if win >= 3 else None


def _pose_to_quad(
    eye_midpoint: np.ndarray, angle: float, scale: float, canonical_eye_midpoint: np.ndarray
) -> np.ndarray:
    """Build an OFIQ square mapping the denoised eye anchor to canonical center."""
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-math.sin(angle), math.cos(angle)])
    top_left = eye_midpoint - scale * (
        canonical_eye_midpoint[0] * u + canonical_eye_midpoint[1] * v
    )
    side = scale * OFIQ_SIZE
    top_right = top_left + side * u
    bottom_right = top_right + side * v
    bottom_left = top_left + side * v
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _smooth_eye_midpoints_within_bound(
    midpoints: np.ndarray, scales: np.ndarray, tolerance_px: float
) -> np.ndarray:
    """Minimize eye-path curvature while bounding output-space anchor error."""
    if tolerance_px <= 0:
        return midpoints.copy()
    from scipy.optimize import lsq_linear
    from scipy.sparse import diags

    n = len(midpoints)
    second_difference = diags(
        [
            np.full(n - 2, _MIN_CURVATURE_DIAGONAL[0]),
            np.full(n - 2, _MIN_CURVATURE_DIAGONAL[1]),
            np.full(n - 2, _MIN_CURVATURE_DIAGONAL[2]),
        ],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )
    source_tolerance = tolerance_px * scales / math.sqrt(2.0)
    smoothed = np.empty_like(midpoints)
    for axis in range(2):
        series = midpoints[:, axis]
        solved = None
        for factor in _ANCHOR_BUDGET_RELAXATIONS:
            result = lsq_linear(
                second_difference,
                np.zeros(n - 2),
                bounds=(series - source_tolerance * factor, series + source_tolerance * factor),
                tol=1e-6,
                max_iter=_SOLVER_MAX_ITER,
            )
            if result.success:
                solved = result.x
                if factor != 1.0:
                    logger.warning(
                        "Eye-anchor smoothing: solver could not honour the %.1f px budget "
                        "(%s); relaxed it to %.1f px — the crop is still rendered",
                        tolerance_px,
                        result.message,
                        tolerance_px * factor,
                    )
                break
        if solved is None:
            # Never drop a track: fall back to the detected path (exact anchor, no
            # smoothing) so this video still produces a crop. Logged, never silent.
            logger.warning(
                "Eye-anchor smoothing: solver failed at every budget up to %.1f px; "
                "using the raw detected eye path — the crop is still rendered",
                tolerance_px * _ANCHOR_BUDGET_RELAXATIONS[-1],
            )
            solved = series
        smoothed[:, axis] = solved
    return smoothed


def smooth_quad_pose_anchored_to_eyes(
    corners_per_frame: list[np.ndarray | None],
    eye_centers_per_frame: list[np.ndarray | None],
    smoothing: EyeAnchorSmoothing,
) -> list[np.ndarray | None]:
    """Fit/smooth OFIQ pose from dense eye contours and re-anchor per frame.

    Each observation contains the two confidence-weighted eye-contour centers.
    Their vector determines crop angle and scale relative to the OFIQ canonical
    interocular distance. Pose uses two zero-phase filter passes; translation
    minimizes trajectory curvature subject to the configured output-pixel error
    bound. The single resulting quad is shared by pixels and annotations.
    """
    if len(corners_per_frame) != len(eye_centers_per_frame):
        raise ValueError("Corner and eye-center series must have equal length")
    fps = smoothing.fps
    window_seconds = smoothing.window_seconds
    anchor_tolerance_px = smoothing.anchor_tolerance_px
    min_frames = smoothing.min_frames
    valid = [
        i
        for i, (quad, centers) in enumerate(
            zip(corners_per_frame, eye_centers_per_frame, strict=True)
        )
        if quad is not None and centers is not None
    ]
    if len(valid) < min_frames:
        return [None] * len(corners_per_frame)

    filled_centers = _interpolate_series(eye_centers_per_frame)
    pairs = np.stack(filled_centers)
    eye_vectors = pairs[:, 1] - pairs[:, 0]
    angles = np.unwrap(np.arctan2(eye_vectors[:, 1], eye_vectors[:, 0]))
    log_scales = np.log(np.linalg.norm(eye_vectors, axis=1) / _OFIQ_EYE_DISTANCE)
    eye_midpoints = pairs.mean(axis=1)
    window = _savgol_window(len(pairs), fps, window_seconds)
    if window is not None:
        from scipy.signal import savgol_filter

        for _ in range(_SAVGOL_PASSES):
            angles = savgol_filter(angles, window, 2)
            log_scales = savgol_filter(log_scales, window, 2)
    scales = np.exp(log_scales)
    eye_midpoints = _smooth_eye_midpoints_within_bound(eye_midpoints, scales, anchor_tolerance_px)
    canonical_eye_midpoint = np.mean(_ALIGN_OFIQ_DST[:2], axis=0)
    output: list[np.ndarray | None] = [None] * len(corners_per_frame)
    for i in valid:
        output[i] = _pose_to_quad(
            eye_midpoints[i], float(angles[i]), float(scales[i]), canonical_eye_midpoint
        )
    return output


def _eye_centers_from_detection(
    det: dict, frame_id: int, tid: int, min_confidence: float, min_landmarks: int
) -> np.ndarray | None:
    """Confidence-weighted centers of both six-point CIGPose eye contours.

    Returns None when this frame carries no usable eye evidence (missing contour
    landmarks, or either eye short of `min_landmarks` at `min_confidence`). The
    frame then contributes nothing to the anchor fit; it never aborts the video
    (user decision 2026-10-10: no track is dropped for want of evidence).
    """
    points = np.asarray(det.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(det.get("keypoint_scores", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 71:
        logger.debug("Track %d frame %d: no eye-contour landmarks", tid, frame_id)
        return None
    if scores.ndim != 1 or len(scores) < 71:
        logger.debug("Track %d frame %d: no eye-contour confidence scores", tid, frame_id)
        return None
    centers = []
    for group in _EYE_LANDMARK_GROUPS:
        group_scores = scores[group]
        valid = group_scores >= min_confidence
        if int(valid.sum()) < min_landmarks:
            logger.debug(
                "Track %d frame %d: eye has %d contour landmarks at confidence >= %s (needs %d)",
                tid,
                frame_id,
                int(valid.sum()),
                min_confidence,
                min_landmarks,
            )
            return None
        centers.append(np.average(points[group][valid], axis=0, weights=group_scores[valid]))
    return np.asarray(centers, dtype=np.float32)


def plan_track_stabilization(
    track_corners: dict[int, list[np.ndarray | None]],
    track_eye_centers: dict[int, list[np.ndarray | None]],
    face_config: FaceCropConfig,
    fps: float,
) -> dict[int, TrackStabilization]:
    """Plan one final eye-anchored OFIQ render quad per frame and track."""
    plan: dict[int, TrackStabilization] = {}
    smoothing = EyeAnchorSmoothing(
        fps=fps,
        window_seconds=face_config.stabilization_window_seconds,
        anchor_tolerance_px=face_config.stabilization_anchor_tolerance_px,
        min_frames=face_config.stabilization_min_frames,
    )
    for tid, corners in track_corners.items():
        eyes = track_eye_centers[tid]
        per_frame = smooth_quad_pose_anchored_to_eyes(corners, eyes, smoothing)
        usable_n = sum(
            corner is not None and eye is not None
            for corner, eye in zip(corners, eyes, strict=True)
        )
        if usable_n >= face_config.stabilization_min_frames:
            logger.info("  Track %d: eye-anchored stabilization engaged (%d frames)", tid, usable_n)
        else:
            logger.info(
                "  Track %d: fewer than %d eye-anchored quads; retaining per-frame OFIQ alignment",
                tid,
                face_config.stabilization_min_frames,
            )
        plan[tid] = TrackStabilization(per_frame, face_config.stabilization_anchor_tolerance_px)
    return plan


def _valid_crop_corners(
    corners: np.ndarray | None,
    det: dict,
    frame_bboxes: list[tuple[int, list]],
    face_config: FaceCropConfig,
) -> np.ndarray | None:
    """Per-detection crop inclusion rule shared by both render paths."""
    if corners is None:
        return None
    if any(
        _bbox_iou(det["bbox"], other_bbox) > face_config.max_overlap_iou
        for other_id, other_bbox in frame_bboxes
        if other_id != det["track_id"]
    ):
        return None
    return corners


def plan_stabilized_track_crops(
    frame_data_orig: dict,
    start_frame: int,
    face_config: FaceCropConfig,
    total_frames: int,
    fps: float,
) -> tuple[list[dict[int, np.ndarray | None]], dict[int, TrackStabilization]]:
    """Pass 1: build landmark-anchored render quads from sidecar data only."""
    corners_by_track: dict[int, dict[int, np.ndarray | None]] = {}
    eye_centers_by_track: dict[int, dict[int, np.ndarray | None]] = {}
    frame_plan: list[dict[int, np.ndarray | None]] = []
    for frame_id in range(total_frames):
        detections = frame_data_orig.get(str(start_frame + frame_id), [])
        boxes = [(det["track_id"], det["bbox"]) for det in detections]
        frame_plan_item: dict[int, np.ndarray | None] = {}
        for det in detections:
            tid = det["track_id"]
            corners = _get_or_compute_corners(det, face_config)
            corners_by_track.setdefault(tid, {})[frame_id] = corners
            eye_centers_by_track.setdefault(tid, {})[frame_id] = (
                _eye_centers_from_detection(
                    det,
                    frame_id,
                    tid,
                    face_config.stabilization_eye_min_confidence,
                    face_config.stabilization_eye_min_landmarks,
                )
                if corners is not None
                else None
            )
            frame_plan_item[tid] = _valid_crop_corners(corners, det, boxes, face_config)
        frame_plan.append(frame_plan_item)

    corner_series = {
        tid: [values.get(fid) for fid in range(total_frames)]
        for tid, values in corners_by_track.items()
    }
    eye_series = {
        tid: [values.get(fid) for fid in range(total_frames)]
        for tid, values in eye_centers_by_track.items()
    }
    for tid, eyes in eye_series.items():
        with_corners = sum(c is not None for c in corner_series[tid])
        without_eyes = sum(
            c is not None and e is None for c, e in zip(corner_series[tid], eyes, strict=True)
        )
        if without_eyes:
            logger.warning(
                "Track %d: %d of %d croppable frames carry no usable eye evidence; they do "
                "not contribute to the anchor fit (the video is still processed)",
                tid,
                without_eyes,
                with_corners,
            )
    return frame_plan, plan_track_stabilization(corner_series, eye_series, face_config, fps)


def _render_quad_for_frame(
    stabilization: TrackStabilization | None, frame_id: int, raw_quad: np.ndarray
) -> np.ndarray:
    """Return the final render quad for one frame (raw OFIQ when not engaged)."""
    if stabilization is not None and frame_id < len(stabilization.per_frame):
        quad = stabilization.per_frame[frame_id]
        if quad is not None:
            return quad
    return raw_quad


def render_stabilized_track_frames(
    video_path: Path,
    frame_track_plan: list[dict[int, np.ndarray | None]],
    stabilizations: dict[int, TrackStabilization],
    total_frames: int,
) -> dict[int, list[tuple[int, np.ndarray | None]]]:
    """Pass 2: warp each frame through its planned quad using O(1) source memory."""
    track_frames: dict[int, list[tuple[int, np.ndarray | None]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot re-open video for stabilization pass 2: {video_path}")
    pbar = make_tqdm(
        total=total_frames, unit="fr", desc=f"{video_path.name[:32]} (stab)", dynamic_ncols=True
    )
    try:
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_plan = frame_track_plan[frame_id] if frame_id < len(frame_track_plan) else {}
            for tid, raw_quad in frame_plan.items():
                if raw_quad is None:
                    track_frames.setdefault(tid, []).append((frame_id, None))
                    continue
                quad = _render_quad_for_frame(stabilizations.get(tid), frame_id, raw_quad)
                track_frames.setdefault(tid, []).append(
                    (frame_id, _corners_to_warp(frame, quad, OFIQ_SIZE))
                )
            frame_id += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
    return track_frames
