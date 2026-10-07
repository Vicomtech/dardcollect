"""Corner-trajectory stabilization for face crops (issue #9).

Split out of ``dardcollect/face_geometry.py`` (2026-10-05) when that module
crossed the 600-line god-file cap. Holds the per-track smoothing plan and the
two-pass render: pass 1 smooths the corner trajectory over the clip sidecar JSON
(no pixels held); pass 2 re-decodes once and renders each frame through its own
smoothed OFIQ quad (O(1) source-frame memory).

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
    OFIQ_SIZE,
    _bbox_iou,
    _corners_to_warp,
    _get_or_compute_corners,
    _output_warp_matrix,
)
from dardcollect.pipeline_utils import make_tqdm

if TYPE_CHECKING:
    from dardcollect.config import FaceCropConfig

logger = logging.getLogger(__name__)

# Two-pass Savitzky-Golay cascade: the same low-pass is applied twice so its
# magnitude response is squared — a sharper roll-off than a single pass of the
# same window. Measured on 519 RAVDESS tracks (2026-10-06): at the shipped 0.4 s
# window the cascade cuts the residual frame-to-frame wobble ~3.2x (accel
# 1.49 -> 0.46 px/frame) with the SAME eye-tracking error vs raw (max 4.0 ->
# 4.1 px); a single pass would need a ~1.2 s window to reach that jitter level,
# paying >2x the tracking error. Number of passes is fixed, not a config knob
# (the window stays the only tuning surface).
_SAVGOL_PASSES = 2


def smooth_track_corners(
    corners_per_frame: list[np.ndarray | None],
    fps: float,
    window_seconds: float,
    min_frames: int = 5,
) -> list[np.ndarray | None]:
    """Stabilization (issue #9): smooth the per-frame corner trajectory.

    Takes the per-frame corner arrays ([TL, TR, BR, BL] in source-frame pixel
    coordinates; ``None`` where corner computation failed) of ONE track and
    returns, for every frame, the corners to render that frame with:

    - Frames with valid corners get the Savitzky-Golay low-pass filtered
      trajectory value — the face stays centred on its real (slowly moving)
      eye landmarks while the residual jitter that survives keypoint smoothing
      is removed. This is what the crop must do: follow the face, not wobble.
    - Gap frames (``None``) are interpolated for the filter input; leading and
      trailing gaps stay ``None`` in the output (no crop that frame).

    The filter runs over the gap-interpolated trajectory so missing frames do
    not distort the window. It is applied twice (see ``_SAVGOL_PASSES``): the
    cascade's sharper roll-off removes more frame-to-frame jitter at the same
    window than one pass, without the tracking error a longer single window
    would cost. Returns ``None`` for every frame when fewer than *min_frames*
    valid corners exist (caller falls back to per-frame corners), matching the
    previous median-based gate.

    Args:
        corners_per_frame: Per-frame corner arrays or None.
        fps: Clip frame rate (window length = ``window_seconds * fps``).
        window_seconds: Savitzky-Golay window in seconds (smoothing strength).
        min_frames: Minimum frames with valid corners to engage stabilization.

    Returns:
        Per-frame (4, 2) float32 corners (None where unavailable), or all-None
        if under *min_frames* valid corners.
    """
    from scipy.signal import savgol_filter

    n = len(corners_per_frame)
    valid_idx = [i for i, c in enumerate(corners_per_frame) if c is not None]
    if len(valid_idx) < min_frames:
        return [None] * n

    filled = _interpolate_corners(corners_per_frame)
    arr = np.stack(filled)  # (N, 4, 2)
    win = _savgol_window(n, fps, window_seconds)
    if win is None:
        # Too short to filter: already-interpolated corners are the best value.
        return [c.astype(np.float32) for c in arr]
    for corner in range(arr.shape[1]):
        for axis in range(2):
            for _ in range(_SAVGOL_PASSES):
                arr[:, corner, axis] = savgol_filter(arr[:, corner, axis], win, 2)
    return [
        arr[i].astype(np.float32) if corners_per_frame[i] is not None else None for i in range(n)
    ]


def _interpolate_corners(
    corners_per_frame: list[np.ndarray | None],
) -> list[np.ndarray]:
    """Linearly interpolate ``None`` gaps in a per-frame corner series.

    Leading/trailing gaps clamp to the first/last valid corner. Requires at
    least one valid corner (callers guarantee this before invoking).
    """
    valid_idx: list[int] = []
    valid: list[np.ndarray] = []
    for i, c in enumerate(corners_per_frame):
        if c is not None:
            valid_idx.append(i)
            valid.append(c)
    first, last = valid_idx[0], valid_idx[-1]
    stacked = np.stack(valid)  # (M, 4, 2)
    filled: list[np.ndarray] = []
    for i in range(len(corners_per_frame)):
        c = corners_per_frame[i]
        if c is not None:
            filled.append(c)
        elif i < first:
            filled.append(stacked[0])
        elif i > last:
            filled.append(stacked[-1])
        else:
            j = np.searchsorted(valid_idx, i)
            lo, hi = valid_idx[j - 1], valid_idx[j]
            t = (i - lo) / (hi - lo)
            lo_c, hi_c = corners_per_frame[lo], corners_per_frame[hi]
            assert lo_c is not None and hi_c is not None
            filled.append((1.0 - t) * lo_c + t * hi_c)
    return filled


def _savgol_window(n_frames: int, fps: float, window_seconds: float) -> int | None:
    """Odd Savitzky-Golay window that fits *n_frames*, or None if too short."""
    win = int(window_seconds * fps) | 1  # bitwise OR 1 -> odd
    win = min(win, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return win if win >= 3 else None


@dataclass
class TrackStabilization:
    """Per-track stabilization result consumed by the render + sidecar paths."""

    per_frame: list[np.ndarray | None]
    median: np.ndarray | None
    max_deviation_px: float
    mean_deviation_px: float
    band_tolerance_px: float = 0.0


def rate_limit_corners(
    corners_per_frame: list[np.ndarray | None],
    factor: float,
    min_frames: int = 5,
) -> list[np.ndarray | None]:
    """Clip per-track corner steps to ``factor`` x the track's median step.

    Measured on RAVDESSfake (2026-10-06): the extreme crop-wobble cases
    (cascade accel > 2 px/frame, 5.8 % of tracks) are driven by detection jumps
    — their largest raw corner step is ~76 px vs ~33 px on normal tracks — not
    by continuous jitter, which is why neither a tolerance band nor a median
    pre-filter fixes them. Limiting each frame's step to ``factor`` x the
    track's median step (factor=5) cuts the extreme wobble ~3x while leaving
    normal tracks essentially unchanged. ``factor <= 0`` disables; ``None``
    frames are preserved.
    """
    if not factor or factor <= 0:
        return corners_per_frame
    n = len(corners_per_frame)
    valid_idx = [i for i, c in enumerate(corners_per_frame) if c is not None]
    if len(valid_idx) < min_frames:
        return corners_per_frame
    arr = np.stack(_interpolate_corners(corners_per_frame))  # (N, 4, 2)
    out = arr.copy()
    for corner in range(arr.shape[1]):
        for axis in range(2):
            x = arr[:, corner, axis]
            limit = float(np.median(np.abs(np.diff(x))) * factor)
            if limit <= 0:
                continue
            for i in range(1, n):
                step = out[i, corner, axis] - out[i - 1, corner, axis]
                if abs(step) > limit:
                    out[i, corner, axis] = out[i - 1, corner, axis] + np.sign(step) * limit
    return [
        out[i].astype(np.float32) if corners_per_frame[i] is not None else None for i in range(n)
    ]


def _quad_to_params(q: np.ndarray) -> np.ndarray:
    """(tx, ty, theta, scale) of a similarity quad [TL,TR,BR,BL] (source px)."""
    v1 = q[1] - q[0]
    v2 = q[2] - q[1]
    scale = (np.linalg.norm(v1) + np.linalg.norm(v2)) / (2.0 * OFIQ_SIZE)
    theta = math.atan2(v1[1], v1[0])
    return np.array([q[0][0], q[0][1], theta, scale])


def _params_to_quad(p: np.ndarray) -> np.ndarray:
    """Rebuild the similarity quad from (tx, ty, theta, scale)."""
    tx, ty, theta, scale = p
    d = scale * OFIQ_SIZE
    c, s = math.cos(theta), math.sin(theta)
    tl = np.array([tx, ty])
    tr = tl + d * np.array([c, s])
    br = tr + d * np.array([-s, c])
    bl = tl + d * np.array([-s, c])
    return np.array([tl, tr, br, bl], dtype=np.float64)


def _min_curvature_in_box(P: np.ndarray, tolerance: float) -> np.ndarray:
    """Smoothest 2D curve within ``tolerance`` of P (min second-difference).

    Solves ``min ||D2 r||^2 s.t. |r-P| <= tolerance`` per axis (bounds); the
    bound-constrained least squares is the correct way to 'freeze inside the
    tolerance, recenter smoothly outside it' — a switch/hysteresis controller
    instead injects velocity discontinuities that make the wobble worse.
    """
    from scipy.optimize import lsq_linear

    n = len(P)
    if n < 4 or tolerance <= 0:
        return P.copy()
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1.0, -2.0, 1.0
    r = np.empty_like(P)
    for axis in range(2):
        r[:, axis] = lsq_linear(
            D, np.zeros(n - 2), bounds=(P[:, axis] - tolerance, P[:, axis] + tolerance)
        ).x
    return r


def _crop_wobble_px(quads: list[np.ndarray | None]) -> float:
    """High-frequency energy (accel RMS, output px) of the crop trajectory."""
    valid = [q for q in quads if q is not None]
    if len(valid) < 3:
        return 0.0
    trans = np.array([_output_warp_matrix(q)[:2, 2] for q in valid])
    accel = trans[2:] - 2 * trans[1:-1] + trans[:-2]
    return float(np.sqrt(np.mean(np.sum(accel * accel, axis=1))))


def band_crop_trajectory(
    corners_per_frame: list[np.ndarray | None],
    tolerance_px: float,
    activate_px: float,
    min_frames: int = 5,
) -> list[np.ndarray | None]:
    """Min-curvature tolerance band on the crop translation (eye tolerance).

    The OFIQ crop may sit up to *tolerance_px* (source px) off the smoothed
    trajectory — the user-accepted budget on eye placement — which lets the
    window stay still inside the band and recenter smoothly when the face
    leaves it. Applied only when the cascade trajectory still wobbles more than
    *activate_px* (output px/frame): the calibration on RAVDESSfake showed the
    band is unnecessary (and costs alignment) on already-stable tracks.
    Rotation/scale stay from the cascade so the quad remains a similarity.
    ``tolerance_px <= 0`` disables.
    """
    if not tolerance_px or tolerance_px <= 0:
        return corners_per_frame
    idx = [i for i, c in enumerate(corners_per_frame) if c is not None]
    if len(idx) < min_frames:
        return corners_per_frame
    if activate_px and _crop_wobble_px(corners_per_frame) < activate_px:
        return corners_per_frame
    banded_frames = [c for c in corners_per_frame if c is not None]
    tl = np.array([q[0] for q in banded_frames], dtype=np.float64)
    banded = _min_curvature_in_box(tl, tolerance_px)
    out = list(corners_per_frame)
    for k, i in enumerate(idx):
        p = _quad_to_params(banded_frames[k])
        p[0], p[1] = banded[k, 0], banded[k, 1]
        out[i] = _params_to_quad(p)
    return out


def compute_track_mean_corners(
    corners_per_frame: list[np.ndarray | None],
    min_frames: int = 5,
) -> np.ndarray | None:
    """Component-wise median quad of a track's valid corners, or None.

    Retained as the stabilization reference: the smoothed trajectory's central
    position. A median is robust against the outlier frames where landmark
    estimation briefly degrades. Returns None when fewer than *min_frames*
    valid corners exist.
    """
    valid = [c for c in corners_per_frame if c is not None]
    if len(valid) < min_frames:
        return None
    stacked = np.stack(valid)  # (N, 4, 2)
    return np.median(stacked, axis=0).astype(np.float32)


def plan_track_stabilization(
    track_corners: dict[int, list[np.ndarray | None]],
    face_config: FaceCropConfig,
    fps: float,
) -> dict[int, TrackStabilization]:
    """Plan per-track stabilization from the per-frame corner series.

    Pure over the corner series (no pixels): rate-limit detection jumps, then
    smooth each track's corner trajectory (Savitzky-Golay), then optionally
    band the crop translation (tolerance on eye placement), and compute the
    residual deviation from the median for observability. Pass 1 of the render.
    """
    plan: dict[int, TrackStabilization] = {}
    for tid, corners in track_corners.items():
        limited = rate_limit_corners(corners, face_config.stabilization_max_step_median_factor)
        per_frame = smooth_track_corners(
            limited,
            fps,
            face_config.stabilization_window_seconds,
            face_config.stabilization_min_frames,
        )
        band_px = 0.0
        if face_config.stabilization_band_tolerance_px > 0:
            banded = band_crop_trajectory(
                per_frame,
                face_config.stabilization_band_tolerance_px,
                face_config.stabilization_band_activate_px,
                face_config.stabilization_min_frames,
            )
            if banded is not per_frame:
                per_frame = banded
                band_px = face_config.stabilization_band_tolerance_px
        median = compute_track_mean_corners(corners, face_config.stabilization_min_frames)
        stable_n = len([c for c in corners if c is not None])
        if median is None:
            logger.info(
                "  Track %d: stabilization requested but < %d stable corners — per-frame fallback",
                tid,
                face_config.stabilization_min_frames,
            )
            max_dev = mean_dev = 0.0
        else:
            per_corner_max = [
                float(np.linalg.norm(c - median, axis=1).max()) for c in corners if c is not None
            ]
            max_dev = max(per_corner_max)
            mean_dev = float(np.mean(per_corner_max))
            logger.info(
                "  Track %d: smoothed stabilization engaged (%d stable frames, "
                "residual vs median max %.1f px / mean %.1f px)",
                tid,
                stable_n,
                max_dev,
                mean_dev,
            )
        plan[tid] = TrackStabilization(
            per_frame=per_frame,
            median=median,
            max_deviation_px=max_dev,
            mean_deviation_px=mean_dev,
            band_tolerance_px=band_px,
        )
    return plan


def _valid_crop_corners(
    corners: np.ndarray | None,
    det: dict,
    frame_bboxes: list[tuple[int, list]],
    face_config: FaceCropConfig,
) -> np.ndarray | None:
    """Per-detection OFIQ crop decision: the corners, or None when the frame
    must not contribute a crop for that track (corners unavailable, or the
    bbox overlaps another detection beyond ``max_overlap_iou``).

    Shared by the per-frame rendering path (face_crops.py) and the
    stabilization plan so both apply the exact same inclusion rule.
    """
    if corners is None:
        return None
    if any(
        _bbox_iou(det["bbox"], ob) > face_config.max_overlap_iou
        for oid, ob in frame_bboxes
        if oid != det["track_id"]
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
    """Stabilization pass 1 — corner-only plan over the clip sidecar JSON.

    No decode, no pixels held: one pass over the per-frame detections records,
    per track, the per-frame OFIQ corners, then smooths each track's corner
    trajectory so pass 2 can render every frame through its own smoothed quad.
    The previous single-pass design retained every full-resolution source frame
    in memory (~11 GB for a 60 s 1080p clip); this plan holds only (4, 2)
    corner arrays and the smoothed per-frame series.

    Returns:
        (frame_track_plan, stabilizations) — one ``{track_id: corners | None}``
        dict per frame (index = relative frame id; None = no usable crop that
        frame) and the per-track :class:`TrackStabilization` (per_frame None
        series = track falls back to per-frame corners).
    """
    # tid → {relative_frame_id: corners}; gaps (track absent that frame) are
    # filled with None when the per-frame series is built, so every track's
    # series is indexed by frame id and stays aligned with the rendered frames.
    track_corners: dict[int, dict[int, np.ndarray | None]] = {}
    frame_track_plan: list[dict[int, np.ndarray | None]] = []
    for frame_id in range(total_frames):
        detections = frame_data_orig.get(str(start_frame + frame_id), [])
        frame_bboxes = [(d["track_id"], d["bbox"]) for d in detections]
        plan: dict[int, np.ndarray | None] = {}
        for det in detections:
            tid = det["track_id"]
            corners = _get_or_compute_corners(det, face_config)
            track_corners.setdefault(tid, {})[frame_id] = corners
            plan[tid] = _valid_crop_corners(corners, det, frame_bboxes, face_config)
        frame_track_plan.append(plan)

    series_by_track = {
        tid: [by_frame.get(fid) for fid in range(total_frames)]
        for tid, by_frame in track_corners.items()
    }
    return frame_track_plan, plan_track_stabilization(series_by_track, face_config, fps)


def render_stabilized_track_frames(
    video_path: Path,
    frame_track_plan: list[dict[int, np.ndarray | None]],
    stabilizations: dict[int, TrackStabilization],
    total_frames: int,
) -> dict[int, list[tuple[int, np.ndarray | None]]]:
    """Stabilization pass 2 — re-decode and render each detection through its
    smoothed per-frame OFIQ quad (per-frame corners when the track fell back).

    Holds only the current source frame (O(1) memory) and returns the same
    structure the per-frame path produces:
    ``track_id -> [(relative_frame_idx, ofiq_crop_or_None), ...]``.
    """
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
            plan = frame_track_plan[frame_id] if frame_id < len(frame_track_plan) else {}
            for tid, per_frame_corners in plan.items():
                if per_frame_corners is None:
                    track_frames.setdefault(tid, []).append((frame_id, None))
                    continue
                stab = stabilizations.get(tid)
                smoothed = _smoothed_quad_for_frame(stab, frame_id, per_frame_corners)
                track_frames.setdefault(tid, []).append(
                    (frame_id, _corners_to_warp(frame, smoothed, OFIQ_SIZE))
                )
            frame_id += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
    return track_frames


def _smoothed_quad_for_frame(
    stab: TrackStabilization | None,
    frame_id: int,
    per_frame_corners: np.ndarray,
) -> np.ndarray:
    """The smoothed quad to render *frame_id* with (per-frame quad as fallback)."""
    if stab is not None and frame_id < len(stab.per_frame):
        smoothed = stab.per_frame[frame_id]
        if smoothed is not None:
            return smoothed
    return per_frame_corners
