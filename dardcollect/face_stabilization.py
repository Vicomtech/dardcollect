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
)
from dardcollect.pipeline_utils import make_tqdm

if TYPE_CHECKING:
    from dardcollect.config import FaceCropConfig

logger = logging.getLogger(__name__)


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
    not distort the window. Returns ``None`` for every frame when fewer than
    *min_frames* valid corners exist (caller falls back to per-frame corners),
    matching the previous median-based gate.

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

    Pure over the corner series (no pixels): smooth each track's corner
    trajectory and compute the residual deviation from its median for
    observability. This is pass 1 of the stabilization render.
    """
    plan: dict[int, TrackStabilization] = {}
    for tid, corners in track_corners.items():
        per_frame = smooth_track_corners(
            corners,
            fps,
            face_config.stabilization_window_seconds,
            face_config.stabilization_min_frames,
        )
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
