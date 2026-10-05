"""person_clips_helpers.py — helper functions for person clip extraction.

Extracted from person_clips.py to reduce size (person_clips.py: 602 → ~400 lines).
These 8 helpers are cohesive utilities for scene-change detection, frame filtering,
and progressive flush orchestration.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dardcollect.config import ClipExtractionConfig, DetectorConfig
from dardcollect.pipeline_utils import (
    check_face_visibility,
    check_frontal_face,
    scene_changed,
)
from dardcollect.poser import PoseEstimator
from dardcollect.tracker import Segment

logger = logging.getLogger(__name__)


# ── Scene-cut signals 1–2 (luminance histogram, bbox-area ratio) ─────────────
# Live here, next to signal 3 (block_delta_cut), so all three signals share one
# home. pipeline_utils.scene_changed (the public orchestrator) calls them.


def _histogram_cut(prev_frame: np.ndarray, curr_frame: np.ndarray, hist_threshold: float) -> bool:
    """Signal 1: luminance-histogram correlation drop below *hist_threshold*."""
    small_prev = cv2.resize(prev_frame, (128, 72), interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_frame, (128, 72), interpolation=cv2.INTER_AREA)

    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)

    hist_prev = cv2.calcHist([gray_prev], [0], None, [64], [0, 256])
    hist_curr = cv2.calcHist([gray_curr], [0], None, [64], [0, 256])
    cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return bool(float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)) < hist_threshold)


def _bbox_area_cut(
    prev_bboxes: "np.ndarray", curr_bboxes: "np.ndarray", bbox_area_ratio_threshold: float
) -> bool:
    """Signal 2: max-detection-area ratio above threshold (wide shot vs close-up)."""
    if len(prev_bboxes) == 0 or len(curr_bboxes) == 0:
        return False

    def _max_area(bboxes: "np.ndarray") -> float:
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        return float(np.max(widths * heights))

    prev_area = _max_area(prev_bboxes)
    curr_area = _max_area(curr_bboxes)
    if prev_area <= 0 or curr_area <= 0:
        return False
    return max(prev_area / curr_area, curr_area / prev_area) >= bbox_area_ratio_threshold


# ── Scene-cut signal 3: spatial block-delta (issue #4) ────────────────────────

# Grid the downscaled frame is divided into (4×4 = 16 cells).
_BLOCK_DELTA_GRID = 4
# Downscaled side length the frame is reduced to before gridding.
_BLOCK_DELTA_SIZE = 64


def block_delta_cut(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float,
    fraction: float,
) -> bool:
    """Third scene-cut signal: spatial block-histogram delta.

    Splits both downscaled grayscale frames into a 4×4 grid and counts cells
    whose mean luminance differs by more than *threshold*. A cut is declared
    when the changed fraction exceeds *fraction*. Fires for same-set
    shot/reverse-shot cuts that the global luminance histogram (spatially
    invariant) survives, because the spatial layout flips even when the global
    histogram does not.

    Args:
        prev_frame: Previous BGR frame.
        curr_frame: Current BGR frame.
        threshold: Per-cell mean-luminance delta that marks a block "changed".
        fraction: Fraction of changed blocks that declares a cut [0, 1].

    Returns:
        True if the block-delta signal fires.
    """
    small_prev = cv2.resize(
        prev_frame, (_BLOCK_DELTA_SIZE, _BLOCK_DELTA_SIZE), interpolation=cv2.INTER_AREA
    )
    small_curr = cv2.resize(
        curr_frame, (_BLOCK_DELTA_SIZE, _BLOCK_DELTA_SIZE), interpolation=cv2.INTER_AREA
    )
    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)

    cell = _BLOCK_DELTA_SIZE // _BLOCK_DELTA_GRID
    changed = 0
    total = _BLOCK_DELTA_GRID * _BLOCK_DELTA_GRID
    for row in range(_BLOCK_DELTA_GRID):
        for col in range(_BLOCK_DELTA_GRID):
            y0, y1 = row * cell, (row + 1) * cell
            x0, x1 = col * cell, (col + 1) * cell
            prev_mean = float(gray_prev[y0:y1, x0:x1].mean())
            curr_mean = float(gray_curr[y0:y1, x0:x1].mean())
            if abs(prev_mean - curr_mean) > threshold:
                changed += 1
    return (changed / total) >= fraction


def filter_detections(
    det_bboxes: np.ndarray,
    det_scores: np.ndarray,
    width: int,
    height: int,
    clip_config: ClipExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop bboxes covering too large a fraction of the frame (title cards,
    overlays) or with extreme aspect ratio (furniture, animals, not persons)."""
    if len(det_bboxes) == 0:
        return det_bboxes, det_scores
    frame_area = width * height
    box_w = det_bboxes[:, 2] - det_bboxes[:, 0]
    box_h = det_bboxes[:, 3] - det_bboxes[:, 1]
    bbox_areas = box_w * box_h
    aspect_ratios = box_w / np.maximum(box_h, 1.0)
    keep = (bbox_areas / frame_area <= clip_config.max_bbox_area_percent / 100.0) & (
        aspect_ratios <= clip_config.max_detection_aspect_ratio
    )
    return det_bboxes[keep], det_scores[keep]


def load_resume_start(progress_path: Path, total_frames: int, cap) -> int:
    """Return the frame to resume from (0 if no/invalid progress file)."""
    if not progress_path.exists():
        return 0
    try:
        with open(progress_path) as f:
            progress_data = json.load(f)
            last_frame = progress_data.get("last_processed_frame", 0)
            if last_frame > 0 and last_frame < total_frames - 1:
                logger.info(
                    "  RESUMING from frame %d (%.1f%%)",
                    last_frame,
                    (last_frame / total_frames) * 100,
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame + 1)
                return last_frame + 1
    except Exception as e:
        logger.warning("  Failed to load progress file, starting from 0: %s", e)
    return 0


def compute_face_flags(
    tracklets_kpts: list,
    height: int,
    clip_config: ClipExtractionConfig,
    det_config: DetectorConfig,
    poser: PoseEstimator | None,
) -> tuple[bool, bool]:
    """Return (face_visible, mouth_open) by checking every tracklet's face.

    Iterates all tracks (not just the first) so mouth_open captures any speaking
    person in a multi-person frame.
    """
    face_visible = False
    mouth_open = False
    if poser is None or not clip_config.require_face_visibility:
        return face_visible, mouth_open
    for _t, keypoints, kpt_scores in tracklets_kpts:
        if keypoints is None:
            continue
        assert kpt_scores is not None, "kpt_scores should be set when keypoints is set"
        is_visible = check_face_visibility(
            keypoints,
            kpt_scores,
            height,
            clip_config.min_face_size_percent,
            det_config.pose_keypoint_threshold,
        )
        if is_visible and clip_config.require_frontal_face:
            is_visible = check_frontal_face(
                keypoints,
                kpt_scores,
                clip_config.frontal_symmetry_threshold,
                det_config.pose_keypoint_threshold,
            )
        if is_visible:
            face_visible = True
            if clip_config.enable_visual_speaking and poser.check_mouth_open(
                keypoints,
                kpt_scores,
                min_score=det_config.pose_keypoint_threshold,
            ):
                mouth_open = True
    return face_visible, mouth_open


def build_frame_data(tracklets_kpts: list) -> list[dict]:
    """Build the per-tracklet frame_data list for the current frame."""
    current_frame_data: list[dict] = []
    for t, kpts, kpt_scores in tracklets_kpts:
        data_entry = {
            "track_id": t.track_id,
            "bbox": [round(x, 1) for x in t.tlbr.tolist()],
            "score": round(float(t.det_score), 3),
        }
        if kpts is not None:
            assert kpt_scores is not None, "kpt_scores should be set when kpts is set"
            data_entry["keypoints"] = [[round(x, 1), round(y, 1)] for x, y in kpts.tolist()]
            data_entry["keypoint_scores"] = [round(float(s), 3) for s in kpt_scores.tolist()]
        current_frame_data.append(data_entry)
    return current_frame_data


def should_progressive_flush(
    pending_segments: list,
    frame_id: int,
    frames_since_flush: int,
    clip_config: ClipExtractionConfig,
    fps: float,
) -> bool:
    """Progressive-flush predicate: flush once pending segments are stable (gap
    exceeds twice the merge window) or every ~30s of video."""
    if not pending_segments:
        return False
    last_seg_end = pending_segments[-1].end_frame
    gap = frame_id - last_seg_end
    return gap > max(clip_config.merge_gap_frames * 2, 30) or frames_since_flush > 30 * fps


def save_progress(progress_path: Path, frame_id: int, video_path: Path) -> None:
    """Save resumption progress to a JSON file."""
    try:
        with open(progress_path, "w") as f:
            json.dump(
                {
                    "last_processed_frame": frame_id,
                    "timestamp": time.time(),
                    "video": video_path.name,
                },
                f,
            )
    except Exception as e:
        logger.warning("Failed to save progress: %s", e)


@dataclass
class SceneView:
    """One frame's inputs to the scene-change predicate (single argument)."""

    clip_config: ClipExtractionConfig
    prev_frame: np.ndarray | None
    frame_id: int
    last_scene_change_frame: int
    prev_det_bboxes: np.ndarray
    det_bboxes: np.ndarray
    frame: np.ndarray


def is_scene_change(view: SceneView) -> bool:
    """Scene-change predicate (cooldown-gated cut detector: histogram + bbox
    area + opt-in block-delta signal)."""
    cooldown = 8  # frames to suppress re-detection immediately after a cut
    return bool(
        view.clip_config.scene_change_detection
        and view.prev_frame is not None
        and view.frame_id - view.last_scene_change_frame > cooldown
        and scene_changed(
            view.prev_frame,
            view.frame,
            view.prev_det_bboxes,
            view.det_bboxes,
            view.clip_config,
        )
    )


def _split_segment(seg: Segment, max_frames: int) -> list[Segment]:
    """Split an over-long segment into <= *max_frames* sub-segments."""
    parts: list[Segment] = []
    start = seg.start_frame
    while start < seg.end_frame:
        end = min(start + max_frames - 1, seg.end_frame)
        ratio = (end - start + 1) / seg.frame_count
        face_frames = int(seg.face_visible_frames * ratio)
        # Consecutive face frames: recount from the sub-clip's frame_data if
        # available; otherwise conservatively assign proportional total.
        if seg.frame_data:
            streak = consec = 0
            for f in range(start, end + 1):
                # face_visible is not stored per-frame; approximate by whether
                # any detection has keypoints.
                if seg.frame_data.get(f, []):
                    consec += 1
                    streak = max(streak, consec)
                else:
                    consec = 0
            sub_consec_face = min(streak, seg.max_consecutive_face_frames)
        else:
            sub_consec_face = int(seg.max_consecutive_face_frames * ratio)
        new_split_seg = Segment(
            start_frame=start,
            end_frame=end,
            track_ids=seg.track_ids.copy(),
            max_persons=seg.max_persons,
            face_visible_frames=max(1, face_frames),
            max_consecutive_face_frames=sub_consec_face,
            mouth_open_frames=int(seg.mouth_open_frames * ratio),
        )
        if seg.frame_data:
            new_split_seg.frame_data = {
                f: d for f, d in seg.frame_data.items() if start <= f <= end
            }
        parts.append(new_split_seg)
        start = end + 1
    return parts


def apply_duration_split(segments: list[Segment], fps: float, max_seconds: float) -> list[Segment]:
    """Split every segment longer than *max_seconds*; keep the rest as-is."""
    max_frames = int(max_seconds * fps)
    final_segments: list[Segment] = []
    for seg in segments:
        if seg.frame_count <= max_frames:
            final_segments.append(seg)
        else:
            final_segments.extend(_split_segment(seg, max_frames))
    return final_segments
