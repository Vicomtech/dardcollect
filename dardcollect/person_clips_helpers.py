"""person_clips_helpers.py — helper functions for person clip extraction.

Extracted from person_clips.py to reduce size (person_clips.py: 602 → ~400 lines).
These 8 helpers are cohesive utilities for scene-change detection, frame filtering,
and progressive flush orchestration.
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.pipeline_utils import (
    check_face_visibility,
    check_frontal_face,
    scene_changed,
)
from dardcollect.poser import PoseEstimator
from dardcollect.tracker import PersonTracker, Segment

logger = logging.getLogger(__name__)


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


def is_scene_change(
    clip_config: ClipExtractionConfig,
    prev_frame: np.ndarray | None,
    frame_id: int,
    last_scene_change_frame: int,
    prev_det_bboxes: np.ndarray,
    det_bboxes: np.ndarray,
    frame: np.ndarray,
) -> bool:
    """Scene-change predicate (cooldown-gated luminance-histogram cut detector)."""
    cooldown = 8  # frames to suppress re-detection immediately after a cut
    return bool(
        clip_config.scene_change_detection
        and prev_frame is not None
        and frame_id - last_scene_change_frame > cooldown
        and scene_changed(
            prev_frame,
            frame,
            clip_config.scene_change_threshold,
            prev_det_bboxes,
            det_bboxes,
            clip_config.scene_change_bbox_area_ratio,
        )
    )


def apply_scene_change(
    frame_id: int,
    curr_segment: Segment | None,
    pending_segments: list[Segment],
    frames_since_flush: int,
    current_face_streak: int,
    *,
    fps: float,
    clip_config: ClipExtractionConfig,
    poser: PoseEstimator | None,
    face_crop_cfg: FaceCropConfig | None,
    video_path: Path,
    output_dir: Path,
    input_dir: Path,
    video_info: dict,
    clip_logger: ExtractionLogger | None,
    tracker: PersonTracker,
    flush_func,  # callable to flush_segments from person_clips
) -> tuple[Segment | None, list[Segment], int, int]:
    """Flush + reset on a scene cut: move the current segment to pending, flush
    all pending segments, reset the face streak + tracker. Returns the updated
    (curr_segment, pending_segments, frames_since_flush, current_face_streak)."""
    logger.debug("  Scene change at frame %d — flushing and resetting tracker", frame_id)
    if curr_segment is not None:
        pending_segments.append(curr_segment)
        curr_segment = None
    # Flush before processing the new scene so merge_segments() never joins
    # segments from opposite sides of the cut.
    if pending_segments:
        flush_func(
            pending_segments,
            fps=fps,
            clip_config=clip_config,
            poser=poser,
            face_crop_cfg=face_crop_cfg,
            video_path=video_path,
            output_dir=output_dir,
            input_dir=input_dir,
            video_info=video_info,
            clip_logger=clip_logger,
        )
        pending_segments = []
        frames_since_flush = 0
    current_face_streak = 0
    tracker.init_tracker()
    return curr_segment, pending_segments, frames_since_flush, current_face_streak
