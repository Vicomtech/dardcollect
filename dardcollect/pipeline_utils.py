"""
Shared utilities for extraction and processing scripts.

Contains common logging handlers and validation functions
used across multiple scripts to eliminate code duplication.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Logging Handler
# ──────────────────────────────────────────────────────────────────────────────


class _TqdmHandler(logging.StreamHandler):
    """Logging handler that routes output through tqdm to avoid breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


# ──────────────────────────────────────────────────────────────────────────────
# Face Keypoint Definitions
# ──────────────────────────────────────────────────────────────────────────────

# Face keypoint indices (from poser.py KEYPOINT_NAMES)
FACE_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}


# ──────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ──────────────────────────────────────────────────────────────────────────────


def _get_frames_from_crop(crop_path: Path) -> "list[np.ndarray]":
    """Read OFIQ frames from either a video (.mp4) or image (.jpg/.png) file.

    Returns:
        List of OFIQ frames (BGR format)
    """
    _log = logging.getLogger(__name__)
    suffix = crop_path.suffix.lower()

    if suffix == ".mp4":
        frames = []
        cap = cv2.VideoCapture(str(crop_path))
        if not cap.isOpened():
            _log.warning("Cannot open video %s", crop_path.name)
            return []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames
    # Read single image (.jpg, .png, etc.)
    image = cv2.imread(str(crop_path))
    if image is None:
        _log.warning("Cannot read image %s", crop_path.name)
        return []
    return [image]


def _cleanup_files(*paths: Path) -> None:
    """Remove partially-written files so they are not mistaken for valid output."""
    _log = logging.getLogger(__name__)
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                _log.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            _log.warning("  Could not remove %s: %s", path.name, e)


def scene_changed(
    prev_frame: "np.ndarray",
    curr_frame: "np.ndarray",
    hist_threshold: float,
    prev_bboxes: "np.ndarray",
    curr_bboxes: "np.ndarray",
    bbox_area_ratio_threshold: float,
) -> bool:
    """Detect a hard scene cut using two complementary signals.

    **Signal 1 – Luminance histogram correlation**
    Frames from the same shot share similar brightness distributions (correlation
    typically > 0.85). A hard cut produces an abrupt change; fires when
    correlation drops below *hist_threshold*. Works on colour and greyscale.

    **Signal 2 – Detection bounding-box area ratio**
    A cut between a wide shot and a close-up of the same person is invisible to
    luminance histograms but shows up as a large ratio between the maximum
    detection area in consecutive frames. Fires when that ratio exceeds
    *bbox_area_ratio_threshold*.

    A scene change is declared when either signal fires.

    :param prev_frame: Previous BGR frame.
    :param curr_frame: Current BGR frame.
    :param hist_threshold: Luminance correlation below this triggers a cut [0, 1].
    :param prev_bboxes: Detection bboxes for the previous frame (N, 4) [x1,y1,x2,y2].
    :param curr_bboxes: Detection bboxes for the current frame (M, 4).
    :param bbox_area_ratio_threshold: max/min area ratio that triggers a cut.
    :return: True if a scene change is detected.
    """
    # ── Signal 1: luminance histogram ────────────────────────────────────────
    small_prev = cv2.resize(prev_frame, (128, 72), interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(curr_frame, (128, 72), interpolation=cv2.INTER_AREA)

    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)

    hist_prev = cv2.calcHist([gray_prev], [0], None, [64], [0, 256])
    hist_curr = cv2.calcHist([gray_curr], [0], None, [64], [0, 256])
    cv2.normalize(hist_prev, hist_prev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    if float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)) < hist_threshold:
        return True

    # ── Signal 2: detection bbox area ratio ───────────────────────────────────
    if len(prev_bboxes) > 0 and len(curr_bboxes) > 0:

        def _max_area(bboxes: "np.ndarray") -> float:
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            return float(np.max(widths * heights))

        prev_area = _max_area(prev_bboxes)
        curr_area = _max_area(curr_bboxes)

        if prev_area > 0 and curr_area > 0:
            ratio = max(prev_area / curr_area, curr_area / prev_area)
            if ratio >= bbox_area_ratio_threshold:
                return True

    return False


def _check_disk_space(path: Path, min_gb: float) -> None:
    """Raise RuntimeError if free disk space on *path* is below *min_gb* gigabytes."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {path} (minimum {min_gb} GB required)")


def check_disk_space(path: Path, min_free_gb: float) -> None:
    """Exit immediately if the filesystem hosting *path* is nearly full.

    Called proactively before every write so the process never produces a
    truncated or corrupted output file.

    :param path: Any path on the target filesystem (the output directory).
    :param min_free_gb: Minimum acceptable free space in gigabytes.
    """
    logger = logging.getLogger(__name__)
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as e:
        # The filesystem may be temporarily unreachable (e.g. expired Kerberos
        # ticket, network hiccup).  We cannot confirm disk space is low, so we
        # warn and let the code proceed.  If the filesystem is truly unavailable
        # the subsequent write call will raise its own OSError and be handled
        # there (with partial-file cleanup and a clean exit).
        logger.warning("Cannot check disk space on %s: %s — skipping check.", path, e)
        return
    free_gb = free_bytes / (1024**3)
    if free_gb < min_free_gb:
        logger.error(
            "Disk space critically low: %.2f GB free on %s (need %.1f GB) — stopping.",
            free_gb,
            path,
            min_free_gb,
        )
        sys.exit(1)


def check_face_visibility(
    keypoints: "np.ndarray",
    scores: "np.ndarray",
    image_height: int,
    min_face_size_percent: float,
    score_threshold: float = 0.3,
) -> bool:
    """Check if a face is visible with sufficient size.

    A face is considered visible if key facial keypoints are detected,
    the face size is at least min_face_size_percent of image height,
    and the eyes are sufficiently far apart (rejects hallucinated clusters).

    :param keypoints: Keypoint coordinates (N, 2).
    :param scores: Keypoint confidence scores (N,).
    :param image_height: Image height in pixels.
    :param min_face_size_percent: Minimum face size as % of image height.
    :param score_threshold: Minimum score to consider keypoint visible.
    :return: True if face is visible with sufficient size.
    """
    # Check if at least nose and one eye are visible
    nose_visible = scores[FACE_KEYPOINTS["nose"]] >= score_threshold
    left_eye_visible = scores[FACE_KEYPOINTS["left_eye"]] >= score_threshold
    right_eye_visible = scores[FACE_KEYPOINTS["right_eye"]] >= score_threshold

    if not (nose_visible and (left_eye_visible or right_eye_visible)):
        return False

    # Reject hallucinated face keypoints: when both eyes are visible but
    # suspiciously close together the model is placing them on the body, not
    # on an actual face. Require inter-eye distance ≥ 1% of image height.
    min_eye_dist_px = image_height * 0.01
    if left_eye_visible and right_eye_visible:
        eye_dist = np.linalg.norm(
            keypoints[FACE_KEYPOINTS["left_eye"]] - keypoints[FACE_KEYPOINTS["right_eye"]]
        )
        if eye_dist < min_eye_dist_px:
            return False

    # Calculate face size from visible face keypoints
    face_points = []
    for idx in FACE_KEYPOINTS.values():
        if scores[idx] >= score_threshold and keypoints[idx][0] >= 0:
            face_points.append(keypoints[idx])

    if len(face_points) < 2:
        return False

    face_points = np.array(face_points)
    min_y = np.min(face_points[:, 1])
    max_y = np.max(face_points[:, 1])
    face_height = max_y - min_y
    estimated_face_height = face_height * 2.5

    min_face_size = image_height * (min_face_size_percent / 100.0)
    return estimated_face_height >= min_face_size


def check_frontal_face(
    keypoints: "np.ndarray",
    scores: "np.ndarray",
    symmetry_threshold: float,
    score_threshold: float = 0.3,
) -> bool:
    """Check if a face is frontal using ear visibility and symmetry.

    :param keypoints: Keypoint coordinates (26, 2).
    :param scores: Keypoint confidence scores (26,).
    :param symmetry_threshold: Minimum symmetry ratio (0.0 - 1.0).
    :param score_threshold: Minimum score for keypoint visibility.
    :return: True if face is considered frontal.
    """
    nose_idx = FACE_KEYPOINTS["nose"]
    l_ear_idx = FACE_KEYPOINTS["left_ear"]
    r_ear_idx = FACE_KEYPOINTS["right_ear"]
    l_eye_idx = FACE_KEYPOINTS["left_eye"]
    r_eye_idx = FACE_KEYPOINTS["right_eye"]

    # Require nose and both eyes — eyes are almost never detected on the back of a
    # head, so this already prevents false frontal passes for rear-facing persons.
    if (
        scores[nose_idx] < score_threshold
        or scores[l_eye_idx] < score_threshold
        or scores[r_eye_idx] < score_threshold
    ):
        return False

    l_ear_visible = scores[l_ear_idx] >= score_threshold
    r_ear_visible = scores[r_ear_idx] >= score_threshold

    # If neither ear is visible we cannot assess yaw; pass the check (eyes already
    # confirmed the person is facing roughly forward).
    if not l_ear_visible and not r_ear_visible:
        return True

    # If only one ear is visible, the face is clearly rotated toward the hidden ear
    # but not necessarily in profile — pass only when the visible ear is far enough
    # from the nose to confirm the head isn't in sharp profile.
    nose_x = keypoints[nose_idx][0]
    if l_ear_visible and not r_ear_visible:
        # Only left ear visible: check it isn't implausibly close to the nose
        dist = abs(keypoints[l_ear_idx][0] - nose_x)
        face_width_est = abs(keypoints[l_eye_idx][0] - keypoints[r_eye_idx][0]) * 2.5 + 1.0
        return dist / face_width_est >= (1.0 - symmetry_threshold)
    if r_ear_visible and not l_ear_visible:
        dist = abs(keypoints[r_ear_idx][0] - nose_x)
        face_width_est = abs(keypoints[l_eye_idx][0] - keypoints[r_eye_idx][0]) * 2.5 + 1.0
        return dist / face_width_est >= (1.0 - symmetry_threshold)

    # Both ears visible: use the standard nose-to-ear symmetry ratio.
    dist_l = abs(keypoints[l_ear_idx][0] - nose_x)
    dist_r = abs(keypoints[r_ear_idx][0] - nose_x)
    if dist_l == 0 or dist_r == 0:
        return True
    ratio = min(dist_l, dist_r) / max(dist_l, dist_r)
    return ratio >= symmetry_threshold


# ── Directory utilities (moved from scripts) ──────────────────────────────────


def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ── Clip utilities (moved from scripts) ───────────────────────────────────────


def save_clip_sidecar_json(
    clip_path: Path,
    metadata: dict,
) -> None:
    """Save metadata for a single clip as a sidecar JSON file."""
    _log = logging.getLogger(__name__)
    sidecar_path = clip_path.with_suffix(".json")

    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        _log.error(
            "Cannot write %s (%s) — removing incomplete file and stopping.",
            sidecar_path.name,
            e,
        )
        _cleanup_files(sidecar_path)
        sys.exit(1)


def _write_video_with_moviepy(
    frames: "list[np.ndarray]",
    output_path: Path,
    fps: float,
) -> bool:
    """Write frames to MP4 using moviepy (same as extracted_person_clips).

    Args:
        frames: List of BGR numpy arrays (H, W, 3)
        output_path: Output MP4 file path
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    _log = logging.getLogger(__name__)
    if not frames:
        _log.error("No frames to write")
        return False

    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Convert BGR to RGB (moviepy uses RGB)
        rgb_frames = [cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in frames]

        # Create a VideoClip from the frames using ImageSequenceClip
        clip = ImageSequenceClip(rgb_frames, durations=[1.0 / fps] * len(rgb_frames))

        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=4,
        )

        success = output_path.exists() and output_path.stat().st_size > 0
        if not success:
            _log.error("Output file is missing or empty")
            return False

        return True

    except Exception as e:
        _log.error("Error writing video with moviepy: %s", e)
        return False


def extract_clip(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> bool:
    """Extract a clip from a video file with audio."""
    _log = logging.getLogger(__name__)
    # Defined here so the except blocks can clean it up even if the error
    # occurs before the variable is assigned inside the try block.
    temp_audio = Path(f"temp-audio-{output_path.stem}.m4a")
    try:
        from moviepy import VideoFileClip

        start_t = start_frame / fps
        end_t = (end_frame + 1) / fps

        with VideoFileClip(str(input_path)) as video:
            new_clip = video.subclipped(start_t, end_t)
            new_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(temp_audio),
                remove_temp=True,
                logger=None,
                threads=4,
            )

        return True

    except OSError as e:
        # Any OS-level write failure (no space, permission denied, read-only
        # filesystem, quota exceeded, …) is unrecoverable — stop cleanly.
        _log.error(
            "Cannot write %s (%s) — removing incomplete files and stopping.",
            output_path.name,
            e,
        )
        _cleanup_files(output_path, temp_audio)
        sys.exit(1)
    except Exception as e:
        # Non-I/O errors (malformed source video, codec issue, …): log and
        # skip this clip, but clean up whatever was partially written.
        _log.error("Error extracting clip %s: %s", output_path.name, e)
        _cleanup_files(output_path, temp_audio)
        return False
