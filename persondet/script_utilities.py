"""
Shared utilities for extraction and processing scripts.

Contains common logging handlers and validation functions
used across multiple scripts to eliminate code duplication.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
