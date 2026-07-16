"""
Shared utilities for pipeline stages.

Includes a tqdm-compatible logging handler, face visibility/frontality checks,
disk-space guards, scene-change detection, and clip/video I/O helpers.
"""

import io
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

# Honor TQDM_DISABLE=1 (set by the orchestrator) so stage output stays clean
# when run under `scripts/run_pipeline.py`. tqdm defaults to interactive
# progress bars on TTYs; when several stages run in parallel subprocesses,
# their bars interleave with the orchestrator's "=== stage START/OK ==="
# prints and become unreadable. Disabling makes stage output a plain log.
TQDM_DISABLED: bool = os.environ.get("TQDM_DISABLE", "").lower() in ("1", "true", "yes")


def make_tqdm(*args, **kwargs):
    """tqdm() factory that respects TQDM_DISABLE."""
    kwargs.setdefault("disable", TQDM_DISABLED)
    return tqdm(*args, **kwargs)


# On Windows the default stdout/stderr codec (cp1252) can't encode characters
# used in log messages (e.g. "→", "✓"). Reconfigure to UTF-8 so the tqdm-backed
# logging handler (below) never crashes on Unicode. Artifact I/O (CSV/JSON) is
# unaffected — those are written with explicit ``encoding="utf-8"``.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

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

# COCO-133 face landmark indices (23 + dlib-68 offset).
# Jawline: 23-39, eyebrows: 40-49, nose: 50-58, eyes: 59-70, lips: 71-90.
FACE_LANDMARK_INDICES: list[int] = list(range(23, 91))


# ──────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ──────────────────────────────────────────────────────────────────────────────


def _get_frames_from_crop(crop_path: Path) -> "list[np.ndarray]":
    """Read all frames from an OFIQ crop: video (.mp4) or single image (.jpg/.png).

    Returns a list of BGR uint8 arrays, one per frame. Empty on failure.
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
    image = cv2.imread(str(crop_path))
    if image is None:
        _log.warning("Cannot read image %s", crop_path.name)
        return []
    return [image]


def source_subdir_prefix(video_path: Path, input_dir: Path) -> str:
    """Derive a filename prefix that identifies the video's source subdirectory.

    For an input laid out as::

        input_dir/
          uuid-A/clip1.webm        →  "uuid-A__"
          uuid-A/sub/clip2.webm    →  "uuid-A__sub__"
          clip3.webm                →  ""

    the prefix is the subdirectory path under ``input_dir`` with ``__`` as
    the separator (filesystem-safe: ``__`` never appears in our source
    filenames). Used to keep the source subdirectory identifiable in flat
    output dirs without creating per-video subfolders (which would break
    the downstream ``glob()``-based stage discovery).
    """
    try:
        rel = video_path.relative_to(input_dir)
    except ValueError:
        return ""
    parts = rel.parent.parts  # subdirs between input_dir and the file
    return "__".join(parts) + ("__" if parts else "")


def make_output_path(output_dir: Path, video_path: Path, input_dir: Path, suffix: str = "") -> Path:
    """Compute an output path that preserves the source subdirectory in the name.

    The output directory is kept FLAT — files land directly in ``output_dir``
    — but their name starts with the source subdirectory prefix, so clips
    from different subdirs never collide and the source subdir is still
    discoverable from the filename.
    """
    prefix = source_subdir_prefix(video_path, input_dir)
    name = f"{prefix}{video_path.stem}{suffix}{video_path.suffix}"
    return output_dir / name


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

    Args:
        prev_frame: Previous BGR frame.
        curr_frame: Current BGR frame.
        hist_threshold: Luminance correlation below this triggers a cut [0, 1].
        prev_bboxes: Detection bboxes for the previous frame (N, 4) [x1,y1,x2,y2].
        curr_bboxes: Detection bboxes for the current frame (M, 4).
        bbox_area_ratio_threshold: max/min area ratio that triggers a cut.

    Returns:
        True if a scene change is detected.
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

    Args:
        path: Any path on the target filesystem (the output directory).
        min_free_gb: Minimum acceptable free space in gigabytes.
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

    Args:
        keypoints: Keypoint coordinates (N, 2).
        scores: Keypoint confidence scores (N,).
        image_height: Image height in pixels.
        min_face_size_percent: Minimum face size as % of image height.
        score_threshold: Minimum score to consider keypoint visible.

    Returns:
        True if face is visible with sufficient size.
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

    Args:
        keypoints: Keypoint coordinates (26, 2).
        scores: Keypoint confidence scores (26,).
        symmetry_threshold: Minimum symmetry ratio (0.0 - 1.0).
        score_threshold: Minimum score for keypoint visibility.

    Returns:
        True if face is considered frontal.
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


# ── Directory utilities (moved from pipeline) ──────────────────────────────────


def get_dir_size(path: Path) -> int:
    """Return total size of all files under *path* in bytes. Returns 0 if path does not exist."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


# ── Clip utilities (moved from pipeline) ───────────────────────────────────────


def save_clip_sidecar_json(
    clip_path: Path,
    metadata: dict,
) -> None:
    """Save metadata for a single clip as a sidecar JSON file.

    Writes to a sibling ``.json.partial`` temp file and atomically renames it
    into place, so concurrent downstream readers (audio_clips, face_crops_video)
    that discover clips via ``rglob("*.json")`` never observe a partially-written
    sidecar (Windows file-lock race during clip extraction). The ``.partial``
    suffix ensures ``rglob("*.json")`` does not match the in-progress file.
    """
    _log = logging.getLogger(__name__)
    sidecar_path = clip_path.with_suffix(".json")
    tmp_path = sidecar_path.with_name(sidecar_path.name + ".partial")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_path, sidecar_path)
    except OSError as e:
        _log.error(
            "Cannot write %s (%s) — removing incomplete file and stopping.",
            sidecar_path.name,
            e,
        )
        _cleanup_files(tmp_path, sidecar_path)
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
    """Extract a clip from a video file with audio.

    Writes to a sibling ``.partial`` temp file and atomically renames it into place on
    success, so concurrent downstream readers (audio_clips, face_crops_video) that scan the
    clips dir via ``rglob("*.mp4")`` never observe a partially-written, moov-less MP4. On
    Windows, a reader opening the in-progress file can lock it and prevent ffmpeg from
    finalizing the moov atom, leaving a corrupt clip (ftyp + mdat, no moov) that blocks the
    pipeline indefinitely; the temp+rename pattern breaks that race.

    The temp uses a ``.partial`` suffix (not ``.tmp.mp4``) specifically so ``rglob("*.mp4")``
    does not match it, and ffmpeg is forced to the mp4 muxer via ``-f mp4`` since the
    extension no longer signals the format. ``os.replace`` also overwrites any stale corrupt
    clip left by a prior interrupted run, self-healing the output dir.
    """
    _log = logging.getLogger(__name__)
    # Defined here so the except blocks can clean it up even if the error
    # occurs before the variable is assigned inside the try block.
    temp_audio = Path(f"temp-audio-{output_path.stem}.m4a")
    temp_clip = output_path.with_name(output_path.name + ".partial")
    try:
        from moviepy import VideoFileClip

        start_t = start_frame / fps
        end_t = (end_frame + 1) / fps

        with VideoFileClip(str(input_path)) as video:
            new_clip = video.subclipped(start_t, end_t)
            new_clip.write_videofile(
                str(temp_clip),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(temp_audio),
                remove_temp=True,
                logger=None,
                threads=4,
                ffmpeg_params=["-f", "mp4"],
            )

        if not temp_clip.exists() or temp_clip.stat().st_size == 0:
            _log.error(
                "Clip extraction produced empty/missing output for %s",
                output_path.name,
            )
            _cleanup_files(temp_clip, temp_audio)
            return False

        os.replace(temp_clip, output_path)
        return True

    except Exception as e:
        # Any error here (codec issue, I/O, malformed source, write denied,
        # audio mux failure…) is per-clip. Log it, remove any partially-written
        # files, return False so the caller continues with the next video. We
        # deliberately do NOT call sys.exit: one bad clip must not abort the
        # whole batch (e.g. 25-video run, one VP8 source that libx264 can't
        # transcode must not kill the other 24).
        _log.error("Cannot extract clip %s: %s — removing incomplete files.", output_path.name, e)
        _cleanup_files(temp_clip, temp_audio)
        return False
