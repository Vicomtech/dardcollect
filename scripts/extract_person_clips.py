#!/usr/bin/env python3
"""
Extract video clips containing people from downloaded videos.

Uses person detection and tracking to identify segments where
people are visible, then extracts those clips as separate files.

All parameters are read from config.yaml.
"""

import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scipy.signal import savgol_filter
from tqdm import tqdm


# Configure logging — route through tqdm so output doesn't break progress bars
class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Setup paths BEFORE importing libraries that might load DLLs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persondet.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import numpy as np
from moviepy import VideoFileClip

import persondet
from persondet import PersonDetector, PersonTracker, PoseEstimator
from persondet.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig
from persondet.face_geometry import face_crop_corners as _compute_face_crop_corners
from persondet.provenance import PROVENANCE_FILENAME, model_info, now_iso, record_stage
from persondet.tracker import TrackingParams


@dataclass
class Segment:
    """Represents a video segment with people."""

    start_frame: int
    end_frame: int
    track_ids: list[int] = field(default_factory=list)
    max_persons: int = 0
    face_visible_frames: int = 0  # Total frames with a visible face
    max_consecutive_face_frames: int = 0  # Longest unbroken run of face-visible frames
    mouth_open_frames: int = 0  # Frames where mouth is detected as open
    frame_data: dict = field(default_factory=dict)  # frame_idx -> list of detection dicts

    @property
    def frame_count(self) -> int:
        """Number of frames in segment."""
        return self.end_frame - self.start_frame + 1

    def duration_seconds(self, fps: float) -> float:
        """Duration in seconds."""
        return self.frame_count / fps if fps > 0 else 0


def suppress_by_keypoints(
    tracklets_kpts, dist_threshold: float = 0.15, score_threshold: float = 0.3
):
    """Remove duplicate tracklets whose keypoints land in nearly the same position.

    Two detections are considered the same person when their mean keypoint
    distance (averaged over mutually-visible keypoints, normalised by the
    larger detection's height) is below *dist_threshold*.  The lower-score
    detection is dropped.

    :param tracklets_kpts: list of (tracklet, keypoints ndarray, kpt_scores ndarray)
    :param dist_threshold: normalised distance below which two detections are duplicates.
    :param score_threshold: minimum keypoint confidence to include in comparison.
    :return: filtered list of the same form.
    """
    if len(tracklets_kpts) < 2:
        return tracklets_kpts

    det_scores = [t.det_score for t, _, _ in tracklets_kpts]
    order = sorted(range(len(tracklets_kpts)), key=lambda i: det_scores[i], reverse=True)

    suppressed = set()
    for pos, i in enumerate(order):
        if i in suppressed:
            continue
        t_i, kpts_i, kscores_i = tracklets_kpts[i]
        if kpts_i is None or kscores_i is None:
            continue
        box_i = t_i.tlbr
        scale = max(box_i[3] - box_i[1], 1.0)  # person height

        for j in order[pos + 1 :]:
            if j in suppressed:
                continue
            _, kpts_j, kscores_j = tracklets_kpts[j]
            if kpts_j is None or kscores_j is None:
                continue

            visible = (kscores_i >= score_threshold) & (kscores_j >= score_threshold)
            if visible.sum() < 3:
                continue

            mean_dist = np.linalg.norm(kpts_i[visible] - kpts_j[visible], axis=1).mean()
            if mean_dist / scale < dist_threshold:
                t_j = tracklets_kpts[j][0]
                logger.debug(
                    "  KPT-suppress track %d (dup of track %d, norm_dist=%.3f)",
                    t_j.track_id,
                    t_i.track_id,
                    mean_dist / scale,
                )
                suppressed.add(j)

    return [tracklets_kpts[i] for i in range(len(tracklets_kpts)) if i not in suppressed]


def suppress_overlapping_tracklets(tracklets, iou_threshold: float = 0.5):
    """Remove duplicate tracklets that cover the same person.

    When two active tracks overlap above *iou_threshold*, the one with the
    lower det_score is dropped.  This fixes cases where a single person
    spawned two separate tracks because the detector returned two overlapping
    boxes in earlier frames.
    """
    if len(tracklets) < 2:
        return tracklets

    boxes = np.array([t.tlbr for t in tracklets])
    scores = np.array([t.det_score for t in tracklets])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        # IoMin catches the case where one box is fully inside a much larger one:
        # standard IoU stays low (small/large ratio) but IoMin hits 1.0.
        iomin = inter / (np.minimum(areas[i], areas[rest]) + 1e-6)
        order = rest[(iou < iou_threshold) & (iomin < iou_threshold)]

    return [tracklets[i] for i in keep]


# Face keypoint indices (from poser.py KEYPOINT_NAMES)
FACE_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
}


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

    # Estimate face size from bounding box of face keypoints
    face_points = np.array(face_points)
    min_y = np.min(face_points[:, 1])
    max_y = np.max(face_points[:, 1])
    face_height = max_y - min_y

    # Face height needs to be scaled up (keypoints are just eyes/nose/ears)
    # Approximate full face height as ~2x the distance between keypoints
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


def check_disk_space(path: Path, min_free_gb: float) -> None:
    """Exit immediately if the filesystem hosting *path* is nearly full.

    Called proactively before every write so the process never produces a
    truncated or corrupted output file.

    :param path: Any path on the target filesystem (the output directory).
    :param min_free_gb: Minimum acceptable free space in gigabytes.
    """
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


def _cleanup_files(*paths: Path) -> None:
    """Remove partially-written files so they are not mistaken for valid output."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            logger.warning("  Could not remove %s: %s", path.name, e)


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


def merge_segments(segments: list[Segment], gap_frames: int) -> list[Segment]:
    """Merge adjacent segments with small gaps."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start_frame)
    merged = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg.start_frame <= last.end_frame + gap_frames:
            last.end_frame = max(last.end_frame, seg.end_frame)
            last.track_ids = list(set(last.track_ids + seg.track_ids))
            last.max_persons = max(last.max_persons, seg.max_persons)
            last.face_visible_frames += seg.face_visible_frames
            last.max_consecutive_face_frames = max(
                last.max_consecutive_face_frames, seg.max_consecutive_face_frames
            )
            last.mouth_open_frames += seg.mouth_open_frames
            last.frame_data.update(seg.frame_data)
        else:
            merged.append(seg)

    return merged


def save_clip_sidecar_json(
    clip_path: Path,
    metadata: dict,
) -> None:
    """Save metadata for a single clip as a sidecar JSON file."""
    sidecar_path = clip_path.with_suffix(".json")

    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        logger.error(
            "Cannot write %s (%s) — removing incomplete file and stopping.",
            sidecar_path.name,
            e,
        )
        _cleanup_files(sidecar_path)
        sys.exit(1)


def extract_clip(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> bool:
    """Extract a clip from a video file with audio."""
    # Defined here so the except blocks can clean it up even if the error
    # occurs before the variable is assigned inside the try block.
    temp_audio = Path(f"temp-audio-{output_path.stem}.m4a")
    try:
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
        logger.error(
            "Cannot write %s (%s) — removing incomplete files and stopping.",
            output_path.name,
            e,
        )
        _cleanup_files(output_path, temp_audio)
        sys.exit(1)
    except Exception as e:
        # Non-I/O errors (malformed source video, codec issue, …): log and
        # skip this clip, but clean up whatever was partially written.
        logger.error("Error extracting clip %s: %s", output_path.name, e)
        _cleanup_files(output_path, temp_audio)
        return False


def process_video(
    video_path: Path,
    detector: PersonDetector,
    tracker: PersonTracker,
    det_config: DetectorConfig,
    clip_config: ClipExtractionConfig,
    poser: PoseEstimator | None = None,
    face_crop_cfg: FaceCropConfig | None = None,
) -> list[dict]:
    """Process a video to find and extract person clips."""
    logger.info("Processing: %s", video_path.name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = total_frames / fps if fps > 0 else 0
    logger.info(
        "  Video: %dx%d, %.1f fps, %d frames (%.1f sec)",
        width,
        height,
        fps,
        total_frames,
        duration,
    )

    tracker.init_tracker()

    # Prepare output directory
    output_dir = Path(clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize video info for per-file JSON
    video_info = {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
    }

    # Tracking state
    pending_segments: list[Segment] = []  # Segments waiting to be flushed
    curr_segment: Segment | None = None  # Current active segment
    current_face_streak: int = 0  # Consecutive frames with a visible face (current segment)

    # State for progressive JSON writing
    # Removed monolithic JSON tracking to improve FPS stability
    pass

    def smooth_segment_keypoints(
        seg: Segment, window_seconds: float = 0.25, polyorder: int = 2
    ) -> None:
        """Smooth keypoints in-place per track using a Savitzky-Golay filter.

        Since extraction is offline we can look at the full segment at once,
        so a polynomial filter beats a causal moving average: it preserves
        peaks and motion onsets while killing frame-to-frame jitter.
        """
        if not seg.frame_data:
            return

        frames = sorted(seg.frame_data.keys())
        if len(frames) < 5:
            return

        # Window must be odd, at least polyorder+2, at most len(frames)
        win = max(int(window_seconds * fps) | 1, polyorder + 2)  # bitwise OR 1 → odd
        if win % 2 == 0:
            win += 1
        win = min(win, len(frames))
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 1:
            return

        # Collect all track IDs present in this segment
        track_ids = {p["track_id"] for fd in seg.frame_data.values() for p in fd}

        for tid in track_ids:
            # Ordered (frame, person_dict) pairs for this track
            track_entries = [
                (f, p)
                for f in frames
                for p in seg.frame_data[f]
                if p["track_id"] == tid and "keypoints" in p
            ]
            if len(track_entries) < win:
                continue

            kpts = np.array([e[1]["keypoints"] for e in track_entries])  # (T, K, 2)
            scores = np.array([e[1]["keypoint_scores"] for e in track_entries])  # (T, K)

            n_kpts = kpts.shape[1]
            smoothed = kpts.copy()
            for k in range(n_kpts):
                # Skip keypoints that are consistently low-confidence (noise/hallucination)
                if scores[:, k].mean() < 0.15:
                    continue
                smoothed[:, k, 0] = savgol_filter(kpts[:, k, 0], win, polyorder)
                smoothed[:, k, 1] = savgol_filter(kpts[:, k, 1], win, polyorder)

            for (f, person), new_kpts in zip(track_entries, smoothed):
                person["keypoints"] = [[round(x, 1), round(y, 1)] for x, y in new_kpts.tolist()]

    def _annotate_face_crop_corners(seg: Segment, fcfg: FaceCropConfig) -> None:
        """Add 'face_crop_corners' to each detection entry that has eye keypoints.

        Called after smooth_segment_keypoints so corners reflect the smoothed
        positions.  Corners are 4 source-frame points [TL, TR, BR, BL] stored
        as a list of [x, y] pairs, independent of output_size.
        """
        for frame_detections in seg.frame_data.values():
            for person in frame_detections:
                if "keypoints" not in person or "keypoint_scores" not in person:
                    continue
                kpts = np.array(person["keypoints"], dtype=np.float32)
                kscores = np.array(person["keypoint_scores"], dtype=np.float32)
                corners = _compute_face_crop_corners(
                    kpts,
                    kscores,
                    align_face=fcfg.align_face,
                    face_padding=fcfg.face_padding,
                    keypoint_threshold=fcfg.pose_keypoint_threshold,
                    min_eye_distance_px=fcfg.min_eye_distance_px,
                )
                if corners is not None:
                    person["face_crop_corners"] = [
                        [round(float(x), 2), round(float(y), 2)] for x, y in corners
                    ]

    def flush_segments(segments_to_flush: list[Segment], force: bool = False) -> list[dict]:
        """Process, filter, extract, and save segments."""
        if not segments_to_flush:
            return []

        # Merge compatible segments within this batch
        merged = merge_segments(segments_to_flush, clip_config.merge_gap_frames)

        # Apply filters
        filtered = [
            s
            for s in merged
            if s.frame_count >= clip_config.min_consecutive_frames
            and s.duration_seconds(fps) >= clip_config.min_clip_duration_seconds
        ]

        # Face visibility filter
        if clip_config.require_face_visibility and poser is not None:
            filtered = [
                s
                for s in filtered
                if s.face_visible_frames >= clip_config.min_face_visible_frames
                and s.max_consecutive_face_frames >= clip_config.min_consecutive_face_frames
            ]

        # Handle max duration splitting
        max_frames = int(clip_config.max_clip_duration_seconds * fps)
        final_segments = []
        for seg in filtered:
            if seg.frame_count <= max_frames:
                final_segments.append(seg)
            else:
                # Split logic
                start = seg.start_frame
                while start < seg.end_frame:
                    end = min(start + max_frames - 1, seg.end_frame)
                    ratio = (end - start + 1) / seg.frame_count
                    face_frames = int(seg.face_visible_frames * ratio)
                    # Consecutive face frames: recount from the sub-clip's frame_data
                    # if available; otherwise conservatively assign proportional total.
                    sub_frames = range(start, end + 1)
                    if seg.frame_data:
                        streak = consec = 0
                        for f in sub_frames:
                            fd = seg.frame_data.get(f, [])
                            # face_visible flag is not stored per-frame; approximate by
                            # checking if any person has keypoints (face data present).
                            if fd:
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
                        sub_data = {f: d for f, d in seg.frame_data.items() if start <= f <= end}
                        new_split_seg.frame_data = sub_data
                    final_segments.append(new_split_seg)
                    start = end + 1

        filtered = final_segments

        for seg in filtered:
            smooth_segment_keypoints(seg)

        if face_crop_cfg is not None:
            for seg in filtered:
                _annotate_face_crop_corners(seg, face_crop_cfg)

        # Convert to dicts and store/save
        batch_seg_dicts = []
        batch_clip_metas = []

        for seg in filtered:
            # Metadata dict
            s_dict = {
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "start_seconds": round(seg.start_frame / fps, 2),
                "end_seconds": round(seg.end_frame / fps, 2),
                "duration_seconds": round(seg.duration_seconds(fps), 2),
                "max_persons": seg.max_persons,
                "unique_tracks": len(seg.track_ids),
                "track_ids": seg.track_ids,
                "face_visible_frames": seg.face_visible_frames,
                "max_consecutive_face_frames": seg.max_consecutive_face_frames,
                "frame_data": seg.frame_data,
            }
            batch_seg_dicts.append(s_dict)

            # Clip extraction
            start_sec = seg.start_frame / fps
            end_sec = seg.end_frame / fps
            start_str = f"{int(start_sec // 60):02d}m{int(start_sec % 60):02d}s"
            end_str = f"{int(end_sec // 60):02d}m{int(end_sec % 60):02d}s"
            clip_name = f"{video_path.stem}_{start_str}-{end_str}.mp4"
            clip_path = output_dir / clip_name

            meta = {
                "source_video": video_path.as_posix(),
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "start_seconds": round(start_sec, 2),
                "end_seconds": round(end_sec, 2),
                "duration_seconds": round(seg.duration_seconds(fps), 2),
                "max_persons": seg.max_persons,
                "unique_tracks": len(seg.track_ids),
                "track_ids": sorted(seg.track_ids),
                "video_info": video_info,
                "face_visible_frames": seg.face_visible_frames,
                "max_consecutive_face_frames": seg.max_consecutive_face_frames,
                "mouth_open_frames": seg.mouth_open_frames,
                "frame_data": seg.frame_data,
            }

            # 1. Extract Clip
            extraction_success = False
            check_disk_space(output_dir, clip_config.min_free_disk_gb)
            logger.info("  Extracting: %s (%.1fs)", clip_name, meta["duration_seconds"])
            t0 = time.time()
            if extract_clip(video_path, clip_path, seg.start_frame, seg.end_frame, fps):
                t_extract = time.time() - t0
                logger.info("  Extraction took %.2fs", t_extract)
                meta["clip_path"] = clip_path.as_posix()
                extraction_success = True
            else:
                meta["error"] = "Extraction failed"

            # 2. Transcription - REMOVED (Handled by separate script)
            # Initialize empty transcription field for schema consistency
            meta["transcription"] = ""

            # 3. Save Sidecar JSON
            if extraction_success:
                save_clip_sidecar_json(clip_path, meta)

            batch_clip_metas.append(meta)

        return batch_clip_metas

    track_params = TrackingParams(
        score_threshold=det_config.tracking_score_threshold,
        min_hits=det_config.tracking_min_hits,
        max_time_lost=det_config.tracking_max_time_lost,
    )

    # RESUME LOGIC
    progress_path = output_dir / f"{video_path.stem}_progress.json"
    start_frame = 0

    if progress_path.exists():
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
                    start_frame = last_frame + 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        except Exception as e:
            logger.warning("  Failed to load progress file, starting from 0: %s", e)

    frame_id = start_frame
    frames_since_flush = 0
    prev_frame: np.ndarray | None = None
    prev_det_bboxes: np.ndarray = np.empty((0, 4))  # for scene-change detection
    last_scene_change_frame: int = start_frame - 1  # cooldown tracker

    pbar = tqdm(
        total=total_frames,
        initial=start_frame,
        unit="fr",
        desc=video_path.name[:40],
        dynamic_ncols=True,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect first so both frames' bboxes are available for scene-change
        # detection before the tracker state is updated.
        det_bboxes, det_scores = detector.get_detections(frame, det_config.detection_threshold)

        # Filter out detections whose bounding box covers an implausibly large
        # fraction of the frame (title cards, scene-wide text overlays, etc.)
        # or is too wide relative to its height (saddles, furniture, animals, etc.).
        if len(det_bboxes) > 0:
            frame_area = width * height
            box_w = det_bboxes[:, 2] - det_bboxes[:, 0]
            box_h = det_bboxes[:, 3] - det_bboxes[:, 1]
            bbox_areas = box_w * box_h
            aspect_ratios = box_w / np.maximum(box_h, 1.0)
            keep = (bbox_areas / frame_area <= clip_config.max_bbox_area_percent / 100.0) & (
                aspect_ratios <= clip_config.max_detection_aspect_ratio
            )
            det_bboxes = det_bboxes[keep]
            det_scores = det_scores[keep]

        # ── Scene-change detection ────────────────────────────────────────────
        _SCENE_CHANGE_COOLDOWN = 8  # frames to suppress re-triggering after a cut
        if (
            clip_config.scene_change_detection
            and prev_frame is not None
            and frame_id - last_scene_change_frame > _SCENE_CHANGE_COOLDOWN
            and scene_changed(
                prev_frame,
                frame,
                clip_config.scene_change_threshold,
                prev_det_bboxes,
                det_bboxes,
                clip_config.scene_change_bbox_area_ratio,
            )
        ):
            logger.debug(
                "  Scene change at frame %d — flushing and resetting tracker",
                frame_id,
            )
            last_scene_change_frame = frame_id
            if curr_segment is not None:
                pending_segments.append(curr_segment)
                curr_segment = None
            # Flush immediately so merge_segments() never sees segments
            # from both sides of the cut in the same batch.
            if pending_segments:
                flush_segments(pending_segments)
                pending_segments = []
                frames_since_flush = 0
            current_face_streak = 0
            tracker.init_tracker()

        prev_frame = frame
        prev_det_bboxes = det_bboxes

        tracklets = tracker.update(det_bboxes.tolist(), det_scores.tolist(), track_params)

        # Compute keypoints for all surviving tracklets upfront so we can
        # (a) run keypoint-based duplicate suppression, and
        # (b) reuse the results for face-visibility checks and frame data
        # without calling the pose model twice per tracklet.
        if tracklets and poser is not None:
            tracklets_kpts = [(t, *poser.get_keypoints(frame, t.tlbr.tolist())) for t in tracklets]
            tracklets_kpts = suppress_by_keypoints(
                tracklets_kpts,
                dist_threshold=0.15,
                score_threshold=det_config.pose_keypoint_threshold,
            )
            tracklets = [t for t, _, _ in tracklets_kpts]
        else:
            tracklets_kpts = [(t, None, None) for t in tracklets]

        face_visible = False
        mouth_open = False
        if tracklets:
            track_ids = [t.track_id for t in tracklets]

            # Check face visibility using already-computed keypoints.
            # Iterate ALL tracks so that mouth_open is checked for every
            # visible-face person, not just the first one found.
            if poser is not None and clip_config.require_face_visibility:
                for t, keypoints, kpt_scores in tracklets_kpts:
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
                        if clip_config.enable_visual_speaking:
                            if poser.check_mouth_open(
                                keypoints,
                                kpt_scores,
                                min_score=det_config.pose_keypoint_threshold,
                            ):
                                mouth_open = True

            # Collect detailed frame data using already-computed keypoints
            current_frame_data = []
            for t, kpts, kpt_scores in tracklets_kpts:
                data_entry = {
                    "track_id": t.track_id,
                    "bbox": [round(x, 1) for x in t.tlbr.tolist()],
                    "score": round(float(t.det_score), 3),
                }
                if kpts is not None:
                    assert kpt_scores is not None, "kpt_scores should be set when kpts is set"
                    data_entry["keypoints"] = [[round(x, 1), round(y, 1)] for x, y in kpts.tolist()]
                    data_entry["keypoint_scores"] = [
                        round(float(s), 3) for s in kpt_scores.tolist()
                    ]

                current_frame_data.append(data_entry)

            if face_visible:
                current_face_streak += 1
            else:
                current_face_streak = 0

            if curr_segment is None:
                new_seg = Segment(
                    start_frame=frame_id,
                    end_frame=frame_id,
                    track_ids=track_ids,
                    max_persons=len(tracklets),
                    face_visible_frames=1 if face_visible else 0,
                    max_consecutive_face_frames=current_face_streak if face_visible else 0,
                    mouth_open_frames=1 if mouth_open else 0,
                )
                if current_frame_data:
                    new_seg.frame_data[frame_id] = current_frame_data
                curr_segment = new_seg
            else:
                curr_segment.end_frame = frame_id
                curr_segment.track_ids = list(set(curr_segment.track_ids + track_ids))
                curr_segment.max_persons = max(curr_segment.max_persons, len(tracklets))
                if face_visible:
                    curr_segment.face_visible_frames += 1
                    curr_segment.max_consecutive_face_frames = max(
                        curr_segment.max_consecutive_face_frames, current_face_streak
                    )
                if mouth_open:
                    curr_segment.mouth_open_frames += 1
                if current_frame_data:
                    curr_segment.frame_data[frame_id] = current_frame_data
        else:
            # No people in this frame — reset face streak
            current_face_streak = 0
            if curr_segment is not None:
                # Segment finished, move to pending
                pending_segments.append(curr_segment)
                curr_segment = None

        frame_id += 1
        frames_since_flush += 1

        # Check for progressive flush
        # Trigger if we have pending segments and sufficient gap or time
        if pending_segments:
            last_seg_end = pending_segments[-1].end_frame
            # Distance from "now" (frame_id)
            gap = frame_id - last_seg_end

            # If gap is large enough, the pending segments are likely stable
            # Flush every 30 seconds of video processing or if massive gap
            if gap > max(clip_config.merge_gap_frames * 2, 30) or frames_since_flush > 30 * fps:
                flush_segments(pending_segments)
                pending_segments = []  # Clear memory
                frames_since_flush = 0

                # Update progress file
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

        pbar.update(1)

    pbar.close()

    # End of video: Flush everything remaining
    if curr_segment is not None:
        pending_segments.append(curr_segment)

    if pending_segments:
        flush_segments(pending_segments, force=True)

    cap.release()

    # SUCCESS: Remove progress file if it exists
    if progress_path.exists():
        try:
            progress_path.unlink()
        except OSError:
            pass

    return []


def main():
    """Main entry point."""
    started_at = now_iso()

    # Load configuration
    try:
        det_config = DetectorConfig.from_yaml(str(CONFIG_PATH))
        clip_config = ClipExtractionConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    face_crop_cfg: FaceCropConfig | None = None
    try:
        face_crop_cfg = FaceCropConfig.from_yaml(str(CONFIG_PATH))
        logger.info(
            "Face crop config loaded — will annotate face_crop_corners (align_face=%s)",
            face_crop_cfg.align_face,
        )
    except Exception:
        logger.info("No face_crop_extraction config found — face_crop_corners will be skipped")

    # Get input path
    input_path = Path(clip_config.input_dir)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    # Collect video files
    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = list(input_path.glob("*.mp4"))
        video_files.extend(input_path.glob("*.avi"))
        video_files.extend(input_path.glob("*.mkv"))
        video_files.extend(input_path.glob("*.mov"))

    if not video_files:
        logger.error("No video files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d video(s) to process", len(video_files))

    # Select model (Updated to YOLOX-Tiny HumanArt for User Request)
    models_dir = Path(det_config.models_path)

    det_filename = "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_filename = "cigpose-m_coco-wholebody_256x192.onnx"

    det_model_path = models_dir / det_filename
    pose_model_path = models_dir / pose_filename

    if not det_model_path.exists():
        logger.error("Detection model not found: %s", det_model_path)
        logger.error("Run scripts/setup_models.py first!")
        sys.exit(1)

    # Initialize components
    logger.info("Initializing detector (%s)...", det_model_path.name)
    detector = PersonDetector(det_config, model_path=str(det_model_path))

    logger.info("Initializing tracker (OC-SORT)...")
    tracker = PersonTracker()

    logger.info("Initializing pose estimator (%s)...", pose_model_path.name)
    poser = PoseEstimator(det_config, model_path=str(pose_model_path))

    # Audio Transcriber - Removed from main loop
    # run scripts/transcribe_clips.py instead

    # Process videos
    output_dir = Path(clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for video_path in video_files:
        done_sentinel = output_dir / f"{video_path.stem}.done"
        if done_sentinel.exists():
            logger.info("SKIP (already done): %s", video_path.name)
            continue

        try:
            results = process_video(
                video_path, detector, tracker, det_config, clip_config, poser, face_crop_cfg
            )
            all_results.extend(results)
            done_sentinel.touch()
        except Exception as e:
            logger.error("Error processing %s: %s", video_path.name, e)

    # Per-file detection JSONs are saved after each video is processed

    # Summary
    total_clips = len(all_results)
    total_duration = sum(r.get("duration_seconds", 0) for r in all_results)
    logger.info(
        "\nSummary: Extracted %d clips (%.1f seconds total)",
        total_clips,
        total_duration,
    )

    record_stage(
        output_dir.parent / PROVENANCE_FILENAME,
        {
            "stage": "extract_person_clips",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {
                "script": "scripts/extract_person_clips.py",
                "persondet_version": persondet.__version__,
            },
            "models": {
                "detector": model_info(det_model_path),
                "pose_estimator": model_info(pose_model_path),
                "tracker": {"name": "OC-SORT", "implementation": "persondet.tracker"},
            },
            "config": asdict(clip_config),
            "stats": {
                "videos_processed": len(video_files),
                "clips_extracted": total_clips,
                "total_duration_seconds": round(total_duration, 2),
            },
        },
    )


if __name__ == "__main__":
    main()
