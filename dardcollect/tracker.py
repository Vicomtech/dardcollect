"""
Segment dataclass and segment manipulation utilities.

NMS and filtering primitives for person tracking segment consolidation.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

# Re-export public API from tracker_ocsort
from .tracker_ocsort import KalmanFilter, PersonTracker, TrackingParams, Tracklet, TrackState

__all__ = [
    "KalmanFilter",
    # Re-exports
    "PersonTracker",
    "Segment",
    "TrackState",
    "TrackingParams",
    "Tracklet",
    "merge_segments",
    "smooth_segment_keypoints",
    "suppress_by_keypoints",
    "suppress_overlapping_tracklets",
]


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

    Args:
        tracklets_kpts: List of (tracklet, keypoints ndarray, kpt_scores ndarray) tuples.
        dist_threshold: Normalised distance below which two detections are duplicates.
        score_threshold: Minimum keypoint confidence to include in comparison.

    Returns:
        Filtered list of the same form.
    """
    _log = logging.getLogger(__name__)
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
                _log.debug(
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


def smooth_segment_keypoints(
    seg: Segment,
    fps: float,
    window_seconds: float = 0.25,
    polyorder: int = 2,
) -> None:
    """Smooth keypoints in-place per track using a Savitzky-Golay filter.

    Since extraction is offline we can look at the full segment at once,
    so a polynomial filter beats a causal moving average: it preserves
    peaks and motion onsets while killing frame-to-frame jitter.
    """
    from scipy.signal import savgol_filter

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
