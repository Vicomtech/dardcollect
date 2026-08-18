"""Frame extraction from the ORIGINAL source videos, at person-detected timestamps.

Where :mod:`dardcollect.frames` explodes a clip into frames, this module goes the other
way: it reads a person clip's sidecar, uses the detections already computed for it, and
pulls a short run of frames out of the *source* video the clip was cut from.

Two properties of the clip sidecar make that possible without any re-inference:

* ``source_video`` — absolute path to the original download;
* ``frame_data`` keyed by **absolute source-video frame number** (a clip cut at 28m56s of
  a 25 fps film starts at key ``"43411"``), each entry carrying the person ``bbox`` and
  the 133 wholebody keypoints.

So the clip is a *timestamp index into the original*: the detections say exactly which
frames of the source contain a person, and picking N consecutive ones costs a seek.

Output is keyed by source video and absolute frame — ``<video_stem>/frame_043411.png`` —
so two clips whose selected windows overlap converge on the same file instead of writing
it twice. That also makes the pass idempotent across reruns and across clip boundaries.

See docs/DESIGN_video_frame_masks.md.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2

from dardcollect.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from dardcollect.pipeline_loggers import FramesExtractionLogger
from dardcollect.pipeline_utils import FACE_LANDMARK_INDICES

logger = logging.getLogger(__name__)

_KEYPOINT_COUNT = 133
_KPT_SCORE_THRESHOLD = 0.3  # must match generate_face_masks._KPT_SCORE_THRESHOLD


def _has_usable_face(detections: Any) -> bool:
    """Whether any detection has face landmarks confident enough to mask.

    Deliberately stricter than :func:`dardcollect.frames._frame_has_face`, which only
    checks that a ``keypoints`` list exists. Frame *selection* has to apply the same
    test the mask stage applies (``_detections_with_faces``), or we would select frames
    on the strength of zero-confidence landmarks and then emit no mask for them.
    """
    if not isinstance(detections, list):
        return False
    for det in detections:
        if not isinstance(det, Mapping):
            continue
        det_map = cast(Mapping[str, Any], det)
        scores = det_map.get("keypoint_scores")
        kpts = det_map.get("keypoints")
        if not (isinstance(kpts, list) and len(kpts) == _KEYPOINT_COUNT):
            continue
        if not (isinstance(scores, list) and len(scores) == _KEYPOINT_COUNT):
            continue
        if any(float(scores[i]) >= _KPT_SCORE_THRESHOLD for i in FACE_LANDMARK_INDICES):
            return True
    return False


def select_consecutive_detected_frames(
    frame_data: Mapping[str, Any],
    frames_per_clip: int,
) -> list[int]:
    """Pick the first run of *frames_per_clip* consecutive frames that all show a face.

    Keys of *frame_data* are absolute source-video frame numbers as strings. Only runs
    where **every** frame has a usable face annotation qualify, so the returned window is
    contiguous in the source video rather than a set of scattered hits — which is what
    "N consecutive frames" asks for.

    Returns the absolute frame numbers, or an empty list when no such run exists (a clip
    can track a person from behind for its whole span and never yield usable landmarks).
    """
    if frames_per_clip <= 0 or not isinstance(frame_data, dict):
        return []

    numbered = sorted(
        (int(key), value) for key, value in frame_data.items() if str(key).lstrip("-").isdigit()
    )
    run: list[int] = []
    for absolute_frame, detections in numbered:
        if run and absolute_frame != run[-1] + 1:
            run = []
        if not _has_usable_face(detections):
            run = []
            continue
        run.append(absolute_frame)
        if len(run) == frames_per_clip:
            return run
    return []


def _read_source_frame(capture: cv2.VideoCapture, absolute_frame: int):
    """Seek to *absolute_frame* and return it, or None if unreadable."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
    ok, frame = capture.read()
    return frame if ok else None


def _write_frame_sidecar(
    frame_json: Path,
    absolute_frame: int,
    detections: Any,
    fps: float,
    clip_uuid: str | None,
    clip_name: str,
    source_video: Path,
) -> str | None:
    """Write one frame's FAIR sidecar. Returns its UUID, or None on failure."""
    frame_uuid = generate_uuid()
    meta: dict[str, Any] = {
        "frame_number": absolute_frame,
        "timestamp": absolute_frame / fps if fps > 0 else 0.0,
        "detections": detections,
        "source_video": str(source_video),
    }
    meta = add_fair_metadata(
        meta, schema_type="person_clip", parent_uuid=clip_uuid, parent_file=clip_name
    )
    meta["uuid"] = frame_uuid
    meta = reorganize_for_fair(meta, "person_clip")

    try:
        with open(frame_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        logger.error("Cannot write frame JSON %s: %s", frame_json.name, e)
        return None
    return frame_uuid


def extract_source_frames_for_clip(
    clip_sidecar: Path,
    output_root: Path,
    frames_per_clip: int,
    overwrite: bool = False,
    frames_logger: FramesExtractionLogger | None = None,
) -> tuple[int, bool]:
    """Extract N consecutive source-video frames for one person clip.

    Args:
        clip_sidecar: Path to a person clip's ``.json`` sidecar.
        output_root: Root of the frames tree; frames land in
            ``<output_root>/<language>/<source video stem>/``.
        frames_per_clip: How many consecutive detected frames to pull (the request's N).
        overwrite: Re-extract frames that already have both ``.png`` and ``.json``.
        frames_logger: Optional CSV provenance logger.

    Returns:
        ``(frames_written, had_qualifying_run)``. Both are needed to report honestly:
        zero frames written means "already extracted" when a run qualified, and "no
        usable run" when it did not, and conflating the two makes a resumed pass look
        like a total failure.
    """
    try:
        with open(clip_sidecar, encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception as e:
        logger.error("Cannot read clip sidecar %s: %s", clip_sidecar.name, e)
        return 0, False

    source_raw = sidecar.get("source_video")
    if not source_raw:
        logger.warning("Clip %s has no source_video, skipping", clip_sidecar.name)
        return 0, False
    source_video = Path(source_raw)
    if not source_video.exists():
        logger.warning("Source video missing for %s: %s", clip_sidecar.name, source_video)
        return 0, False

    frame_data = cast(Mapping[str, Any], sidecar.get("frame_data") or {})
    selected = select_consecutive_detected_frames(frame_data, frames_per_clip)
    if not selected:
        logger.debug("No run of %d detected frames in %s", frames_per_clip, clip_sidecar.name)
        return 0, False

    # Group by source video, not by clip: overlapping clips share these files.
    output_dir = output_root / source_video.parent.name / source_video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for absolute_frame in selected:
        frame_png = output_dir / f"frame_{absolute_frame:06d}.png"
        frame_json = output_dir / f"frame_{absolute_frame:06d}.json"
        if frame_png.exists() and frame_json.exists() and not overwrite:
            continue
        pending.append((absolute_frame, frame_png, frame_json))

    if not pending:
        return 0, True  # every selected frame already on disk (resumed pass)

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        logger.error("Cannot open source video: %s", source_video)
        return 0, True

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    clip_uuid = sidecar.get("uuid")
    written = 0
    try:
        for absolute_frame, frame_png, frame_json in pending:
            frame = _read_source_frame(capture, absolute_frame)
            if frame is None:
                logger.warning("Cannot read frame %d of %s", absolute_frame, source_video.name)
                continue

            try:
                cv2.imwrite(str(frame_png), frame)
            except Exception as e:
                logger.error("Cannot write frame PNG %s: %s", frame_png.name, e)
                continue

            frame_uuid = _write_frame_sidecar(
                frame_json,
                absolute_frame,
                frame_data.get(str(absolute_frame), []),
                fps,
                clip_uuid,
                clip_sidecar.name,
                source_video,
            )
            if frame_uuid is None:
                frame_png.unlink(missing_ok=True)
                continue

            if frames_logger is not None:
                frames_logger.log_frame_extraction(
                    source_clip_path=str(clip_sidecar.with_suffix(".mp4")),
                    frame_number=absolute_frame,
                    timestamp_seconds=absolute_frame / fps if fps > 0 else 0.0,
                    output_path=str(frame_png),
                )
            written += 1
    finally:
        capture.release()

    return written, True
