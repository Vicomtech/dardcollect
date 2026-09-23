"""
person_clips.py — process_video logic for extracting person clips from a video.

Moved from pipeline/extract_person_clips_from_videos.py so that it can be
imported by other modules without pulling in the full script.
"""

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.clip_extraction import ClipBatchContext, extract_clips
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig
from dardcollect.extraction_logger import ClipRecord, ExtractionLogger
from dardcollect.face_geometry import _annotate_face_crop_corners
from dardcollect.fair import reorganize_for_fair
from dardcollect.frame_reader import frame_iter
from dardcollect.person_clips_helpers import (
    apply_duration_split,
    apply_scene_change,
    build_frame_data,
    compute_face_flags,
    filter_detections,
    is_scene_change,
    load_resume_start,
    save_progress,
    should_progressive_flush,
)
from dardcollect.pipeline_utils import (
    check_disk_space,
    make_tqdm,
)
from dardcollect.tracker import (
    Segment,
    TrackingParams,
    merge_segments,
    smooth_segment_keypoints,
    suppress_by_keypoints,
)
from dardcollect.video_writers import save_clip_sidecar_json

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


def _preload_source_local(video_path: Path, clip_config: ClipExtractionConfig) -> Path:
    """Copy a source video to a local cache dir so cv2 + moviepy read from local SSD.

    Network-share sources starve the GPU: ``cv2.VideoCapture.read()`` pulls frames one at
    a time over the network, and ``extract_clip`` re-reads the whole source once per emitted
    clip. Pre-copying once collapses all of that to a single network read + local reads.

    The cache dir MUST be local and outside ``input_dir``. Copy failure (disk full,
    permission, source unreachable) raises — no silent fallback to network read.

    The caller deletes the copy after processing (one file per source, sequential).
    """
    cache_dir = (
        Path(clip_config.local_cache_dir)
        if clip_config.local_cache_dir
        else Path(tempfile.gettempdir())
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space(cache_dir, clip_config.min_free_disk_gb)
    dst = cache_dir / video_path.name
    logger.info("Pre-copying source to local cache: %s -> %s", video_path.name, dst)
    try:
        shutil.copy2(video_path, dst)
    except Exception:
        # Remove any partially-written copy and re-raise (fail loud, no fallback).
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dst


def _remove_local_copy(local_copy: Path | None, clip_config: ClipExtractionConfig) -> None:
    """Delete the local cache copy and remove configured cache dir if empty."""
    if local_copy is not None and local_copy.exists():
        try:
            local_copy.unlink()
        except OSError as exc:
            logger.warning("Could not remove local cache copy %s: %s", local_copy, exc)
            return

    # Only prune explicit local_cache_dir from config (never system temp dir).
    if not clip_config.local_cache_dir:
        return

    cache_dir = Path(clip_config.local_cache_dir)
    try:
        if cache_dir.exists() and cache_dir.is_dir() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
            logger.info("Removed empty local cache dir: %s", cache_dir)
    except OSError:
        # Best-effort cleanup: ignore races/permissions and keep processing.
        pass


def _resolve_source_path(
    video_path: Path, clip_config: ClipExtractionConfig
) -> tuple[Path, Path | None]:
    """Return (read_path, local_copy_or_None) for a source video.

    When ``preload_source_to_local`` is set, copies the source to a local cache and returns
    that copy as the read path (cv2 + moviepy read local SSD). Otherwise returns the
    original path and ``None``. Provenance always uses the original ``video_path``.
    """
    if not clip_config.preload_source_to_local:
        return video_path, None
    local_copy = _preload_source_local(video_path, clip_config)
    return local_copy, local_copy


def _filter_candidate_segments(
    merged: list[Segment],
    clip_config: ClipExtractionConfig,
    fps: float,
    poser: PoseEstimator | None,
) -> list[Segment]:
    """Apply the duration and (opt-in) face-visibility filters to merged segments."""
    filtered = [
        s
        for s in merged
        if s.frame_count >= clip_config.min_consecutive_frames
        and s.duration_seconds(fps) >= clip_config.min_clip_duration_seconds
    ]
    if clip_config.require_face_visibility and poser is not None:
        filtered = [
            s
            for s in filtered
            if s.face_visible_frames >= clip_config.min_face_visible_frames
            and s.max_consecutive_face_frames >= clip_config.min_consecutive_face_frames
        ]
    return filtered


def _read_source_identity(video_path: Path) -> tuple[str | None, str | None]:
    """Read (archive_org_id, archive_org_url) from the source sidecar, if present.

    The sidecar is the same for all clips of a source, so this is read once per
    flush rather than per clip.
    """
    try:
        sidecar = video_path.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar, encoding="utf-8") as f:
                sidecar_data = json.load(f)
            return sidecar_data.get("identifier"), sidecar_data.get("url")
    except Exception as exc:
        logger.warning("Failed to read source sidecar %s: %s", sidecar.name, exc)
    return None, None


def _avg_detection_confidence(seg: Segment) -> float:
    """Mean detection score across the segment's frame_data (0.5 when none)."""
    all_scores = [
        d.get("score", 0.5)
        for detections in seg.frame_data.values()
        for d in detections
        if isinstance(d, dict)
    ]
    return sum(all_scores) / len(all_scores) if all_scores else 0.5


def _write_clip_result(
    r: dict,
    video_path: Path,
    fps: float,
    clip_logger: ExtractionLogger | None,
) -> dict:
    """Write one extracted clip's sidecar + CSV row (serialized, segment order).

    Returns the clip metadata dict (with the schema-consistency ``transcription``
    field always present).
    """
    seg: Segment = r["seg"]
    clip_path: Path = r["clip_path"]
    meta: dict = r["meta"]

    # Transcription is handled by transcribe_video_clips.py; keep field for schema consistency
    meta["transcription"] = ""

    if not r["success"]:
        return meta

    meta = reorganize_for_fair(meta)
    save_clip_sidecar_json(clip_path, meta)

    if clip_logger is not None:
        clip_logger.log_extraction(
            ClipRecord(
                source_video=video_path.name,
                fps=fps,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                start_seconds=r["start_sec"],
                duration_seconds=seg.duration_seconds(fps),
                max_persons_per_frame=seg.max_persons,
                detector_model="yolox-tiny",
                detector_confidence=_avg_detection_confidence(seg),
                output_path=str(clip_path),
            )
        )
    return meta


def flush_segments(
    segments_to_flush: list[Segment],
    *,
    fps: float,
    clip_config: ClipExtractionConfig,
    poser: PoseEstimator | None,
    face_crop_cfg: FaceCropConfig | None,
    video_path: Path,
    output_dir: Path,
    video_info: dict,
    clip_logger: ExtractionLogger | None,
    source_path: Path | None = None,
    encoding: "EncodingConfig | None" = None,
) -> list[dict]:
    """Merge, filter, split, smooth, and write a batch of candidate segments.

    The pipeline: merge adjacent segments → apply duration/face-visibility filters →
    split over-long segments → smooth keypoints per track → write clip videos and
    JSON sidecars. Returns clip metadata dicts for all successfully extracted clips.
    """
    if not segments_to_flush:
        return []
    # Read path for extract_clip: the local pre-copy if preloaded, else the original.
    # video_path stays the provenance/clip-name source (unchanged).
    read_path = source_path if source_path is not None else video_path

    # Merge compatible segments within this batch
    merged = merge_segments(segments_to_flush, clip_config.merge_gap_frames)

    filtered = _filter_candidate_segments(merged, clip_config, fps, poser)

    # Handle max duration splitting
    filtered = apply_duration_split(filtered, fps, clip_config.max_clip_duration_seconds)

    for seg in filtered:
        smooth_segment_keypoints(seg, fps)

    if face_crop_cfg is not None:
        for seg in filtered:
            _annotate_face_crop_corners(seg, face_crop_cfg)

    archive_org_id, archive_org_url = _read_source_identity(video_path)

    check_disk_space(output_dir, clip_config.min_free_disk_gb)

    # Extract all clips (parallel when configured + >1 clip; results in segment order).
    results = extract_clips(
        filtered,
        ClipBatchContext(
            read_path=read_path,
            output_dir=output_dir,
            fps=fps,
            video_path=video_path,
            video_info=video_info,
            archive_org_id=archive_org_id,
            archive_org_url=archive_org_url,
            encoding=encoding,
        ),
        clip_config,
    )

    # Serialize the sidecar write + CSV log in segment order (thread-safe + deterministic,
    # identical to the old serial path). The heavy extraction already ran above.
    return [_write_clip_result(r, video_path, fps, clip_logger) for r in results]


@dataclass
class _TrackerContext:
    """Per-video tracking objects, bundled so per-frame helpers stay low-arity."""

    tracker: PersonTracker
    det_config: DetectorConfig
    track_params: TrackingParams
    poser: PoseEstimator | None
    clip_config: ClipExtractionConfig
    frame_height: int


@dataclass
class _FrameObservation:
    """One frame's tracked people and their face/mouth flags."""

    tracklets: list
    tracklets_kpts: list[tuple]
    face_visible: bool
    mouth_open: bool
    face_streak: int = 0


def _observe_frame(
    frame: np.ndarray,
    det_bboxes: np.ndarray,
    det_scores: np.ndarray,
    ctx: _TrackerContext,
) -> _FrameObservation:
    """Track and flag one frame's people from already-filtered detections."""
    tracklets = ctx.tracker.update(det_bboxes.tolist(), det_scores.tolist(), ctx.track_params)
    if not tracklets:
        return _FrameObservation([], [], face_visible=False, mouth_open=False)
    if ctx.poser is not None:
        tracklets_kpts = [(t, *ctx.poser.get_keypoints(frame, t.tlbr.tolist())) for t in tracklets]
        tracklets_kpts = suppress_by_keypoints(
            tracklets_kpts,
            dist_threshold=0.15,
            score_threshold=ctx.det_config.pose_keypoint_threshold,
        )
        tracklets = [t for t, _, _ in tracklets_kpts]
    else:
        tracklets_kpts = [(t, None, None) for t in tracklets]
    face_visible, mouth_open = compute_face_flags(
        tracklets_kpts, ctx.frame_height, ctx.clip_config, ctx.det_config, ctx.poser
    )
    return _FrameObservation(tracklets, tracklets_kpts, face_visible, mouth_open)


def _accumulate_segment(
    curr_segment: Segment | None,
    frame_id: int,
    obs: _FrameObservation,
) -> Segment:
    """Extend *curr_segment* with this frame's observation, or start a new one."""
    current_frame_data = build_frame_data(obs.tracklets_kpts)
    track_ids = [t.track_id for t in obs.tracklets]
    if curr_segment is None:
        new_seg = Segment(
            start_frame=frame_id,
            end_frame=frame_id,
            track_ids=track_ids,
            max_persons=len(obs.tracklets),
            face_visible_frames=1 if obs.face_visible else 0,
            max_consecutive_face_frames=obs.face_streak if obs.face_visible else 0,
            mouth_open_frames=1 if obs.mouth_open else 0,
        )
        if current_frame_data:
            new_seg.frame_data[frame_id] = current_frame_data
        return new_seg

    curr_segment.end_frame = frame_id
    curr_segment.track_ids = list(set(curr_segment.track_ids + track_ids))
    curr_segment.max_persons = max(curr_segment.max_persons, len(obs.tracklets))
    if obs.face_visible:
        curr_segment.face_visible_frames += 1
        curr_segment.max_consecutive_face_frames = max(
            curr_segment.max_consecutive_face_frames, obs.face_streak
        )
    if obs.mouth_open:
        curr_segment.mouth_open_frames += 1
    if current_frame_data:
        curr_segment.frame_data[frame_id] = current_frame_data
    return curr_segment


def process_video(
    video_path: Path,
    detector: PersonDetector,
    tracker: PersonTracker,
    det_config: DetectorConfig,
    clip_config: ClipExtractionConfig,
    input_dir: Path | None = None,
    poser: PoseEstimator | None = None,
    face_crop_cfg: FaceCropConfig | None = None,
    clip_logger: ExtractionLogger | None = None,
    encoding: "EncodingConfig | None" = None,
) -> list[dict]:
    """Run detection + tracking on a video and extract all qualifying person clips.

    Resumable via a progress JSON file ({video_stem}_progress.json in output_dir).
    Clips are flushed progressively every ~30s of video to bound memory use.
    Returns an empty list (clip metadata is written to disk and logged by clip_logger).
    """
    logger.info("Processing: %s", video_path.name)

    # Optionally pre-copy the source to a local cache so cv2 + moviepy read from local
    # SSD instead of frame-by-frame over a network share (GPU-starving I/O). Provenance
    # (source_video field, clip names, .done) still references the original video_path;
    # only the read path (cv2 + extract_clip) switches to source_path. The local copy is
    # removed on return (one file per source, sequential).
    source_path, local_copy = _resolve_source_path(video_path, clip_config)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        _remove_local_copy(local_copy, clip_config)
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

    # Default: input_dir is the parent of the video. Used to derive the
    # source-subdirectory prefix embedded in clip filenames so the origin of
    # each clip is recoverable from its name without per-subdir output trees.
    if input_dir is None:
        input_dir = video_path.parent

    output_dir = Path(clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_info = {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
    }

    pending_segments: list[Segment] = []  # completed segments awaiting flush
    curr_segment: Segment | None = None  # segment currently being accumulated
    current_face_streak: int = 0  # consecutive frames with a visible face

    track_params = TrackingParams(
        score_threshold=det_config.tracking_score_threshold,
        min_hits=det_config.tracking_min_hits,
        max_time_lost=det_config.tracking_max_time_lost,
    )
    tracker_ctx = _TrackerContext(
        tracker=tracker,
        det_config=det_config,
        track_params=track_params,
        poser=poser,
        clip_config=clip_config,
        frame_height=height,
    )

    progress_path = output_dir / f"{video_path.stem}_progress.json"
    start_frame = load_resume_start(progress_path, total_frames, cap)

    # All three flush sites share these keyword arguments; bind them once.
    flush = partial(
        flush_segments,
        fps=fps,
        clip_config=clip_config,
        poser=poser,
        face_crop_cfg=face_crop_cfg,
        video_path=video_path,
        output_dir=output_dir,
        video_info=video_info,
        clip_logger=clip_logger,
        source_path=source_path,
        encoding=encoding,
    )

    frames_since_flush = 0
    prev_frame: np.ndarray | None = None
    prev_det_bboxes: np.ndarray = np.empty((0, 4))  # for scene-change detection
    last_scene_change_frame: int = start_frame - 1  # cooldown tracker

    pbar = make_tqdm(
        total=total_frames,
        initial=start_frame,
        unit="fr",
        desc=video_path.name[:40],
        dynamic_ncols=True,
    )

    # Read-ahead decode (producer thread) or inline cap.read — yields (frame_id, frame).
    # The reader thread (if any) is joined on generator close (finally), so it never leaks.
    for frame_id, frame in frame_iter(cap, start_frame, clip_config):
        # Detect before updating tracker so both frames' bboxes are available
        # for scene-change detection before track state changes.
        det_bboxes, det_scores = detector.get_detections(frame, det_config.detection_threshold)

        # Drop bboxes covering too large a fraction of the frame (title cards,
        # overlays) or with extreme aspect ratio (furniture, animals, not persons).
        det_bboxes, det_scores = filter_detections(
            det_bboxes, det_scores, width, height, clip_config
        )

        if is_scene_change(
            clip_config,
            prev_frame,
            frame_id,
            last_scene_change_frame,
            prev_det_bboxes,
            det_bboxes,
            frame,
        ):
            last_scene_change_frame = frame_id
            curr_segment, pending_segments, frames_since_flush, current_face_streak = (
                apply_scene_change(
                    frame_id,
                    curr_segment,
                    pending_segments,
                    frames_since_flush,
                    current_face_streak,
                    tracker=tracker,
                    flush_func=flush,
                )
            )

        prev_frame = frame
        prev_det_bboxes = det_bboxes

        obs = _observe_frame(frame, det_bboxes, det_scores, tracker_ctx)

        if obs.tracklets:
            # current_face_streak is computed here (not in the helper) so the
            # observation stays a pure per-frame value.
            current_face_streak = current_face_streak + 1 if obs.face_visible else 0
            obs.face_streak = current_face_streak
            curr_segment = _accumulate_segment(curr_segment, frame_id, obs)
        else:
            # No people in this frame — reset face streak and close the segment.
            current_face_streak = 0
            if curr_segment is not None:
                pending_segments.append(curr_segment)
                curr_segment = None

        # frame_id is assigned by _frame_iter each iteration (read-ahead or inline).
        frames_since_flush += 1

        # Progressive flush: once pending segments are stable (large enough gap
        # that no further merge can affect them), write to avoid memory buildup.
        if should_progressive_flush(
            pending_segments, frame_id, frames_since_flush, clip_config, fps
        ):
            flush(pending_segments)
            pending_segments = []
            frames_since_flush = 0
            save_progress(progress_path, frame_id, video_path)

        pbar.update(1)

    pbar.close()

    # Final flush for whatever is still in memory
    if curr_segment is not None:
        pending_segments.append(curr_segment)

    if pending_segments:
        flush(pending_segments)

    cap.release()
    _remove_local_copy(local_copy, clip_config)

    if progress_path.exists():
        try:
            progress_path.unlink()
        except OSError:
            pass

    return []
