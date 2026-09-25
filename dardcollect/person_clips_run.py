"""person_clips_run.py — per-video run state + frame loop for clip extraction.

Split from person_clips.py: ``process_video()`` concentrated source setup, the
whole frame loop, and teardown in one 185-line function (C901 12, 68
statements, 10 args). This module holds the mutable run state (``_VideoRun``)
and small phase helpers; ``person_clips.process_video()`` is the thin wiring
dispatcher taking a single ``VideoProcessRequest``. The flush/write path lives
in :mod:`dardcollect.person_clips_flush`.

The frame loop itself is the algorithm (detect → filter → scene-cut →
track/observe → accumulate → progressive flush), so ``_step_frame`` keeps the
per-frame order exactly as before — only the state it threads through is now
an explicit object instead of a dozen locals.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.person_clips_flush import _flush_batch, _FlushBatch
from dardcollect.person_clips_helpers import (
    SceneView,
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
    suppress_by_keypoints,
)

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessRequest:
    """Everything ``process_video`` needs for one source film (single argument)."""

    video_path: Path
    detector: PersonDetector
    tracker: PersonTracker
    det_config: DetectorConfig
    clip_config: ClipExtractionConfig
    poser: PoseEstimator | None = None
    face_crop_cfg: FaceCropConfig | None = None
    clip_logger: ExtractionLogger | None = None
    encoding: "EncodingConfig | None" = None


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


@dataclass
class _VideoRun:
    """Mutable per-video run state threaded through the frame loop."""

    req: VideoProcessRequest
    cap: cv2.VideoCapture
    fps: float
    total_frames: int
    width: int
    height: int
    output_dir: Path
    video_info: dict
    tracker_ctx: _TrackerContext
    source_path: Path
    local_copy: Path | None
    progress_path: Path
    pbar: tqdm
    start_frame: int
    curr_segment: Segment | None = None
    pending_segments: list[Segment] = field(default_factory=list)
    frames_since_flush: int = 0
    current_face_streak: int = 0
    prev_frame: np.ndarray | None = None
    prev_det_bboxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    last_scene_change_frame: int = 0


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


def _flush_run_segments(run: _VideoRun, segments: list[Segment]) -> list[dict]:
    """Flush *segments* with this run's bound context (same kwargs at every site)."""
    req = run.req
    return _flush_batch(
        _FlushBatch(
            segments=segments,
            fps=run.fps,
            clip_config=req.clip_config,
            poser=req.poser,
            face_crop_cfg=req.face_crop_cfg,
            video_path=req.video_path,
            output_dir=run.output_dir,
            video_info=run.video_info,
            clip_logger=req.clip_logger,
            source_path=run.source_path,
            encoding=req.encoding,
        )
    )


def _apply_scene_cut(run: _VideoRun, frame_id: int) -> None:
    """Flush + reset on a scene cut: move the current segment to pending, flush
    all pending segments, reset the face streak + tracker."""
    logger.debug("  Scene change at frame %d — flushing and resetting tracker", frame_id)
    if run.curr_segment is not None:
        run.pending_segments.append(run.curr_segment)
        run.curr_segment = None
    # Flush before processing the new scene so merge_segments() never joins
    # segments from opposite sides of the cut.
    if run.pending_segments:
        _flush_run_segments(run, run.pending_segments)
        run.pending_segments = []
        run.frames_since_flush = 0
    run.current_face_streak = 0
    run.tracker_ctx.tracker.init_tracker()


def _open_run(req: VideoProcessRequest) -> _VideoRun | None:
    """Resolve the source, open the capture, and build the run state.

    Returns None when the video cannot be opened (caller returns []).
    """
    # Optionally pre-copy the source to a local cache so cv2 + moviepy read from local
    # SSD instead of frame-by-frame over a network share (GPU-starving I/O). Provenance
    # (source_video field, clip names, .done) still references the original video_path;
    # only the read path (cv2 + extract_clip) switches to source_path. The local copy is
    # removed on return (one file per source, sequential).
    source_path, local_copy = _resolve_source_path(req.video_path, req.clip_config)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", req.video_path)
        _remove_local_copy(local_copy, req.clip_config)
        return None

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

    req.tracker.init_tracker()

    output_dir = Path(req.clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_info = {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
    }

    tracker_ctx = _TrackerContext(
        tracker=req.tracker,
        det_config=req.det_config,
        track_params=TrackingParams(
            score_threshold=req.det_config.tracking_score_threshold,
            min_hits=req.det_config.tracking_min_hits,
            max_time_lost=req.det_config.tracking_max_time_lost,
        ),
        poser=req.poser,
        clip_config=req.clip_config,
        frame_height=height,
    )

    progress_path = output_dir / f"{req.video_path.stem}_progress.json"
    start_frame = load_resume_start(progress_path, total_frames, cap)

    pbar = make_tqdm(
        total=total_frames,
        initial=start_frame,
        unit="fr",
        desc=req.video_path.name[:40],
        dynamic_ncols=True,
    )
    return _VideoRun(
        req=req,
        cap=cap,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        output_dir=output_dir,
        video_info=video_info,
        tracker_ctx=tracker_ctx,
        source_path=source_path,
        local_copy=local_copy,
        progress_path=progress_path,
        pbar=pbar,
        start_frame=start_frame,
        last_scene_change_frame=start_frame - 1,
    )


def _step_frame(run: _VideoRun, frame_id: int, frame: np.ndarray) -> None:
    """Run one frame through detect → scene-cut → observe → accumulate → flush."""
    req = run.req
    # Detect before updating tracker so both frames' bboxes are available
    # for scene-change detection before track state changes.
    det_bboxes, det_scores = req.detector.get_detections(frame, req.det_config.detection_threshold)

    # Drop bboxes covering too large a fraction of the frame (title cards,
    # overlays) or with extreme aspect ratio (furniture, animals, not persons).
    det_bboxes, det_scores = filter_detections(
        det_bboxes, det_scores, run.width, run.height, req.clip_config
    )

    if is_scene_change(
        SceneView(
            clip_config=req.clip_config,
            prev_frame=run.prev_frame,
            frame_id=frame_id,
            last_scene_change_frame=run.last_scene_change_frame,
            prev_det_bboxes=run.prev_det_bboxes,
            det_bboxes=det_bboxes,
            frame=frame,
        )
    ):
        run.last_scene_change_frame = frame_id
        _apply_scene_cut(run, frame_id)

    run.prev_frame = frame
    run.prev_det_bboxes = det_bboxes

    obs = _observe_frame(frame, det_bboxes, det_scores, run.tracker_ctx)

    if obs.tracklets:
        # current_face_streak is computed here (not in the helper) so the
        # observation stays a pure per-frame value.
        run.current_face_streak = run.current_face_streak + 1 if obs.face_visible else 0
        obs.face_streak = run.current_face_streak
        run.curr_segment = _accumulate_segment(run.curr_segment, frame_id, obs)
    else:
        # No people in this frame — reset face streak and close the segment.
        run.current_face_streak = 0
        if run.curr_segment is not None:
            run.pending_segments.append(run.curr_segment)
            run.curr_segment = None

    # frame_id is assigned by _frame_iter each iteration (read-ahead or inline).
    run.frames_since_flush += 1

    # Progressive flush: once pending segments are stable (large enough gap
    # that no further merge can affect them), write to avoid memory buildup.
    if should_progressive_flush(
        run.pending_segments, frame_id, run.frames_since_flush, req.clip_config, run.fps
    ):
        _flush_run_segments(run, run.pending_segments)
        run.pending_segments = []
        run.frames_since_flush = 0
        save_progress(run.progress_path, frame_id, req.video_path)

    run.pbar.update(1)


def _drain_run(run: _VideoRun) -> None:
    """Final flush for whatever is still in memory, then release + clean up."""
    if run.curr_segment is not None:
        run.pending_segments.append(run.curr_segment)

    if run.pending_segments:
        _flush_run_segments(run, run.pending_segments)

    run.cap.release()
    _remove_local_copy(run.local_copy, run.req.clip_config)

    if run.progress_path.exists():
        try:
            run.progress_path.unlink()
        except OSError:
            pass
