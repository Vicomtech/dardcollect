"""
person_clips.py — process_video logic for extracting person clips from a video.

Moved from pipeline/extract_person_clips_from_videos.py so that it can be
imported by other modules without pulling in the full script.
"""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.face_geometry import _annotate_face_crop_corners
from dardcollect.fair import add_fair_metadata, reorganize_for_fair
from dardcollect.person_clips_helpers import (
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
    extract_clip,
    make_tqdm,
    save_clip_sidecar_json,
)
from dardcollect.tracker import (
    Segment,
    TrackingParams,
    merge_segments,
    smooth_segment_keypoints,
    suppress_by_keypoints,
)

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


def _remove_local_copy(local_copy: Path | None) -> None:
    """Delete the local cache copy after a source is processed (best-effort)."""
    if local_copy is not None and local_copy.exists():
        try:
            local_copy.unlink()
        except OSError as exc:
            logger.warning("Could not remove local cache copy %s: %s", local_copy, exc)


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


def flush_segments(
    segments_to_flush: list[Segment],
    *,
    fps: float,
    clip_config: ClipExtractionConfig,
    poser: PoseEstimator | None,
    face_crop_cfg: FaceCropConfig | None,
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    video_info: dict,
    clip_logger: ExtractionLogger | None,
    source_path: Path | None = None,
    force: bool = False,
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
                        # face_visible is not stored per-frame; approximate by
                        # whether any detection has keypoints.
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
        smooth_segment_keypoints(seg, fps)

    if face_crop_cfg is not None:
        for seg in filtered:
            _annotate_face_crop_corners(seg, face_crop_cfg)

    # Convert to dicts and store/save
    batch_clip_metas = []

    for seg in filtered:
        # Clip extraction
        start_sec = seg.start_frame / fps
        end_sec = seg.end_frame / fps
        start_str = f"{int(start_sec // 60):02d}m{int(start_sec % 60):02d}s"
        end_str = f"{int(end_sec // 60):02d}m{int(end_sec % 60):02d}s"
        # output_dir is the per-source-video dir (caller's responsibility to
        # scope it). Clip name derives from the source video stem, which is
        # unique across the input_dir subtree (timestamps + random hashes).
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

        archive_org_id = None
        archive_org_url = None
        try:
            sidecar = video_path.with_suffix(".json")
            if sidecar.exists():
                with open(sidecar, encoding="utf-8") as f:
                    sidecar_data = json.load(f)
                archive_org_id = sidecar_data.get("identifier")
                archive_org_url = sidecar_data.get("url")
        except Exception as exc:
            logger.warning("Failed to read source sidecar %s: %s", sidecar.name, exc)

        meta = add_fair_metadata(
            meta,
            schema_type="person_clip",
            archive_org_id=archive_org_id,
            archive_org_url=archive_org_url,
        )

        extraction_success = False
        check_disk_space(output_dir, clip_config.min_free_disk_gb)
        logger.info("  Extracting: %s (%.1fs)", clip_name, meta["duration_seconds"])
        t0 = time.time()
        if extract_clip(read_path, clip_path, seg.start_frame, seg.end_frame, fps):
            t_extract = time.time() - t0
            logger.info("  Extraction took %.2fs", t_extract)
            meta["clip_path"] = clip_path.as_posix()
            extraction_success = True
        else:
            meta["error"] = "Extraction failed"

        # Transcription is handled by transcribe_video_clips.py; keep field for schema consistency
        meta["transcription"] = ""

        if extraction_success:
            meta = reorganize_for_fair(meta, "person_clip")
            save_clip_sidecar_json(clip_path, meta)

            if clip_logger is not None:
                all_scores = [
                    d.get("score", 0.5)
                    for detections in seg.frame_data.values()
                    for d in detections
                    if isinstance(d, dict)
                ]
                avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.5

                clip_logger.log_extraction(
                    source_video=video_path.name,
                    fps=fps,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    start_seconds=start_sec,
                    duration_seconds=seg.duration_seconds(fps),
                    max_persons_per_frame=seg.max_persons,
                    detector_model="yolox-tiny",
                    detector_confidence=avg_confidence,
                    output_path=str(clip_path),
                )

        batch_clip_metas.append(meta)

    return batch_clip_metas


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
        _remove_local_copy(local_copy)
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

    progress_path = output_dir / f"{video_path.stem}_progress.json"
    start_frame = load_resume_start(progress_path, total_frames, cap)

    frame_id = start_frame
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                    fps=fps,
                    clip_config=clip_config,
                    poser=poser,
                    face_crop_cfg=face_crop_cfg,
                    video_path=video_path,
                    output_dir=output_dir,
                    input_dir=input_dir,
                    video_info=video_info,
                    clip_logger=clip_logger,
                    tracker=tracker,
                    flush_func=flush_segments,
                    source_path=source_path,
                )
            )

        prev_frame = frame
        prev_det_bboxes = det_bboxes

        tracklets = tracker.update(det_bboxes.tolist(), det_scores.tolist(), track_params)

        # Compute keypoints once per tracklet to share results across
        # duplicate suppression, face-visibility checks, and frame data.
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

        if tracklets:
            track_ids = [t.track_id for t in tracklets]
            # Iterate all tracks (not just the first) so mouth_open captures
            # any speaking person in a multi-person frame.
            face_visible, mouth_open = compute_face_flags(
                tracklets_kpts, height, clip_config, det_config, poser
            )
            current_frame_data = build_frame_data(tracklets_kpts)

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

        # Progressive flush: once pending segments are stable (large enough gap
        # that no further merge can affect them), write to avoid memory buildup.
        if should_progressive_flush(
            pending_segments, frame_id, frames_since_flush, clip_config, fps
        ):
            flush_segments(
                pending_segments,
                fps=fps,
                clip_config=clip_config,
                poser=poser,
                face_crop_cfg=face_crop_cfg,
                video_path=video_path,
                input_dir=input_dir,
                output_dir=output_dir,
                video_info=video_info,
                clip_logger=clip_logger,
                source_path=source_path,
            )
            pending_segments = []
            frames_since_flush = 0
            save_progress(progress_path, frame_id, video_path)

        pbar.update(1)

    pbar.close()

    # Final flush for whatever is still in memory
    if curr_segment is not None:
        pending_segments.append(curr_segment)

    if pending_segments:
        flush_segments(
            pending_segments,
            fps=fps,
            clip_config=clip_config,
            poser=poser,
            face_crop_cfg=face_crop_cfg,
            video_path=video_path,
            input_dir=input_dir,
            output_dir=output_dir,
            video_info=video_info,
            clip_logger=clip_logger,
            force=True,
            source_path=source_path,
        )

    cap.release()
    _remove_local_copy(local_copy)

    if progress_path.exists():
        try:
            progress_path.unlink()
        except OSError:
            pass

    return []
