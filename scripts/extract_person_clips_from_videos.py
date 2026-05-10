#!/usr/bin/env python3
"""
Extract video clips containing people from downloaded videos.

Uses person detection and tracking to identify segments where
people are visible, then extracts those clips as separate files.

All parameters are read from config.yaml.
"""

import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from dardcollect.fair import add_fair_metadata, reorganize_for_fair
from dardcollect.pipeline_utils import (
    _TqdmHandler,
    check_disk_space,
    check_face_visibility,
    check_frontal_face,
    extract_clip,
    save_clip_sidecar_json,
    scene_changed,
)

# Configure logging — route through tqdm so output doesn't break progress bars


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Setup paths BEFORE importing libraries that might load DLLs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import numpy as np

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig, get_log_level
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.face_geometry import _annotate_face_crop_corners
from dardcollect.tracker import (
    Segment,
    TrackingParams,
    merge_segments,
    smooth_segment_keypoints,
    suppress_by_keypoints,
)


def process_video(
    video_path: Path,
    detector: PersonDetector,
    tracker: PersonTracker,
    det_config: DetectorConfig,
    clip_config: ClipExtractionConfig,
    poser: PoseEstimator | None = None,
    face_crop_cfg: FaceCropConfig | None = None,
    clip_logger: ExtractionLogger | None = None,
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
            smooth_segment_keypoints(seg, fps)

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

            # Extract archive.org metadata if available
            archive_org_id = None
            archive_org_url = None
            try:
                sidecar = video_path.with_suffix(".json")
                if sidecar.exists():
                    with open(sidecar, encoding="utf-8") as f:
                        sidecar_data = json.load(f)
                    archive_org_id = sidecar_data.get("identifier")
                    archive_org_url = sidecar_data.get("url")
            except Exception:
                pass

            # Add FAIR metadata (UUID, schema version, source tracking)
            meta = add_fair_metadata(
                meta,
                schema_type="person_clip",
                archive_org_id=archive_org_id,
                archive_org_url=archive_org_url,
            )

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
                meta = reorganize_for_fair(meta, "person_clip")
                save_clip_sidecar_json(clip_path, meta)

                # Log extraction to CSV (incremental write)
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
    # Load configuration
    try:
        det_config = DetectorConfig.from_yaml(str(CONFIG_PATH))
        clip_config = ClipExtractionConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    face_crop_cfg: FaceCropConfig | None = None
    try:
        face_crop_cfg = FaceCropConfig.from_yaml(str(CONFIG_PATH))
        logger.info("Face crop config loaded — will annotate arcface + ofiq crop corners")
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
        video_files = list(input_path.rglob("*.mp4"))
        video_files.extend(input_path.rglob("*.avi"))
        video_files.extend(input_path.rglob("*.mkv"))
        video_files.extend(input_path.rglob("*.mov"))

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
    # run scripts/transcribe_video_clips.py instead

    # Process videos
    output_dir = Path(clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extraction logger (CSV audit trail)
    downloads_csv = Path(clip_config.input_dir).parent / "downloads.csv"
    clip_logger = ExtractionLogger(output_dir=str(output_dir), downloads_csv_path=downloads_csv)

    all_results = []
    for video_path in video_files:
        done_sentinel = output_dir / f"{video_path.stem}.done"
        if done_sentinel.exists():
            logger.info("SKIP (already done): %s", video_path.name)
            continue

        try:
            results = process_video(
                video_path,
                detector,
                tracker,
                det_config,
                clip_config,
                poser,
                face_crop_cfg,
                clip_logger=clip_logger,
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

    # Print extraction log summary
    clip_logger.print_summary()


if __name__ == "__main__":
    main()
