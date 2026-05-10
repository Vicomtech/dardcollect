#!/usr/bin/env python3
"""
Extract normalized face crop videos from person clip videos.

Reads the sidecar JSONs written by extract_person_clips_from_videos.py (which contain
SG-smoothed keypoints and pre-computed face crop corners) instead of
re-running detection, so face crops use the same smoothed keypoint positions
as the rest of the pipeline.

One crop format is produced per track:

  output_dir/ofiq/  — 616×616, OFIQ-aligned (BSI-OFIQ convention).
    Wider framing with eyes at y≈272, nose at y≈336, mouth at y≈402.
    Matches the format expected by all OFIQ quality measures: sharpness,
    expression neutrality, head pose, compression artifacts, background
    uniformity, and face occlusion.
    Input to filter_face_crops_by_quality.py and annotate_face_quality.py.

The sidecar JSON for each OFIQ video includes an 'arcface_crop_corners_in_ofiq'
field — the 4 corners (TL, TR, BR, BL) of the ArcFace 112×112 region in OFIQ
frame pixel coordinates.  Because both formats align to fixed canonical landmark
positions, this region is constant across all frames and all clips.  Downstream
scripts (filter_face_crops_by_quality.py, annotate_face_quality.py) use it to
extract 112×112 ArcFace crops from OFIQ frames for MagFace scoring.

All parameters are read from config.yaml under the 'face_crop_extraction' key.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from dardcollect.audio import _mux_audio
from dardcollect.fair import add_fair_metadata, reorganize_for_fair
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger
from dardcollect.pipeline_utils import _cleanup_files, _TqdmHandler, check_disk_space

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from dardcollect.config import FaceCropConfig, get_log_level
from dardcollect.face_geometry import (
    ARCFACE_CROP_CORNERS_IN_OFIQ,
    OFIQ_SIZE,
    _bbox_iou,
    _corners_to_warp,
    _get_or_compute_corners,
    _transform_bbox,
    _transform_keypoints,
)


def _is_track_complete(ofiq_path: Path) -> bool:
    """Check if the OFIQ crop for a track has been fully written."""
    return ofiq_path.exists() and ofiq_path.with_suffix(".json").exists()


# _mux_audio, check_disk_space, _cleanup_files imported from dardcollect


def _write_video_with_moviepy(
    frames: list[np.ndarray],
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
    if not frames:
        logger.error("No frames to write")
        return False

    try:
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
            logger.error("Output file is missing or empty")
            return False

        return True

    except Exception as e:
        logger.error("Error writing video with moviepy: %s", e)
        return False


# ── Per-video processing ──────────────────────────────────────────────────────


def process_video(
    video_path: Path,
    face_config: FaceCropConfig,
    face_crops_logger: FaceCropsExtractionLogger | None = None,
) -> int:
    """Extract OFIQ face crop videos from a single source clip using its sidecar JSON."""
    output_dir = Path(face_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_sentinel = output_dir / f"{video_path.stem}.done"
    if done_sentinel.exists():
        logger.info("  SKIP (already done): %s", video_path.name)
        return 0

    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        logger.error("  No sidecar JSON for %s — skipping", video_path.name)
        return 0

    with open(json_path, encoding="utf-8") as f:
        clip_data = json.load(f)

    start_frame: int = clip_data.get("start_frame", 0)
    frame_data_orig: dict = clip_data.get("frame_data", {})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "  %dx%d  %.1f fps  %d frames  (%.1fs)",
        width,
        height,
        fps,
        total_frames,
        total_frames / fps if fps > 0 else 0,
    )

    # track_id → list of (relative_frame_idx, ofiq_crop_or_None)
    track_frames: dict[int, list[tuple[int, np.ndarray | None]]] = defaultdict(list)

    frame_id = 0
    pbar = tqdm(total=total_frames, unit="fr", desc=video_path.name[:40], dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        abs_frame = start_frame + frame_id
        detections = frame_data_orig.get(str(abs_frame), [])

        frame_bboxes = [(d["track_id"], d["bbox"]) for d in detections]

        for det in detections:
            tid = det["track_id"]
            bbox = det["bbox"]

            # Compute or get corners from keypoints
            corners = _get_or_compute_corners(det, face_config)
            if corners is None:
                track_frames[tid].append((frame_id, None))
                continue

            overlapping = any(
                _bbox_iou(bbox, ob) > face_config.max_overlap_iou
                for oid, ob in frame_bboxes
                if oid != tid
            )
            if overlapping:
                track_frames[tid].append((frame_id, None))
                continue

            ofiq_crop = _corners_to_warp(frame, corners, OFIQ_SIZE)
            track_frames[tid].append((frame_id, ofiq_crop))

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write one video per track ─────────────────────────────────────────────
    written = 0
    black_ofiq = np.zeros((OFIQ_SIZE, OFIQ_SIZE, 3), dtype=np.uint8)

    arcface_corners_json = [
        [round(float(x), 2), round(float(y), 2)] for x, y in ARCFACE_CROP_CORNERS_IN_OFIQ
    ]

    for tid, frames in track_frames.items():
        valid_frames = [(fid, oc) for fid, oc in frames if oc is not None]
        if len(valid_frames) < face_config.min_track_face_frames:
            logger.debug(
                "  Track %d: only %d valid face frame(s), skipping",
                tid,
                len(valid_frames),
            )
            continue

        stem = f"{video_path.stem}_face_{tid}"
        ofiq_path = output_dir / f"{stem}.mp4"
        ofiq_sidecar = ofiq_path.with_suffix(".json")

        if _is_track_complete(ofiq_path):
            logger.info("  SKIP (already complete): %s", stem)
            continue

        if ofiq_path.exists():
            logger.info("  Incomplete write detected, cleaning up: %s", stem)
            _cleanup_files(ofiq_path, ofiq_sidecar)

        check_disk_space(output_dir, face_config.min_free_disk_gb)

        # Collect frames to write and extract keypoints
        frames_to_write = []
        # Map from output frame index to detections (same as person_clips)
        frame_data = {}
        if face_config.skip_no_face_frames:
            output_frame_idx = 0
            for fid, oc in valid_frames:
                if oc is not None:
                    frames_to_write.append(oc)
                    # Extract keypoints from original frame data
                    abs_frame = start_frame + fid
                    detections = frame_data_orig.get(str(abs_frame), [])
                    for det in detections:
                        if det.get("track_id") == tid:
                            kpts = det.get("keypoints", [])
                            scores = det.get("keypoint_scores", [])
                            corners = _get_or_compute_corners(det, face_config)
                            if corners is not None and kpts and scores:
                                kpts_array = np.array(kpts, dtype=np.float32)
                                scores_array = np.array(scores, dtype=np.float32)
                                transformed_kpts, _, M = _transform_keypoints(
                                    kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                                )
                                entry: dict = {
                                    "track_id": tid,
                                    "score": det.get("score"),
                                    "keypoints": transformed_kpts,
                                    "keypoint_scores": scores,
                                    "face_crop_corners_arcface": arcface_corners_json,
                                }
                                if M is not None and det.get("bbox"):
                                    entry["bbox"] = _transform_bbox(det["bbox"], M)
                                frame_data[str(output_frame_idx)] = [entry]
                            break
                    output_frame_idx += 1
        else:
            first_fid = frames[0][0]
            last_fid = frames[-1][0]
            ofiq_dict = {fid: oc for fid, oc in frames if oc is not None}
            last_ofiq = black_ofiq
            output_frame_idx = 0
            for fid in range(first_fid, last_fid + 1):
                if fid in ofiq_dict:
                    last_ofiq = ofiq_dict[fid]
                frames_to_write.append(last_ofiq)
                abs_frame = start_frame + fid
                detections = frame_data_orig.get(str(abs_frame), [])
                for det in detections:
                    if det.get("track_id") == tid:
                        kpts = det.get("keypoints", [])
                        scores = det.get("keypoint_scores", [])
                        corners = _get_or_compute_corners(det, face_config)
                        if corners is not None and kpts and scores:
                            kpts_array = np.array(kpts, dtype=np.float32)
                            scores_array = np.array(scores, dtype=np.float32)
                            transformed_kpts, _, M = _transform_keypoints(
                                kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                            )
                            entry = {
                                "track_id": tid,
                                "score": det.get("score"),
                                "keypoints": transformed_kpts,
                                "keypoint_scores": scores,
                                "face_crop_corners_arcface": arcface_corners_json,
                            }
                            if M is not None and det.get("bbox"):
                                entry["bbox"] = _transform_bbox(det["bbox"], M)
                            frame_data[str(output_frame_idx)] = [entry]
                        break
                output_frame_idx += 1

        # Write video using moviepy (same as extracted_person_clips)
        success = _write_video_with_moviepy(frames_to_write, ofiq_path, fps)

        if not success:
            logger.error("Failed to write video for %s — stopping.", stem)
            _cleanup_files(ofiq_path)
            sys.exit(1)

        first_fid, last_fid = frames[0][0], frames[-1][0]
        n_valid = len(valid_frames)
        n_output_frames = last_fid - first_fid + 1
        duration_seconds = round(n_output_frames / fps, 3) if fps > 0 else 0

        if face_config.include_audio and not face_config.skip_no_face_frames:
            start_t = first_fid / fps
            end_t = (last_fid + 1) / fps
            _mux_audio(video_path, ofiq_path, start_t, end_t)

        meta = {
            "source_video": str(video_path),
            "track_id": tid,
            "start_frame": 0,
            "end_frame": n_output_frames - 1,
            "start_seconds": 0.0,
            "end_seconds": duration_seconds,
            "duration_seconds": duration_seconds,
            "video_info": {
                "fps": round(fps, 3),
                "width": OFIQ_SIZE,
                "height": OFIQ_SIZE,
                "duration_seconds": duration_seconds,
            },
            "valid_face_frames": n_valid,
            "crop_format": "ofiq",
            "output_size": OFIQ_SIZE,
            "frame_data": frame_data,
        }

        # Add FAIR metadata (UUID, schema version, parent link)
        parent_clip_uuid = clip_data.get("uuid")
        parent_clip_file = video_path.name
        meta = add_fair_metadata(
            meta,
            schema_type="face_crop",
            parent_uuid=parent_clip_uuid,
            parent_file=parent_clip_file,
        )

        try:
            meta = reorganize_for_fair(meta, "face_crop")
            with open(ofiq_sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            logger.error("Cannot write %s (%s) — stopping.", ofiq_sidecar.name, e)
            _cleanup_files(ofiq_path, ofiq_sidecar)
            sys.exit(1)

        # Log face crop extraction (for traceability)
        if face_crops_logger is not None:
            # Get average detection confidence from frame_data
            avg_confidence = 0.5
            if frame_data and frame_data.get("0"):
                detections = frame_data.get("0", [])
                if detections and isinstance(detections[0], dict):
                    score = detections[0].get("score", 0.5)
                    avg_confidence = float(score) if score else 0.5

            face_crops_logger.log_face_crop_extraction(
                source_type="person_clip",
                source_path=str(video_path),
                face_bbox=f"0,0,{OFIQ_SIZE},{OFIQ_SIZE}",  # Full OFIQ frame
                confidence=avg_confidence,
                output_path=str(ofiq_path),
            )

        logger.info(
            "  Wrote %s  (%d valid face frames / %d span frames)",
            stem,
            n_valid,
            last_fid - first_fid + 1,
        )
        written += 1

    return written


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    try:
        face_config = FaceCropConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    input_path = Path(face_config.input_dir)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            video_files.extend(input_path.glob(ext))

    if not video_files:
        logger.error("No video files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d video(s) to process", len(video_files))

    output_dir = Path(face_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face crops logger
    clips_csv = Path(face_config.input_dir) / "clips_extraction.csv"
    face_crops_logger = FaceCropsExtractionLogger(
        output_dir=str(output_dir), clips_csv_path=clips_csv
    )

    total_written = 0
    for video_path in video_files:
        done_sentinel = output_dir / f"{video_path.stem}.done"
        if done_sentinel.exists():
            logger.info("SKIP (already done): %s", video_path.name)
            continue

        logger.info("Processing: %s", video_path.name)
        try:
            n = process_video(video_path, face_config, face_crops_logger)
            total_written += n
            done_sentinel.touch()
        except Exception as e:
            logger.error("Error processing %s: %s", video_path.name, e)

    logger.info("\nDone. Wrote %d OFIQ face crop video(s) total.", total_written)
    face_crops_logger.print_summary()


if __name__ == "__main__":
    main()
