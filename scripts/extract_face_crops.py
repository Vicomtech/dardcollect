#!/usr/bin/env python3
"""
Extract normalized face crop videos from person clip videos.

Reads the sidecar JSONs written by extract_person_clips.py (which contain
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
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persondet.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import imageio_ffmpeg
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import persondet
from persondet.config import FaceCropConfig
from persondet.face_geometry import (
    _ALIGN_OFIQ_DST,
    _ALIGN_OFIQ_INDICES,
    ARCFACE_CROP_CORNERS_IN_OFIQ,
    OFIQ_SIZE,
    face_crop_corners,
)
from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage

# ── Geometry helpers ──────────────────────────────────────────────────────────


def _bbox_iou(a: list, b: list) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _corners_to_warp(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Warp *frame* to an output_size square given 4 source-frame corners [TL,TR,BR,BL]."""
    S = output_size
    src = corners[:3].astype(np.float32)
    dst = np.array([[0, 0], [S, 0], [S, S]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _transform_keypoints(
    keypoints: list,
    keypoint_scores: list,
    keypoints_source_array: np.ndarray,
    kpt_scores_array: np.ndarray,
    output_size: int,
) -> tuple[list, list]:
    """Transform keypoints to the output crop space using the OFIQ alignment transform.

    Args:
        keypoints: List of [x, y] keypoint coordinates in source frame (for validation)
        keypoint_scores: Corresponding confidence scores (for validation)
        keypoints_source_array: numpy array of all keypoints for accurate transformation
        kpt_scores_array: numpy array of keypoint scores
        output_size: Output crop size (e.g., 616) - not used, kept for compatibility

    Returns:
        (transformed_keypoints, keypoint_scores) where coordinates are in crop space
    """
    if len(keypoints_source_array) == 0:
        return [], keypoint_scores

    # Compute the affine matrix using the same landmarks as face_crop_corners
    indices = _ALIGN_OFIQ_INDICES
    dst_pts_full = _ALIGN_OFIQ_DST

    n_kpts = len(kpt_scores_array)
    src_list, dst_list = [], []
    for kpt_idx, canonical in zip(indices, dst_pts_full):
        if kpt_idx >= n_kpts or kpt_scores_array[kpt_idx] < 0.2:
            continue
        src_list.append(keypoints_source_array[kpt_idx].astype(np.float32))
        dst_list.append(canonical)

    if len(src_list) < 3:
        # Not enough landmarks to compute transform
        return keypoints, keypoint_scores

    src_pts = np.array(src_list, dtype=np.float32)
    dst_pts = np.array(dst_list, dtype=np.float32)
    M, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

    if M is None:
        return keypoints, keypoint_scores

    # Transform each keypoint using affine matrix
    transformed = []
    for kpt in keypoints_source_array:
        # Apply affine transform: [x', y'] = M @ [x, y, 1]
        pt = np.array([float(kpt[0]), float(kpt[1]), 1.0])
        transformed_pt = M @ pt
        transformed.append([float(transformed_pt[0]), float(transformed_pt[1])])

    return transformed, keypoint_scores


def _get_or_compute_corners(
    det: dict, face_config: FaceCropConfig, frame_id: str = ""
) -> np.ndarray | None:
    """Get corners from detection or compute from keypoints.

    Args:
        det: Detection dict with optional 'face_crop_corners_ofiq' field
        face_config: Configuration with keypoint threshold
        frame_id: Optional frame ID for logging

    Returns:
        (4, 2) corner array or None if computation fails
    """
    # Return pre-computed corners if available
    if "face_crop_corners_ofiq" in det:
        corners = np.array(det["face_crop_corners_ofiq"], dtype=np.float32)
        if corners.shape == (4, 2):
            return corners

    # Otherwise compute from keypoints
    keypoints = det.get("keypoints", [])
    keypoint_scores = det.get("keypoint_scores", [])

    if not keypoints or not keypoint_scores:
        return None

    kpts_array = np.array(keypoints, dtype=np.float32)
    scores_array = np.array(keypoint_scores, dtype=np.float32)

    # Use a lower threshold (0.2 instead of 0.3) to capture more marginal detections
    corners = face_crop_corners(
        keypoints=kpts_array,
        kpt_scores=scores_array,
        mode="ofiq",
        keypoint_threshold=0.2,
        min_eye_distance_px=face_config.min_eye_distance_px,
    )

    if corners is None and len(keypoint_scores) >= 3:
        # Debug: Check eye scores
        eye_scores = [
            keypoint_scores[1] if len(keypoint_scores) > 1 else 0,
            keypoint_scores[2] if len(keypoint_scores) > 2 else 0,
        ]
        logger.debug(
            f"[{frame_id}] Corner computation failed. Eye scores: {eye_scores}, threshold: 0.2"
        )

    return corners


def _cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            logger.warning("  Could not remove %s: %s", path.name, e)


def _is_track_complete(ofiq_path: Path) -> bool:
    """Check if the OFIQ crop for a track has been fully written."""
    return ofiq_path.exists() and ofiq_path.with_suffix(".json").exists()


def _mux_audio(
    source_path: Path,
    face_crop_path: Path,
    start_t: float,
    end_t: float,
) -> None:
    """Replace a video-only face crop file with one that includes audio."""
    tmp_path = face_crop_path.with_suffix(".tmp.mp4")
    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(face_crop_path),
                "-ss",
                str(start_t),
                "-to",
                str(end_t),
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(tmp_path),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "  ffmpeg audio mux failed for %s — keeping video-only file.\n  %s",
                face_crop_path.name,
                result.stderr.decode(errors="replace").strip(),
            )
            _cleanup_files(tmp_path)
            return
        tmp_path.replace(face_crop_path)
    except Exception as e:
        logger.warning("  Audio mux error for %s: %s", face_crop_path.name, e)
        _cleanup_files(tmp_path)


def check_disk_space(path: Path, min_free_gb: float) -> None:
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as e:
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
                            # Compute or get corners and transform keypoints to crop space
                            corners = _get_or_compute_corners(det, face_config)
                            if corners is not None and kpts and scores:
                                kpts_array = np.array(kpts, dtype=np.float32)
                                scores_array = np.array(scores, dtype=np.float32)
                                transformed_kpts, _ = _transform_keypoints(
                                    kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                                )
                                frame_data[str(output_frame_idx)] = [
                                    {
                                        "track_id": tid,
                                        "keypoints": transformed_kpts,
                                        "keypoint_scores": scores,
                                    }
                                ]
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
                # Extract keypoints from original frame data
                abs_frame = start_frame + fid
                detections = frame_data_orig.get(str(abs_frame), [])
                for det in detections:
                    if det.get("track_id") == tid:
                        kpts = det.get("keypoints", [])
                        scores = det.get("keypoint_scores", [])
                        # Compute or get corners and transform keypoints to crop space
                        corners = _get_or_compute_corners(det, face_config)
                        if corners is not None and kpts and scores:
                            kpts_array = np.array(kpts, dtype=np.float32)
                            scores_array = np.array(scores, dtype=np.float32)
                            transformed_kpts, _ = _transform_keypoints(
                                kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                            )
                            frame_data[str(output_frame_idx)] = [
                                {
                                    "track_id": tid,
                                    "keypoints": transformed_kpts,
                                    "keypoint_scores": scores,
                                }
                            ]
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

        if face_config.include_audio and not face_config.skip_no_face_frames:
            start_t = first_fid / fps
            end_t = (last_fid + 1) / fps
            _mux_audio(video_path, ofiq_path, start_t, end_t)

        meta = {
            "source_video": str(video_path),
            "track_id": tid,
            "fps": round(fps, 3),
            "first_frame": first_fid,
            "last_frame": last_fid,
            "start_seconds": round(first_fid / fps, 3) if fps > 0 else 0,
            "end_seconds": round(last_fid / fps, 3) if fps > 0 else 0,
            "total_frames_in_span": last_fid - first_fid + 1,
            "valid_face_frames": n_valid,
            "crop_format": "ofiq",
            "output_size": OFIQ_SIZE,
            "arcface_crop_corners_in_ofiq": arcface_corners_json,
            "frame_data": frame_data,
        }
        try:
            with open(ofiq_sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            logger.error("Cannot write %s (%s) — stopping.", ofiq_sidecar.name, e)
            _cleanup_files(ofiq_path, ofiq_sidecar)
            sys.exit(1)

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
    started_at = now_iso()

    try:
        face_config = FaceCropConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

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

    total_written = 0
    for video_path in video_files:
        done_sentinel = output_dir / f"{video_path.stem}.done"
        if done_sentinel.exists():
            logger.info("SKIP (already done): %s", video_path.name)
            continue

        logger.info("Processing: %s", video_path.name)
        try:
            n = process_video(video_path, face_config)
            total_written += n
            done_sentinel.touch()
        except Exception as e:
            logger.error("Error processing %s: %s", video_path.name, e)

    logger.info("\nDone. Wrote %d OFIQ face crop video(s) total.", total_written)

    record_stage(
        output_dir.parent / PROVENANCE_FILENAME,
        {
            "stage": "extract_face_crops",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {
                "script": "scripts/extract_face_crops.py",
                "persondet_version": persondet.__version__,
            },
            "config": asdict(face_config),
            "stats": {
                "videos_processed": len(video_files),
                "face_crops_written": total_written,
            },
        },
    )


if __name__ == "__main__":
    main()
