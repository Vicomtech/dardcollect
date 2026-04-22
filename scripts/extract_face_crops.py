#!/usr/bin/env python3
"""
Extract normalized face crop videos from person clip videos.

Reads the sidecar JSONs written by extract_person_clips.py (which contain
SG-smoothed keypoints) instead of re-running detection, so the face crops
use the same smoothed keypoint positions as the rest of the pipeline.

Output videos are square (output_size × output_size) and suitable for
identity recognition, expression recognition, deepfake detection, etc.

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
logging.basicConfig(handlers=[_handler], level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persondet.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import imageio_ffmpeg
import numpy as np

import persondet
from persondet.config import FaceCropConfig
from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage

# ── Face keypoint indices (RTMPose / RTMW wholebody) ─────────────────────────
_KPT_NOSE = 0
_KPT_L_EYE = 1
_KPT_R_EYE = 2
_KPT_L_EAR = 3
_KPT_R_EAR = 4

# ArcFace canonical eye positions for a 112×112 output (insightface convention).
# Scaled by (output_size / 112) at crop time to support arbitrary output resolutions.
# These positions produce crops compatible with IResNet50-based models (ArcFace, MagFace).
_ARCFACE_L_EYE_112 = np.array([38.2946, 51.6963], dtype=np.float32)
_ARCFACE_R_EYE_112 = np.array([73.5318, 51.5014], dtype=np.float32)


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _bbox_iou(a: list, b: list) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_face_crop(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    output_size: int,
    face_padding: float,
    align_face: bool,
    keypoint_threshold: float,
    min_eye_distance_px: float,
) -> np.ndarray | None:
    """Return a normalized face crop, or None.

    When align_face=True (recommended), applies a similarity transform that maps
    the detected eye keypoints to ArcFace canonical positions scaled to output_size.
    The resulting crop is directly compatible with IResNet50-based models (ArcFace,
    MagFace) without any further alignment.

    When align_face=False, falls back to an axis-aligned square crop centred on the
    midface with face_padding extra context. face_padding has no effect when
    align_face=True (ArcFace canonical positions encode the appropriate layout).
    """
    if scores[_KPT_L_EYE] < keypoint_threshold or scores[_KPT_R_EYE] < keypoint_threshold:
        return None

    l_eye = keypoints[_KPT_L_EYE].astype(np.float32)
    r_eye = keypoints[_KPT_R_EYE].astype(np.float32)

    eye_dist = float(np.linalg.norm(r_eye - l_eye))
    if eye_dist < min_eye_distance_px:
        return None

    if align_face:
        scale = output_size / 112.0
        # COCO _KPT_L_EYE = person's left = viewer's right (larger x in image).
        # ArcFace L_EYE (x=38) = viewer's left.  Map viewer's-side to viewer's-side
        # so the similarity transform is a pure scale+translate (no 180° flip).
        src_pts = np.stack([l_eye, r_eye])  # (2, 2)
        dst_pts = np.stack(
            [
                _ARCFACE_R_EYE_112 * scale,  # COCO l_eye (viewer's right) → ArcFace R
                _ARCFACE_L_EYE_112 * scale,  # COCO r_eye (viewer's left)  → ArcFace L
            ]
        )  # (2, 2)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if M is None:
            return None
    else:
        eye_mid = (l_eye + r_eye) * 0.5
        face_center = eye_mid + np.array([0.0, eye_dist * 0.5], dtype=np.float32)
        half_size = eye_dist * 1.5 * (1.0 + face_padding)
        cx, cy = float(face_center[0]), float(face_center[1])
        src = np.array(
            [
                [cx - half_size, cy - half_size],
                [cx + half_size, cy - half_size],
                [cx - half_size, cy + half_size],
            ],
            dtype=np.float32,
        )
        dst = np.array([[0, 0], [output_size, 0], [0, output_size]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)

    return cv2.warpAffine(
        frame,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            logger.warning("  Could not remove %s: %s", path.name, e)


def _is_track_complete(out_path: Path) -> bool:
    """Check if a track has been fully written (output + sidecar JSON exist)."""
    sidecar_path = out_path.with_suffix(".json")
    return out_path.exists() and sidecar_path.exists()


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


# ── Per-video processing ──────────────────────────────────────────────────────


def process_video(
    video_path: Path,
    face_config: FaceCropConfig,
) -> int:
    """Extract face crop videos from a single source clip using its sidecar JSON."""
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
    frame_data: dict = clip_data.get("frame_data", {})

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

    # track_id → list of (relative_frame_idx, crop_or_None)
    track_frames: dict[int, list[tuple[int, np.ndarray | None]]] = defaultdict(list)

    frame_id = 0
    pbar = tqdm(total=total_frames, unit="fr", desc=video_path.name[:40], dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        abs_frame = start_frame + frame_id
        detections = frame_data.get(str(abs_frame), [])

        frame_bboxes = [(d["track_id"], d["bbox"]) for d in detections]

        for det in detections:
            tid = det["track_id"]
            bbox = det["bbox"]

            if "keypoints" not in det or "keypoint_scores" not in det:
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

            keypoints = np.array(det["keypoints"], dtype=np.float32)
            kpt_scores = np.array(det["keypoint_scores"], dtype=np.float32)

            crop = _extract_face_crop(
                frame,
                keypoints,
                kpt_scores,
                output_size=face_config.output_size,
                face_padding=face_config.face_padding,
                align_face=face_config.align_face,
                keypoint_threshold=face_config.pose_keypoint_threshold,
                min_eye_distance_px=face_config.min_eye_distance_px,
            )

            track_frames[tid].append((frame_id, crop))

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write one video per track ─────────────────────────────────────────────
    written = 0
    skipped = 0
    black_frame = np.zeros((face_config.output_size, face_config.output_size, 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    for tid, frames in track_frames.items():
        valid_crops = [(fid, c) for fid, c in frames if c is not None]
        if len(valid_crops) < face_config.min_track_face_frames:
            logger.debug(
                "  Track %d: only %d valid face frame(s), skipping",
                tid,
                len(valid_crops),
            )
            continue

        out_name = f"{video_path.stem}_face_{tid}.mp4"
        out_path = output_dir / out_name
        sidecar_path = out_path.with_suffix(".json")

        # Check if track is already complete (both video and sidecar exist)
        if _is_track_complete(out_path):
            logger.info("  SKIP (already complete): %s", out_name)
            skipped += 1
            continue

        # If only video exists without sidecar, it's an incomplete write from a crash
        if out_path.exists():
            logger.info("  Incomplete write detected, cleaning up: %s", out_name)
            _cleanup_files(out_path, sidecar_path)

        check_disk_space(output_dir, face_config.min_free_disk_gb)
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            fps,
            (face_config.output_size, face_config.output_size),
        )
        if not writer.isOpened():
            logger.error("Cannot open %s for writing — stopping.", out_name)
            _cleanup_files(out_path)
            sys.exit(1)

        if face_config.skip_no_face_frames:
            for _, crop in valid_crops:
                writer.write(crop)
        else:
            first_fid = frames[0][0]
            last_fid = frames[-1][0]
            frame_dict = {fid: crop for fid, crop in frames if crop is not None}
            last_valid = black_frame
            for fid in range(first_fid, last_fid + 1):
                if fid in frame_dict:
                    last_valid = frame_dict[fid]
                writer.write(last_valid)

        writer.release()

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.error("Output file %s is missing or empty — stopping.", out_name)
            _cleanup_files(out_path)
            sys.exit(1)

        if face_config.include_audio and not face_config.skip_no_face_frames:
            first_fid_audio = frames[0][0]
            last_fid_audio = frames[-1][0]
            _mux_audio(
                source_path=video_path,
                face_crop_path=out_path,
                start_t=first_fid_audio / fps,
                end_t=(last_fid_audio + 1) / fps,
            )

        n_valid = len(valid_crops)
        first_fid, last_fid = frames[0][0], frames[-1][0]
        meta = {
            "source_video": str(video_path),
            "track_id": tid,
            "output_size": face_config.output_size,
            "align_face": face_config.align_face,
            "face_padding": face_config.face_padding,
            "first_frame": first_fid,
            "last_frame": last_fid,
            "start_seconds": round(first_fid / fps, 3) if fps > 0 else 0,
            "end_seconds": round(last_fid / fps, 3) if fps > 0 else 0,
            "total_frames_in_span": last_fid - first_fid + 1,
            "valid_face_frames": n_valid,
        }
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            logger.error("Cannot write %s (%s) — stopping.", sidecar_path.name, e)
            _cleanup_files(sidecar_path, out_path)
            sys.exit(1)

        logger.info(
            "  Wrote %s  (%d valid face frames / %d span frames)",
            out_name,
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

    logger.info("\nDone. Wrote %d face crop video(s) total.", total_written)

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
