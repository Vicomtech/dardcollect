#!/usr/bin/env python3
"""
Extract normalized face crop videos from person clip videos.

Reads the sidecar JSONs written by extract_person_clips.py (which contain
SG-smoothed keypoints and pre-computed face crop corners) instead of
re-running detection, so face crops use the same smoothed keypoint positions
as the rest of the pipeline.

Two crop formats are produced for every track, in separate subdirectories:

  output_dir/arcface/  — 112×112, ArcFace-aligned (insightface convention).
    Tight, face-filling crops with eyes mapped to fixed canonical positions.
    Compatible with MagFace (IResNet50) and any ArcFace identity model.
    Input to filter_face_crops_by_quality.py.

  output_dir/ofiq/     — 616×616, OFIQ-aligned (BSI-OFIQ convention).
    Wider framing with eyes at y≈272, nose at y≈336, mouth at y≈402.
    Matches the internal aligned-face format expected by all OFIQ quality
    measures: sharpness, expression neutrality, head pose, compression
    artifacts, background uniformity, and face occlusion.
    Input to annotate_face_quality.py.

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
from persondet.face_geometry import ARCFACE_SIZE, OFIQ_SIZE
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


def _cleanup_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            logger.warning("  Could not remove %s: %s", path.name, e)


def _is_track_complete(arcface_path: Path, ofiq_path: Path) -> bool:
    """Check if both crop formats for a track have been fully written."""
    for p in (arcface_path, ofiq_path):
        if not p.exists() or not p.with_suffix(".json").exists():
            return False
    return True


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
    """Extract face crop videos from a single source clip using its sidecar JSON.

    Produces two crop formats per track into output_dir/arcface/ and output_dir/ofiq/.
    """
    output_dir = Path(face_config.output_dir)
    arcface_dir = output_dir / "arcface"
    ofiq_dir = output_dir / "ofiq"
    arcface_dir.mkdir(parents=True, exist_ok=True)
    ofiq_dir.mkdir(parents=True, exist_ok=True)

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

    # track_id → list of (relative_frame_idx, arcface_crop_or_None, ofiq_crop_or_None)
    track_frames: dict[int, list[tuple[int, np.ndarray | None, np.ndarray | None]]] = defaultdict(
        list
    )

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

            has_arcface = "face_crop_corners_arcface" in det
            has_ofiq = "face_crop_corners_ofiq" in det

            if not has_arcface and not has_ofiq:
                track_frames[tid].append((frame_id, None, None))
                continue

            overlapping = any(
                _bbox_iou(bbox, ob) > face_config.max_overlap_iou
                for oid, ob in frame_bboxes
                if oid != tid
            )
            if overlapping:
                track_frames[tid].append((frame_id, None, None))
                continue

            arcface_crop: np.ndarray | None = None
            if has_arcface:
                corners = np.array(det["face_crop_corners_arcface"], dtype=np.float32)
                arcface_crop = _corners_to_warp(frame, corners, ARCFACE_SIZE)

            ofiq_crop: np.ndarray | None = None
            if has_ofiq:
                corners = np.array(det["face_crop_corners_ofiq"], dtype=np.float32)
                ofiq_crop = _corners_to_warp(frame, corners, OFIQ_SIZE)

            track_frames[tid].append((frame_id, arcface_crop, ofiq_crop))

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write one video pair per track ────────────────────────────────────────
    written = 0
    black_arcface = np.zeros((ARCFACE_SIZE, ARCFACE_SIZE, 3), dtype=np.uint8)
    black_ofiq = np.zeros((OFIQ_SIZE, OFIQ_SIZE, 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore

    for tid, frames in track_frames.items():
        valid_frames = [(fid, ac, oc) for fid, ac, oc in frames if ac is not None or oc is not None]
        if len(valid_frames) < face_config.min_track_face_frames:
            logger.debug(
                "  Track %d: only %d valid face frame(s), skipping",
                tid,
                len(valid_frames),
            )
            continue

        stem = f"{video_path.stem}_face_{tid}"
        arcface_path = arcface_dir / f"{stem}.mp4"
        ofiq_path = ofiq_dir / f"{stem}.mp4"
        arcface_sidecar = arcface_path.with_suffix(".json")
        ofiq_sidecar = ofiq_path.with_suffix(".json")

        if _is_track_complete(arcface_path, ofiq_path):
            logger.info("  SKIP (already complete): %s", stem)
            continue

        if arcface_path.exists() or ofiq_path.exists():
            logger.info("  Incomplete write detected, cleaning up: %s", stem)
            _cleanup_files(arcface_path, arcface_sidecar, ofiq_path, ofiq_sidecar)

        check_disk_space(output_dir, face_config.min_free_disk_gb)

        arcface_writer = cv2.VideoWriter(
            str(arcface_path), fourcc, fps, (ARCFACE_SIZE, ARCFACE_SIZE)
        )
        ofiq_writer = cv2.VideoWriter(str(ofiq_path), fourcc, fps, (OFIQ_SIZE, OFIQ_SIZE))

        if not arcface_writer.isOpened() or not ofiq_writer.isOpened():
            logger.error("Cannot open output file(s) for %s — stopping.", stem)
            _cleanup_files(arcface_path, ofiq_path)
            sys.exit(1)

        if face_config.skip_no_face_frames:
            for _, ac, oc in valid_frames:
                if ac is not None:
                    arcface_writer.write(ac)
                if oc is not None:
                    ofiq_writer.write(oc)
        else:
            first_fid = frames[0][0]
            last_fid = frames[-1][0]
            arcface_dict = {fid: ac for fid, ac, _ in frames if ac is not None}
            ofiq_dict = {fid: oc for fid, _, oc in frames if oc is not None}
            last_arcface = black_arcface
            last_ofiq = black_ofiq
            for fid in range(first_fid, last_fid + 1):
                if fid in arcface_dict:
                    last_arcface = arcface_dict[fid]
                if fid in ofiq_dict:
                    last_ofiq = ofiq_dict[fid]
                arcface_writer.write(last_arcface)
                ofiq_writer.write(last_ofiq)

        arcface_writer.release()
        ofiq_writer.release()

        for p in (arcface_path, ofiq_path):
            if not p.exists() or p.stat().st_size == 0:
                logger.error("Output file %s is missing or empty — stopping.", p.name)
                _cleanup_files(arcface_path, ofiq_path)
                sys.exit(1)

        first_fid, last_fid = frames[0][0], frames[-1][0]
        n_valid = len(valid_frames)

        if face_config.include_audio and not face_config.skip_no_face_frames:
            start_t = first_fid / fps
            end_t = (last_fid + 1) / fps
            _mux_audio(video_path, arcface_path, start_t, end_t)
            _mux_audio(video_path, ofiq_path, start_t, end_t)

        base_meta = {
            "source_video": str(video_path),
            "track_id": tid,
            "first_frame": first_fid,
            "last_frame": last_fid,
            "start_seconds": round(first_fid / fps, 3) if fps > 0 else 0,
            "end_seconds": round(last_fid / fps, 3) if fps > 0 else 0,
            "total_frames_in_span": last_fid - first_fid + 1,
            "valid_face_frames": n_valid,
        }
        for path, sidecar, fmt, size in (
            (arcface_path, arcface_sidecar, "arcface", ARCFACE_SIZE),
            (ofiq_path, ofiq_sidecar, "ofiq", OFIQ_SIZE),
        ):
            meta = {**base_meta, "crop_format": fmt, "output_size": size}
            try:
                with open(sidecar, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except OSError as e:
                logger.error("Cannot write %s (%s) — stopping.", sidecar.name, e)
                _cleanup_files(arcface_path, arcface_sidecar, ofiq_path, ofiq_sidecar)
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

    logger.info("\nDone. Wrote %d face crop track(s) total (arcface + ofiq pairs).", total_written)

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
