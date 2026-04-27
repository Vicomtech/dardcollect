#!/usr/bin/env python3
"""
Filter face crop videos by facial image quality using the OFIQ unified quality score.

The quality metric follows ISO/IEC 29794-5 (OFIQ): the unified quality score is
the output magnitude of MagFace (IResNet50), which measures how confidently a
face recognition model can embed a given crop — a standard proxy for biometric
sample quality defined in the OFIQ reference implementation
(https://github.com/BSI-OFIQ/OFIQ-Project).

Input directory (produced by extract_face_crops.py):
  input_dir/  — 616×616 OFIQ-aligned crops (flat layout, no subdirs)

The sidecar JSON alongside each video uses the same format as person clip
sidecars: start_frame/end_frame, video_info, and per-frame frame_data entries
containing keypoints, bbox, score, and face_crop_corners_arcface.  MagFace
scoring extracts 112×112 ArcFace crops from OFIQ frames on-the-fly using the
constant region defined in persondet/face_geometry.py.

When a clip passes the quality threshold, the OFIQ video and its sidecar are
moved to output_dir/.

Quality score: MagFace output calibrated to [0, 100] using OFIQ sigmoid
transformation with parameters x₀=23.0, w=2.6 (higher = better).

All parameters are read from config.yaml under the 'face_quality_filtering' key.
"""

import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
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
import onnxruntime as ort

import persondet
from persondet.config import FaceQualityFilterConfig, get_log_level
from persondet.face_geometry import arcface_from_ofiq_frame
from persondet.magface import load_magface, score_frame
from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage

# ── Disk-space guard ──────────────────────────────────────────────────────────


def _check_disk_space(path: Path, min_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {path} (minimum {min_gb} GB required)")


# ── Per-video quality assessment ──────────────────────────────────────────────


def _passes_quality(
    video_path: Path,
    session: ort.InferenceSession,
    threshold: float,
) -> tuple[bool, float]:
    """Score frames sequentially and exit as soon as one meets the threshold.

    Reads OFIQ 616×616 frames, extracts a 112×112 ArcFace crop from each using
    the precomputed constant region, then scores with MagFace.

    The returned max_score is the highest score seen up to the passing frame —
    a lower bound on the clip's true peak quality, but sufficient for filtering
    and for relative comparison between clips.

    :param video_path: Path to the OFIQ face crop .mp4 file.
    :param session: Loaded MagFace ONNX session.
    :param threshold: Minimum quality score required to pass.
    :return: (passes, max_score) tuple.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — skipping", video_path.name)
        return False, 0.0

    max_score = 0.0
    try:
        while True:
            ret, ofiq_frame = cap.read()
            if not ret:
                break
            arcface_frame = arcface_from_ofiq_frame(ofiq_frame)
            score = score_frame(session, arcface_frame)
            if score > max_score:
                max_score = score
            if max_score >= threshold:
                return True, max_score
    finally:
        cap.release()

    return False, max_score


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = FaceQualityFilterConfig.from_yaml(str(CONFIG_PATH))
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\nRun extract_face_crops.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(output_dir, cfg.min_free_disk_gb)

    video_files = sorted(input_dir.glob("*_face_*.mp4"))
    if not video_files:
        logger.info("No face crop videos found in %s", input_dir)
        return

    logger.info(
        "Found %d OFIQ crop videos in %s — quality threshold %.2f",
        len(video_files),
        input_dir,
        cfg.quality_threshold,
    )

    session = load_magface(cfg.gpu_id)

    started_at = now_iso()
    videos_assessed = 0
    videos_passed = 0
    videos_skipped = 0
    all_scores = []

    for video_path in tqdm(video_files, desc="Quality filtering", unit="video"):
        sidecar_path = video_path.with_suffix(".json")
        dest_video = output_dir / video_path.name
        dest_sidecar = output_dir / sidecar_path.name

        # Idempotency: already moved
        if dest_video.exists():
            logger.debug("Already in output dir, skipping: %s", video_path.name)
            videos_skipped += 1
            continue

        if not sidecar_path.exists():
            logger.warning("Missing sidecar JSON for %s — skipping", video_path.name)
            continue

        _check_disk_space(output_dir, cfg.min_free_disk_gb)

        try:
            passes, max_score = _passes_quality(video_path, session, cfg.quality_threshold)
        except Exception as exc:
            logger.error("Error assessing %s: %s", video_path.name, exc)
            continue

        videos_assessed += 1
        all_scores.append(max_score)

        if passes:
            shutil.move(str(video_path), dest_video)
            shutil.move(str(sidecar_path), dest_sidecar)
            videos_passed += 1
            logger.info("PASS %s (score=%.4f) → %s", video_path.name, max_score, output_dir)
        else:
            logger.debug("FAIL %s (score=%.4f)", video_path.name, max_score)

    logger.info(
        "Done. Assessed: %d  Passed: %d  Skipped (already done): %d",
        videos_assessed,
        videos_passed,
        videos_skipped,
    )

    # Print quality score distribution
    if all_scores:
        scores_array = np.array(all_scores)
        logger.info(
            "Quality score distribution (calibrated [0, 100]):\n"
            "  Min: %.2f  |  Max: %.2f  |  Mean: %.2f  |  Median: %.2f\n"
            "  P10: %.2f  |  P25: %.2f  |  P50: %.2f  |  P75: %.2f  |  P90: %.2f",
            scores_array.min(),
            scores_array.max(),
            scores_array.mean(),
            np.median(scores_array),
            np.percentile(scores_array, 10),
            np.percentile(scores_array, 25),
            np.percentile(scores_array, 50),
            np.percentile(scores_array, 75),
            np.percentile(scores_array, 90),
        )

    record_stage(
        output_dir.parent / PROVENANCE_FILENAME,
        {
            "stage": "filter_face_crops_by_quality",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {
                "script": "scripts/filter_face_crops_by_quality.py",
                "persondet_version": persondet.__version__,
            },
            "config": asdict(cfg),
            "stats": {
                "videos_assessed": videos_assessed,
                "videos_passed": videos_passed,
                "videos_skipped_already_done": videos_skipped,
            },
        },
    )


if __name__ == "__main__":
    main()
