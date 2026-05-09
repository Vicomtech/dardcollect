#!/usr/bin/env python3
"""
Filter face crop videos and images by facial image quality using the OFIQ unified quality score.

The quality metric follows ISO/IEC 29794-5 (OFIQ): the unified quality score is
the output magnitude of MagFace (IResNet50), which measures how confidently a
face recognition model can embed a given crop — a standard proxy for biometric
sample quality defined in the OFIQ reference implementation
(https://github.com/BSI-OFIQ/OFIQ-Project).

Input directory (produced by extract_face_crops_from_videos.py or
extract_face_crops_from_images.py):
  input_dir/  — 616×616 OFIQ-aligned crops (flat layout, no subdirs)

Supported crop formats:
  - Videos: .mp4 files (from extract_face_crops_from_videos.py) with per-frame quality assessment
  - Images: .jpg/.png files (from extract_face_crops_from_images.py) with single-frame assessment

The sidecar JSON alongside each crop uses the same format: keypoints, bbox, score,
and face_crop_corners_arcface. MagFace scoring extracts 112×112 ArcFace crops from
OFIQ frames on-the-fly using the constant region defined in persondet/face_geometry.py.

When a clip/image passes the quality threshold, the crop and its sidecar are
moved to output_dir/.

Quality score: MagFace output calibrated to [0, 100] using OFIQ sigmoid
transformation with parameters x₀=23.0, w=2.6 (higher = better).

All parameters are read from config.yaml under the 'face_quality_filtering' key.
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from persondet.script_utilities import _TqdmHandler

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

from persondet.config import FaceQualityFilterConfig, get_log_level
from persondet.face_geometry import arcface_from_ofiq_frame
from persondet.magface import load_magface, score_frame
from persondet.pipeline_loggers import FilteredFaceCropsLogger

# ── Disk-space guard ──────────────────────────────────────────────────────────


def _check_disk_space(path: Path, min_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {path} (minimum {min_gb} GB required)")


# Track if we've logged the actual execution provider during inference
_provider_logged = False


def _get_frames_from_crop(crop_path: Path) -> list[np.ndarray]:
    """Read OFIQ frames from either a video (.mp4) or image (.jpg/.png) file.

    Returns:
        List of OFIQ frames (BGR format)
    """
    suffix = crop_path.suffix.lower()

    if suffix == ".mp4":
        # Read frames from video
        frames = []
        cap = cv2.VideoCapture(str(crop_path))
        if not cap.isOpened():
            logger.warning("Cannot open video %s", crop_path.name)
            return []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames
    # Read single image (.jpg, .png, etc.)
    image = cv2.imread(str(crop_path))
    if image is None:
        logger.warning("Cannot read image %s", crop_path.name)
        return []
    return [image]


# ── Per-crop quality assessment ──────────────────────────────────────────────


def _passes_quality(
    crop_path: Path,
    session: ort.InferenceSession,
    threshold: float,
) -> tuple[bool, float]:
    """Score frames from a crop (video or image) and exit as soon as one meets the threshold.

    Reads OFIQ 616×616 frames/image, extracts a 112×112 ArcFace crop from each using
    the precomputed constant region, then scores with MagFace.

    The returned max_score is the highest score seen up to the passing frame —
    a lower bound on the crop's true peak quality, but sufficient for filtering
    and for relative comparison between crops.

    :param crop_path: Path to the OFIQ face crop (.mp4 video or .jpg/.png image).
    :param session: Loaded MagFace ONNX session.
    :param threshold: Minimum quality score required to pass.
    :return: (passes, max_score) tuple.
    """
    global _provider_logged

    frames = _get_frames_from_crop(crop_path)
    if not frames:
        logger.warning("No frames from %s — skipping", crop_path.name)
        return False, 0.0

    max_score = 0.0
    for ofiq_frame in frames:
        arcface_frame = arcface_from_ofiq_frame(ofiq_frame)
        score = score_frame(session, arcface_frame)

        # Log actual execution provider on first frame
        if not _provider_logged:
            _provider_logged = True
            providers = session.get_providers()
            if providers:
                logger.info("  Actual execution provider during inference: %s", providers[0])
        if score > max_score:
            max_score = score
        if max_score >= threshold:
            return True, max_score

    return False, max_score


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = FaceQualityFilterConfig.from_yaml(str(CONFIG_PATH))
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\n"
            "Run extract_face_crops_from_videos.py or extract_face_crops_from_images.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(output_dir, cfg.min_free_disk_gb)

    # Find both video and image crops
    crop_files = sorted(input_dir.glob("*_face_*.mp4"))
    crop_files.extend(sorted(input_dir.glob("*_face_*.jpg")))
    crop_files.extend(sorted(input_dir.glob("*_face_*.png")))
    crop_files = sorted(set(crop_files))  # Remove duplicates and re-sort

    if not crop_files:
        logger.info("No face crops (videos or images) found in %s", input_dir)
        return

    logger.info(
        "Found %d OFIQ face crops in %s — quality threshold %.2f",
        len(crop_files),
        input_dir,
        cfg.quality_threshold,
    )

    session = load_magface(cfg.gpu_id)

    videos_assessed = 0
    videos_passed = 0
    videos_skipped = 0
    all_scores = []

    # Initialize filtered crops logger
    face_crops_csv = input_dir / "face_crops_extraction.csv"
    filter_logger = FilteredFaceCropsLogger(
        output_dir=str(output_dir), face_crops_csv_path=face_crops_csv
    )

    for crop_path in tqdm(crop_files, desc="Quality filtering", unit="crop"):
        sidecar_path = crop_path.with_suffix(".json")
        dest_crop = output_dir / crop_path.name
        dest_sidecar = output_dir / sidecar_path.name

        # Idempotency: already moved
        if dest_crop.exists():
            logger.debug("Already in output dir, skipping: %s", crop_path.name)
            videos_skipped += 1
            continue

        if not sidecar_path.exists():
            logger.warning("Missing sidecar JSON for %s — skipping", crop_path.name)
            continue

        _check_disk_space(output_dir, cfg.min_free_disk_gb)

        try:
            passes, max_score = _passes_quality(crop_path, session, cfg.quality_threshold)
        except Exception as exc:
            logger.error("Error assessing %s: %s", crop_path.name, exc)
            continue

        videos_assessed += 1
        all_scores.append(max_score)

        if passes:
            shutil.move(str(crop_path), dest_crop)
            shutil.move(str(sidecar_path), dest_sidecar)
            videos_passed += 1

            # Log filtered crop (for traceability)
            filter_logger.log_filtered_crop(
                source_crop_path=str(crop_path),
                magface_score=float(max_score),
                filter_threshold=float(cfg.quality_threshold),
                output_path=str(dest_crop),
            )

            logger.info("PASS %s (score=%.4f) → %s", crop_path.name, max_score, output_dir)
        else:
            logger.debug("FAIL %s (score=%.4f)", crop_path.name, max_score)

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

    # Print filtered crops summary
    filter_logger.print_summary()


if __name__ == "__main__":
    main()
