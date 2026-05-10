#!/usr/bin/env python3
"""
Extract video clips containing people from downloaded videos.

Uses person detection and tracking to identify segments where
people are visible, then extracts those clips as separate files.

All parameters are read from config.yaml.
"""

import logging
import sys
from pathlib import Path

from dardcollect.pipeline_utils import _TqdmHandler

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

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig, get_log_level
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.person_clips import process_video


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
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)

    # Initialize components
    logger.info("Initializing detector (%s)...", det_model_path.name)
    detector = PersonDetector(det_config, model_path=str(det_model_path))

    logger.info("Initializing tracker (OC-SORT)...")
    tracker = PersonTracker()

    logger.info("Initializing pose estimator (%s)...", pose_model_path.name)
    poser = PoseEstimator(det_config, model_path=str(pose_model_path))

    # Audio Transcriber - Removed from main loop
    # run pipeline/transcribe_video_clips.py instead

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
