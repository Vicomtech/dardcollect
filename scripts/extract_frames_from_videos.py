#!/usr/bin/env python3
"""
Extract video frames as PNG images with FAIR-compliant metadata.

Converts video files (from extract_person_clips_from_videos.py,
extract_face_crops_from_videos.py, or filter_face_crops_by_quality.py) into
frame sequences with per-frame JSON sidecars and a frames_manifest.json for
discovery.

Each frame gets:
- frame_XXXXXX.png (zero-padded 6-digit frame number)
- frame_XXXXXX.json (frame metadata with UUID, parent reference, detection data)

Manifest JSON lists all frames with their UUIDs for batch discovery.

Usage:
  python scripts/extract_frames_from_videos.py \
    --input-dir DARD/extracted_person_clips \
    --output-dir DARD/extracted_frames/person_clips \
    --type person_clip

All parameters are read from config.yaml under the 'frame_extraction' key.
"""

import logging
import sys
from pathlib import Path

from tqdm import tqdm

from dardcollect.config import FrameExtractionConfig
from dardcollect.frames import extract_frames
from dardcollect.pipeline_loggers import FramesExtractionLogger
from dardcollect.pipeline_utils import _TqdmHandler

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Path to config file
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def main(config_path: str | None = None) -> None:
    """Main entry point."""
    if config_path is None:
        config_path = str(CONFIG_PATH)

    cfg = FrameExtractionConfig.from_yaml(config_path)

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Initialize frames logger
    clips_csv = Path(cfg.input_dir) / "clips_extraction.csv"
    frames_logger = FramesExtractionLogger(output_dir=str(output_dir), clips_csv_path=clips_csv)

    # Find all video files
    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        logger.warning("No MP4 files found in %s", input_dir)
        sys.exit(0)

    clip_type = cfg.get_type()
    logger.info("Extracting frames from %d videos (type: %s)", len(video_files), clip_type)

    for video_path in tqdm(video_files, desc="Extracting frames", unit="video"):
        sidecar_path = video_path.with_suffix(".json")

        # Create output directory for this video
        video_output_dir = output_dir / video_path.stem

        extract_frames(
            video_path,
            sidecar_path,
            video_output_dir,
            clip_type=clip_type,
            overwrite=cfg.overwrite,
            frames_logger=frames_logger,
        )

    logger.info("Frame extraction complete")
    frames_logger.print_summary()


if __name__ == "__main__":
    main()
