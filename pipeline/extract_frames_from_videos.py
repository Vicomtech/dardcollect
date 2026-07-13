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
  python pipeline/extract_frames_from_videos.py \
    --input-dir DARD/extracted_person_clips \
    --output-dir DARD/extracted_frames/person_clips \
    --type person_clip

All parameters are read from config.yaml under the 'frame_extraction' key.
"""

import logging
import os
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
CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG", Path(__file__).parent.parent / "configs" / "config.archive_all.yaml"
    )
)


def main(config_path: str | None = None) -> None:
    """Extract PNG frames from video clips with FAIR-compliant metadata sidecars.

    Reads video files from the configured input directory and writes each frame
    as a PNG image with a companion JSON sidecar containing frame-level metadata
    (UUID, timestamp, detection data, parent reference). Also produces a
    frames_manifest.json for batch discovery.

    Args:
        config_path: Path to config.yaml. If None, uses the default config
            file alongside the pipeline scripts.
    """
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

    # Find all video files (recursively: clips may live in per-source-subdir trees)
    video_files = sorted(input_dir.rglob("*.mp4"))

    if not video_files:
        logger.warning("No MP4 files found in %s", input_dir)
        sys.exit(0)

    clip_type = cfg.get_type()
    logger.info("Extracting frames from %d videos (type: %s)", len(video_files), clip_type)

    for video_path in tqdm(video_files, desc="Extracting frames", unit="video"):
        sidecar_path = video_path.with_suffix(".json")

        # Mirror input_dir subtree under output_dir so frames keep the same
        # per-source-subdir layout as the face crops (filtered_face_crops/0c9460bf-.../
        # → extracted_frames/0c9460bf-.../clip_face_1/frame_*.png). Video stems are
        # unique within input_dir, so the final stem dir is unambiguous.
        rel_parent = video_path.relative_to(input_dir).parent
        video_output_dir = output_dir / rel_parent / video_path.stem

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
