#!/usr/bin/env python3
"""
Extract normalized face crop images from static images.

Reads image files and their detection JSONs (written by extract_persons_from_images.py),
extracts 616×616 OFIQ-aligned face crops for each detected person, and saves as .jpg
files with matching .json sidecars.

One crop file is produced per person per image:
  output_dir/  — 616×616, OFIQ-aligned (BSI-OFIQ convention)
    ImageName_face_0.jpg
    ImageName_face_0.json  ← same format as video face crop sidecars

The sidecar JSON includes:
  - uuid: unique identifier
  - face_crop_corners_arcface: 4 corners of the 112×112 ArcFace region in OFIQ space
  - keypoints/keypoint_scores: transformed to OFIQ crop space
  - source image metadata and parent reference

Downstream scripts (filter_face_crops_by_quality.py, annotate_face_quality.py) use
the face_crop_corners_arcface field to extract 112×112 ArcFace crops for MagFace scoring.

All parameters are read from config.yaml under the 'face_crop_extraction' key.
"""

import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from dardcollect.config import FaceCropConfig, get_log_level
from dardcollect.face_crops import process_image
from dardcollect.pipeline_loggers import ImageFaceCropsExtractionLogger
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler

CONFIG_PATH = Path(
    os.environ.get("DARDCOLLECT_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
)


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@add_timer
def main():
    """Extract 616×616 OFIQ-aligned face crop images from static images.

    Reads image files and their detection JSONs (written by extract_persons_from_images.py),
    extracts OFIQ-format face crops for each detected person using process_image,
    and saves .jpg files with companion .json sidecars containing FAIR metadata.

    Configuration is read from config.yaml under the 'face_crop_extraction' key.
    """
    try:
        face_config = FaceCropConfig.from_yaml(
            str(CONFIG_PATH), section="image_face_crop_extraction"
        )
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    input_path = Path(face_config.input_dir)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    # Find image files (search recursively for language subfolders)
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp", ".webp")
    image_files = []

    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            image_files = [input_path]
    else:
        for ext in image_extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))  # Remove duplicates and sort

    if not image_files:
        logger.error("No image files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d image(s) to process", len(image_files))

    output_dir = Path(face_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_dir = Path(face_config.detections_dir) if face_config.detections_dir else input_path

    # Initialize traceability logger
    image_detection_csv = detections_dir / "image_person_detection.csv"
    extraction_logger = ImageFaceCropsExtractionLogger(
        output_dir=str(output_dir), image_detection_csv_path=image_detection_csv
    )

    total_written = 0
    for image_path in tqdm(image_files, desc="Extracting face crops from images", unit="image"):
        json_path = detections_dir / (image_path.stem + ".json")
        try:
            n = process_image(image_path, json_path, face_config, output_dir, extraction_logger)
            total_written += n
        except Exception as e:
            logger.error("Error processing %s: %s", image_path.name, e)

    logger.info("Done. Wrote %d OFIQ face crop image(s) total.", total_written)
    extraction_logger.print_summary()


if __name__ == "__main__":
    main()
