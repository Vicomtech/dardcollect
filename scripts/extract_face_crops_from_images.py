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
the arcface_crop_corners_in_ofiq field to extract 112×112 ArcFace crops for MagFace scoring.

All parameters are read from config.yaml under the 'face_crop_extraction' key.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from persondet.config import FaceCropConfig, get_log_level
from persondet.face_geometry import face_crop_corners
from persondet.fair import add_fair_metadata, reorganize_for_fair
from persondet.pipeline_loggers import ImageFaceCropsExtractionLogger
from persondet.provenance import now_iso
from persondet.script_utilities import _TqdmHandler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# OFIQ crop size (all quality measures expect 616×616)
OFIQ_SIZE = 616

# ArcFace crop corners in OFIQ space (constant for all images)
# These are the 4 corners [TL, TR, BR, BL] of the 112×112 ArcFace region in OFIQ coordinates
ARCFACE_CROP_CORNERS_IN_OFIQ = [
    [252, 240],  # TL
    [364, 240],  # TR
    [364, 352],  # BR
    [252, 352],  # BL
]


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _corners_to_warp(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Warp frame to an output_size square given 4 source-frame corners [TL,TR,BR,BL]."""
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
    corners: np.ndarray,
) -> tuple[list, list]:
    """Transform keypoints to the output crop space using the OFIQ alignment transform."""
    if len(keypoints_source_array) == 0:
        return [], keypoint_scores

    # Compute the affine transform from source corners
    S = output_size
    src = corners[:3].astype(np.float32)
    dst = np.array([[0, 0], [S, 0], [S, S]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)

    transformed = []
    for kpt in keypoints_source_array:
        pt = np.array([float(kpt[0]), float(kpt[1]), 1.0])
        transformed_pt = M @ pt
        transformed.append([float(transformed_pt[0]), float(transformed_pt[1])])

    return transformed, keypoint_scores


def _get_or_compute_corners(det: dict, face_config: FaceCropConfig) -> np.ndarray | None:
    """Get corners from detection or compute from keypoints."""
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

    corners = face_crop_corners(
        keypoints=kpts_array,
        kpt_scores=scores_array,
        mode="ofiq",
        keypoint_threshold=0.2,
        min_eye_distance_px=face_config.min_eye_distance_px,
    )

    if corners is None:
        return None

    return corners.astype(np.float32)


def process_image(
    image_path: Path,
    detection_json_path: Path,
    face_config: FaceCropConfig,
    output_dir: Path,
    logger_instance: ImageFaceCropsExtractionLogger | None = None,
) -> int:
    """Extract OFIQ face crop images from a single source image using its detection JSON.

    Returns:
        Number of crops written
    """
    if not detection_json_path.exists():
        logger.warning("No detection JSON for %s — skipping", image_path.name)
        return 0

    with open(detection_json_path, encoding="utf-8") as f:
        detection_data = json.load(f)

    detections = detection_data.get("detections", [])
    if not detections:
        logger.debug("No detections in %s", image_path.name)
        return 0

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Cannot read image: %s", image_path.name)
        return 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image_rgb.shape[:2]

    written = 0
    arcface_corners_json = [
        [round(float(x), 2), round(float(y), 2)] for x, y in ARCFACE_CROP_CORNERS_IN_OFIQ
    ]

    for person_idx, det in enumerate(detections):
        # Compute or get corners from keypoints
        corners = _get_or_compute_corners(det, face_config)
        if corners is None:
            logger.debug("  Person %d: cannot compute face crop corners, skipping", person_idx)
            continue

        # Check face visibility
        face_visible = det.get("face_visible", False)
        if not face_visible:
            logger.debug("  Person %d: face not visible, skipping", person_idx)
            continue

        # Extract OFIQ crop
        ofiq_crop = _corners_to_warp(image_rgb, corners, OFIQ_SIZE)

        # Transform keypoints to OFIQ space
        keypoints = det.get("keypoints", [])
        keypoint_scores = det.get("keypoint_scores", [])
        if keypoints and keypoint_scores:
            kpts_array = np.array(keypoints, dtype=np.float32)
            scores_array = np.array(keypoint_scores, dtype=np.float32)
            transformed_kpts, transformed_scores = _transform_keypoints(
                keypoints, keypoint_scores, kpts_array, scores_array, OFIQ_SIZE, corners
            )
        else:
            transformed_kpts, transformed_scores = [], []

        # Build sidecar metadata
        stem = f"{image_path.stem}_face_{person_idx}"
        sidecar_meta = {
            "uuid": detection_data.get("uuid", ""),  # Parent image UUID
            "image_path": image_path.name,
            "person_idx": person_idx,
            "source_image_size": {
                "width": image_width,
                "height": image_height,
            },
            "bbox_in_source": det.get("bbox_tlbr", []),
            "bbox_confidence": det.get("bbox_confidence", 0.0),
            "keypoints": transformed_kpts,
            "keypoint_scores": transformed_scores,
            "crop_format": "ofiq",
            "output_size": OFIQ_SIZE,
            "arcface_crop_corners_in_ofiq": arcface_corners_json,
            "extracted_at": now_iso(),
        }

        # Add FAIR metadata
        sidecar_meta = add_fair_metadata(
            sidecar_meta,
            schema_type="face_crop",
            parent_uuid=detection_data.get("uuid", ""),
        )

        # Reorganize for FAIR
        sidecar_meta = reorganize_for_fair(sidecar_meta, "face_crop")

        # Write crop image (BGR for cv2)
        ofiq_crop_bgr = cv2.cvtColor(ofiq_crop, cv2.COLOR_RGB2BGR)
        crop_path = output_dir / f"{stem}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(crop_path), ofiq_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.warning("Failed to write crop: %s", crop_path)
            continue

        # Write sidecar JSON
        json_path = crop_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_meta, f, indent=2)

        # Log extraction to traceability CSV
        if logger_instance:
            bbox = det.get("bbox_tlbr", [None, None, None, None])
            face_bbox = f"{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}"
            logger_instance.log_face_crop_extraction(
                crop_id=sidecar_meta.get("uuid", stem),
                source_image=image_path.name,
                source_image_path=str(image_path.absolute()),
                face_bbox=face_bbox,
                confidence=float(det.get("bbox_confidence", 0.0)),
                width=OFIQ_SIZE,
                height=OFIQ_SIZE,
                output_path=str(crop_path.absolute()),
            )

        logger.debug("  Wrote crop: %s", crop_path.name)
        written += 1

    return written


def main():
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

    # Initialize traceability logger
    extraction_logger = ImageFaceCropsExtractionLogger(output_dir=str(output_dir))

    total_written = 0
    for image_path in tqdm(image_files, desc="Extracting face crops from images", unit="image"):
        json_path = image_path.with_suffix(".json")
        try:
            n = process_image(image_path, json_path, face_config, output_dir, extraction_logger)
            total_written += n
        except Exception as e:
            logger.error("Error processing %s: %s", image_path.name, e)

    logger.info("Done. Wrote %d OFIQ face crop image(s) total.", total_written)
    extraction_logger.print_summary()


if __name__ == "__main__":
    main()
