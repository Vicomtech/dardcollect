#!/usr/bin/env python3
"""
Extract person detections from static images.

Scans archive_org_public_domain/images/ for image files (.jpg, .png, etc),
detects people in each image, and writes detection JSON sidecars with
bounding boxes, pose keypoints, and FAIR metadata.

Each detection gets:
- UUID (unique identifier per image)
- Person-level detection data (bounding box, keypoints, score)
- Face visibility check and frontal face assessment
- FAIR metadata with archive.org source reference

Writes detection sidecars (.json) to output_detections_dir (separate folder,
not next to source images). Images can then be processed by
extract_face_crops_from_images.py to extract normalized face crops
(same convergent pipeline as videos).

All parameters are read from config.yaml under the 'image_extraction' key.
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dardcollect import PersonDetector, PoseEstimator
from dardcollect.config import DetectorConfig, FaceCropConfig, ImageExtractionConfig, get_log_level
from dardcollect.face_geometry import face_crop_corners
from dardcollect.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from dardcollect.gpu_setup import setup_gpu_paths
from dardcollect.pipeline_loggers import ImagePersonDetectionLogger
from dardcollect.pipeline_utils import (
    _TqdmHandler,
    check_frontal_face,
)
from dardcollect.provenance import now_iso

# Setup GPU paths BEFORE importing heavy libraries
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
setup_gpu_paths(str(CONFIG_PATH))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp", ".webp"}

# Face keypoint indices (from poser.py KEYPOINT_NAMES)
# Now imported from dardcollect.pipeline_utils


# Validation functions imported from dardcollect.pipeline_utils
# check_face_visibility, check_frontal_face


def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting image person detection with FAIR integration...")

    cfg = ImageExtractionConfig.from_yaml(str(CONFIG_PATH))

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_detections_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration for detector and pose models
    detector_cfg = DetectorConfig.from_yaml(str(CONFIG_PATH))
    face_crop_cfg = FaceCropConfig.from_yaml(str(CONFIG_PATH))

    # Verify and load detection and pose models
    models_dir = Path(detector_cfg.models_path)

    det_filename = "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_filename = "cigpose-m_coco-wholebody_256x192.onnx"

    det_model_path = models_dir / det_filename
    pose_model_path = models_dir / pose_filename

    if not det_model_path.exists():
        logger.error("Detection model not found: %s", det_model_path)
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)

    if not pose_model_path.exists():
        logger.error("Pose model not found: %s", pose_model_path)
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)

    # Initialize detectors
    logger.info("Initializing person detector (%s)...", det_model_path.name)
    try:
        detector = PersonDetector(detector_cfg, model_path=str(det_model_path))
    except Exception as e:
        logger.error("Failed to initialize detector: %s", e)
        sys.exit(1)

    logger.info("Initializing pose estimator (%s)...", pose_model_path.name)
    try:
        poser = PoseEstimator(detector_cfg, model_path=str(pose_model_path))
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize pose estimator: %s", e)
        sys.exit(1)

    # Initialize traceability logger
    downloads_csv = input_dir.parent / "downloads.csv"
    detection_logger = ImagePersonDetectionLogger(
        output_dir=str(output_dir), downloads_csv_path=downloads_csv
    )

    # Find image files needing detection
    logger.info("Scanning for images needing detection...")
    image_files = []
    for img_path in sorted(input_dir.rglob("*")):
        if img_path.is_dir():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Check if detection already exists in output_dir
        json_filename = img_path.stem + ".json"
        json_path = output_dir / json_filename
        if json_path.exists() and not cfg.overwrite:
            continue

        image_files.append(img_path)

    logger.info("Found %d images needing detection in %s", len(image_files), input_dir)

    if not image_files:
        logger.info("All images processed! Nothing to do.")
        return

    # Process Loop
    success_count = 0
    fail_count = 0

    for img_path in tqdm(image_files, desc="Detecting persons in images", unit="image"):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Failed to read image: %s", img_path.name)
                fail_count += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image_rgb.shape[:2]
            logger.debug(
                f"Processing {img_path.name} ({image_width}x{image_height}), "
                f"threshold={cfg.detection_threshold}"
            )

            # Detect persons
            det_bboxes, det_scores = detector.get_detections(image_rgb, cfg.detection_threshold)
            logger.debug(
                f"Detections: {len(det_bboxes)} persons with scores "
                f"{det_scores[:5] if len(det_scores) > 0 else []}"
            )

            if len(det_bboxes) == 0:
                logger.warning(
                    "No persons detected in %s (threshold=%.2f)",
                    img_path.name,
                    cfg.detection_threshold,
                )
                fail_count += 1
                continue

            # Estimate pose for each person
            detection_data = []
            for det_idx, bbox in enumerate(det_bboxes):
                try:
                    keypoints, keypoints_scores = poser.get_keypoints(image_rgb, bbox)
                    if keypoints is None or len(keypoints) == 0:
                        continue

                    # Attempt face crop corners directly — face_crop_corners checks eye scores
                    # and min inter-eye distance, which is the correct usability predicate.
                    # check_face_visibility's size-based gate fails on full-body archive photos
                    # where the face is small relative to the full-height bounding box.
                    face_crop_corners_arcface = None
                    face_crop_corners_ofiq = None
                    for mode in ("arcface", "ofiq"):
                        try:
                            c = face_crop_corners(
                                keypoints,
                                keypoints_scores,
                                mode,
                                face_crop_cfg.pose_keypoint_threshold,
                                face_crop_cfg.min_eye_distance_px,
                            )
                        except Exception:
                            c = None
                        if mode == "arcface":
                            face_crop_corners_arcface = c
                        else:
                            face_crop_corners_ofiq = c

                    # face_visible iff corners computable (eyes detected + sufficient distance)
                    face_visible = face_crop_corners_ofiq is not None
                    frontal = (
                        check_frontal_face(
                            keypoints,
                            keypoints_scores,
                            cfg.frontal_symmetry_threshold,
                        )
                        if face_visible
                        else False
                    )

                    detection_data.append(
                        {
                            "person_idx": det_idx,
                            "bbox_tlbr": bbox.tolist(),
                            "bbox_confidence": float(det_scores[det_idx]),
                            "keypoints": keypoints.tolist(),
                            "keypoint_scores": keypoints_scores.tolist(),
                            "face_visible": bool(face_visible),
                            "frontal_face": bool(frontal),
                            "face_crop_corners_arcface": face_crop_corners_arcface.tolist()
                            if face_crop_corners_arcface is not None
                            else None,
                            "face_crop_corners_ofiq": face_crop_corners_ofiq.tolist()
                            if face_crop_corners_ofiq is not None
                            else None,
                        }
                    )
                except Exception as e:
                    logger.debug(
                        "Failed to estimate pose for person %d in %s: %s", det_idx, img_path.name, e
                    )
                    continue

            if not detection_data:
                logger.debug("No valid detections with pose in %s", img_path.name)
                fail_count += 1
                continue

            # Build detection metadata with FAIR fields
            detection_meta = {
                "uuid": str(generate_uuid()),
                "image_path": img_path.name,
                "image_size": {
                    "width": image_width,
                    "height": image_height,
                },
                "detection_timestamp": now_iso(),
                "num_persons": len(detection_data),
                "detections": detection_data,
            }

            # Add FAIR metadata (image as source, no parent)
            detection_meta = add_fair_metadata(
                detection_meta,
                schema_type="image_detection",
            )

            # Add detector/pose model metadata
            detection_meta["detector"] = {
                "name": detector_cfg.model_name
                if hasattr(detector_cfg, "model_name")
                else "default",
                "confidence_threshold": cfg.detection_threshold,
            }

            # Reorganize for FAIR
            detection_meta = reorganize_for_fair(detection_meta, "image_detection")

            # Write detection sidecar to output_detections_dir
            json_filename = img_path.stem + ".json"
            json_path = output_dir / json_filename
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detection_meta, f, indent=2)

            # Log detection to traceability CSV
            detection_logger.log_image_detection(
                source_image_path=str(img_path.absolute()),
                num_persons=len(detection_data),
                detector_model=str(detector_cfg.model_name)
                if hasattr(detector_cfg, "model_name")
                else "yolox",
                detector_confidence=float(np.mean([d["bbox_confidence"] for d in detection_data])),
                output_path=str(json_path.absolute()),
            )

            logger.info(
                "[%s] Detected %d persons → %s", img_path.name, len(detection_data), json_path.name
            )
            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", img_path.name, e)
            fail_count += 1

    logger.info(
        "Image detection complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )
    detection_logger.print_summary()


if __name__ == "__main__":
    main()
