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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dardcollect import PersonDetector, PoseEstimator
from dardcollect.config import DetectorConfig, FaceCropConfig, ImageExtractionConfig, get_log_level
from dardcollect.face_geometry import face_crop_corners
from dardcollect.fair import (
    add_fair_metadata,
    generate_uuid,
    reorganize_for_fair,
    validate_against_schema,
)
from dardcollect.gpu_setup import setup_gpu_paths
from dardcollect.modality_loggers import ImagePersonDetectionLogger
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import (
    _TqdmHandler,
    check_frontal_face,
)
from dardcollect.provenance import now_iso

# Setup GPU paths BEFORE importing heavy libraries
CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)
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


def _init_models(detector_cfg: DetectorConfig, models_dir: Path):
    """Load + initialize the YOLOX detector + CIGPose poser. sys.exit on failure."""
    det_model_path = models_dir / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_model_path = models_dir / "cigpose-m_coco-wholebody_256x192.onnx"
    if not det_model_path.exists():
        logger.error("Detection model not found: %s", det_model_path)
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)
    if not pose_model_path.exists():
        logger.error("Pose model not found: %s", pose_model_path)
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)
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
    return detector, poser


def _scan_pending_images(input_dir: Path, output_dir: Path, overwrite: bool) -> list[Path]:
    """Return image files in input_dir that don't yet have a detection sidecar."""
    image_files: list[Path] = []
    for img_path in sorted(input_dir.rglob("*")):
        if img_path.is_dir():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        json_path = output_dir / (img_path.stem + ".json")
        if json_path.exists() and not overwrite:
            continue
        image_files.append(img_path)
    return image_files


@dataclass
class _ImageStageContext:
    """Per-run objects for image detection, bundled to keep the loop helper low-arity."""

    detector: PersonDetector
    poser: PoseEstimator
    cfg: ImageExtractionConfig
    detector_cfg: DetectorConfig
    face_crop_cfg: FaceCropConfig
    detection_logger: ImagePersonDetectionLogger
    output_dir: Path


def _pose_detection_entry(
    bbox,
    det_idx: int,
    det_score: float,
    image_rgb,
    ctx: _ImageStageContext,
) -> dict | None:
    """Estimate pose + face corners for one person; None if pose has no keypoints."""
    cfg = ctx.cfg
    face_crop_cfg = ctx.face_crop_cfg
    try:
        keypoints, keypoints_scores = ctx.poser.get_keypoints(image_rgb, bbox)
        if keypoints is None or len(keypoints) == 0:
            return None

        # Attempt face crop corners directly — face_crop_corners checks eye scores
        # and min inter-eye distance, which is the correct usability predicate.
        # check_face_visibility's size-based gate fails on full-body archive photos
        # where the face is small relative to the full-height bounding box.
        corners_by_mode: dict[str, np.ndarray | None] = {}
        for mode in ("arcface", "ofiq"):
            try:
                corners_by_mode[mode] = face_crop_corners(
                    keypoints,
                    keypoints_scores,
                    mode,
                    face_crop_cfg.pose_keypoint_threshold,
                    face_crop_cfg.min_eye_distance_px,
                )
            except Exception:
                corners_by_mode[mode] = None
        arcface = corners_by_mode["arcface"]
        ofiq = corners_by_mode["ofiq"]

        # face_visible iff corners computable (eyes detected + sufficient distance)
        face_visible = ofiq is not None
        frontal = (
            check_frontal_face(keypoints, keypoints_scores, cfg.frontal_symmetry_threshold)
            if face_visible
            else False
        )
        return {
            "person_idx": det_idx,
            "bbox_tlbr": bbox.tolist(),
            "bbox_confidence": float(det_score),
            "keypoints": keypoints.tolist(),
            "keypoint_scores": keypoints_scores.tolist(),
            "face_visible": bool(face_visible),
            "frontal_face": bool(frontal),
            "face_crop_corners_arcface": arcface.tolist() if arcface is not None else None,
            "face_crop_corners_ofiq": ofiq.tolist() if ofiq is not None else None,
        }
    except Exception as e:
        logger.debug("Failed to estimate pose for person %d: %s", det_idx, e)
        return None


def _detect_persons_in_image(
    image_rgb,
    ctx: _ImageStageContext,
) -> list[dict]:
    """Run detection + pose over one image; returns per-person detection entries.

    Empty list means "no usable person" (logged at debug); the caller counts it
    as a failure. ``detection_threshold``/``frontal_symmetry_threshold`` and the
    image face-crop thresholds come from the image config section.
    """
    det_bboxes, det_scores = ctx.detector.get_detections(image_rgb, ctx.cfg.detection_threshold)
    logger.debug(
        f"Detections: {len(det_bboxes)} persons with scores "
        f"{det_scores[:5] if len(det_scores) > 0 else []}"
    )

    detection_data: list[dict] = []
    for det_idx, bbox in enumerate(det_bboxes):
        entry = _pose_detection_entry(bbox, det_idx, det_scores[det_idx], image_rgb, ctx)
        if entry is not None:
            detection_data.append(entry)
    return detection_data


def _write_image_detection_sidecar(
    img_path: Path,
    image_width: int,
    image_height: int,
    detection_data: list[dict],
    ctx: _ImageStageContext,
) -> Path:
    """Write + validate the FAIR image-detection sidecar; returns its path."""
    detection_meta = {
        "uuid": str(generate_uuid()),
        "image_path": img_path.as_posix(),
        "image_size": {"width": image_width, "height": image_height},
        "detection_timestamp": now_iso(),
        "num_persons": len(detection_data),
        "detections": detection_data,
    }
    detection_meta = add_fair_metadata(detection_meta, schema_type="image_detection")
    detection_meta["detector"] = {
        "name": ctx.detector_cfg.model_name
        if hasattr(ctx.detector_cfg, "model_name")
        else "default",
        "confidence_threshold": ctx.cfg.detection_threshold,
    }
    detection_meta = reorganize_for_fair(detection_meta)

    json_path = ctx.output_dir / (img_path.stem + ".json")
    # Validate the FAIR sidecar against the ratified schema before write
    # (per the project's "validate at write" contract).
    validate_against_schema(detection_meta, "image_detection")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detection_meta, f, indent=2)

    ctx.detection_logger.log_image_detection(
        source_image_path=str(img_path.absolute()),
        num_persons=len(detection_data),
        detector_model=str(ctx.detector_cfg.model_name)
        if hasattr(ctx.detector_cfg, "model_name")
        else "yolox",
        detector_confidence=float(np.mean([d["bbox_confidence"] for d in detection_data])),
        output_path=str(json_path.absolute()),
    )
    return json_path


def _process_one_image(img_path: Path, ctx: _ImageStageContext) -> bool:
    """Detect + write one image's sidecar; True on success, False on any skip/error."""
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read image: %s", img_path.name)
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_rgb.shape[:2]
        logger.debug(
            f"Processing {img_path.name} ({image_width}x{image_height}), "
            f"threshold={ctx.cfg.detection_threshold}"
        )

        detection_data = _detect_persons_in_image(image_rgb, ctx)
        if not detection_data:
            logger.debug("No valid detections with pose in %s", img_path.name)
            return False

        json_path = _write_image_detection_sidecar(
            img_path, image_width, image_height, detection_data, ctx
        )
        logger.info(
            "[%s] Detected %d persons → %s", img_path.name, len(detection_data), json_path.name
        )
        return True
    except Exception as e:
        logger.error("Failed to process %s: %s", img_path.name, e)
        return False


@add_timer
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

    # Load configuration for detector and pose models. Face-crop thresholds
    # (pose_keypoint_threshold, min_eye_distance_px) are read from the IMAGE
    # face-crop section — this is an image stage, so it must not hard-depend on
    # the video face_crop_extraction section (which a lean image-only config
    # omits). The fixture's image and video sections carry identical threshold
    # values, so this is behavior-preserving for the fixture.
    detector_cfg = DetectorConfig.from_yaml(str(CONFIG_PATH))
    face_crop_cfg = FaceCropConfig.from_yaml(str(CONFIG_PATH), section="image_face_crop_extraction")
    models_dir = Path(detector_cfg.models_path)

    # Verify and load detection and pose models
    detector, poser = _init_models(detector_cfg, models_dir)

    # Initialize traceability logger
    downloads_csv = input_dir.parent / "downloads.csv"
    detection_logger = ImagePersonDetectionLogger(
        output_dir=str(output_dir), downloads_csv_path=downloads_csv
    )

    # Find image files needing detection
    logger.info("Scanning for images needing detection...")
    image_files = _scan_pending_images(input_dir, output_dir, cfg.overwrite)
    logger.info("Found %d images needing detection in %s", len(image_files), input_dir)

    if not image_files:
        logger.info("All images processed! Nothing to do.")
        return

    ctx = _ImageStageContext(
        detector=detector,
        poser=poser,
        cfg=cfg,
        detector_cfg=detector_cfg,
        face_crop_cfg=face_crop_cfg,
        detection_logger=detection_logger,
        output_dir=output_dir,
    )

    success_count = 0
    fail_count = 0
    for img_path in tqdm(image_files, desc="Detecting persons in images", unit="image"):
        if _process_one_image(img_path, ctx):
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        "Image detection complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )
    detection_logger.print_summary()


if __name__ == "__main__":
    main()
