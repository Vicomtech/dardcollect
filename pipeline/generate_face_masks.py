#!/usr/bin/env python3
"""
Generate face contour masks for extracted face crops.

Reads face crop images and their sidecar JSON files, extracts the 68 face
landmarks (COCO-133 indices 23-90), and draws a convex hull mask:
  255 (white) inside the face contour, 0 (black) everywhere else.

No GPU required — keypoints are read from existing sidecar JSON files
produced by the extraction pipeline. Falls back to a bounding box mask
if no sidecar or face keypoints are available.

Supports both video and image modality face crops.

All parameters are read from config.yaml under the 'face_mask_generation' key.
"""

import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dardcollect.config import FaceCropConfig, FrameExtractionConfig, get_log_level
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import FACE_LANDMARK_INDICES, _TqdmHandler

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Configuration path
CONFIG_PATH = Path(
    os.environ.get("DARDCOLLECT_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
)

logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

_KPT_SCORE_THRESHOLD = 0.3  # Minimum confidence to include a face landmark


def _load_keypoints(sidecar_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load 133 keypoints + scores from a face crop sidecar JSON.

    Returns:
        (keypoints (133, 2), scores (133,)) or None if sidecar is missing/invalid.
    """
    if not sidecar_path.exists():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        kpts = np.array(data["keypoints"], dtype=np.float32)  # (133, 2)
        scores = np.array(data["keypoint_scores"], dtype=np.float32)  # (133,)
        if kpts.shape == (133, 2) and scores.shape == (133,):
            return kpts, scores
    except Exception:
        pass
    return None


def _mask_from_keypoints(
    kpts: np.ndarray,
    scores: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """Draw convex hull mask from face landmark keypoints (COCO-133 indices 23-90).

    Args:
        kpts: (133, 2) keypoint coordinates in crop image space.
        scores: (133,) confidence scores.
        h: Image height.
        w: Image width.

    Returns:
        Binary mask (H, W) uint8 {0, 255}.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Collect face landmark points with sufficient confidence
    pts = [
        kpts[i].astype(np.int32) for i in FACE_LANDMARK_INDICES if scores[i] >= _KPT_SCORE_THRESHOLD
    ]

    if len(pts) < 3:
        return mask  # Not enough points for a hull

    hull = cv2.convexHull(np.array(pts))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _mask_from_bbox(image_shape: tuple[int, int]) -> np.ndarray:
    """Fallback: white rectangle covering the entire crop."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # Use 80% of the crop center as a conservative face estimate
    margin_y = int(h * 0.1)
    margin_x = int(w * 0.1)
    mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255
    return mask


@add_timer
def main():
    """Main entry point."""
    try:
        import yaml

        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    mask_cfg = config.get("face_mask_generation", {})

    # Read crop dirs from the pipeline config sections (inherits test overrides)
    try:
        _vcfg = FaceCropConfig.from_yaml(str(CONFIG_PATH), section="face_crop_extraction")
        video_crop_dir = Path(_vcfg.output_dir)
    except Exception:
        video_crop_dir = Path(mask_cfg.get("video_crop_dir", "DARD/video_face_crops"))

    try:
        _icfg = FaceCropConfig.from_yaml(str(CONFIG_PATH), section="image_face_crop_extraction")
        image_crop_dir = Path(_icfg.output_dir)
    except Exception:
        image_crop_dir = Path(mask_cfg.get("image_crop_dir", "DARD/image_face_crops"))

    try:
        fcfg = FrameExtractionConfig.from_yaml(str(CONFIG_PATH))
        frame_dir = Path(fcfg.output_dir)
    except Exception:
        frame_dir = Path(mask_cfg.get("frame_dir", "DARD/extracted_frames"))

    modalities = {"video": video_crop_dir, "image": image_crop_dir, "frames": frame_dir}

    total_masks = 0
    total_fallback = 0
    for modality, crop_dir in modalities.items():
        if not crop_dir.exists():
            logger.info("Skipping %s (dir not found): %s", modality, crop_dir)
            continue

        image_extensions = {".jpg", ".jpeg", ".png"}
        crop_files = [
            f
            for f in crop_dir.rglob("*")
            if f.suffix.lower() in image_extensions and not f.name.endswith("_mask.png")
        ]

        if not crop_files:
            logger.info("No face crops found in %s: %s", modality, crop_dir)
            continue

        logger.info("Generating masks for %d %s face crops", len(crop_files), modality)

        for crop_path in tqdm(crop_files, desc=f"Generating masks ({modality})", unit="crop"):
            mask_path = crop_path.parent / f"{crop_path.stem}_mask.png"
            if mask_path.exists():
                continue

            try:
                image = cv2.imread(str(crop_path))
                if image is None:
                    logger.warning("Failed to read image: %s", crop_path.name)
                    continue

                h, w = image.shape[:2]
                sidecar_path = crop_path.with_suffix(".json")
                kpt_data = _load_keypoints(sidecar_path)

                if kpt_data is not None:
                    mask = _mask_from_keypoints(kpt_data[0], kpt_data[1], h, w)
                    if mask.max() == 0:  # All black — hull failed
                        mask = _mask_from_bbox((h, w))
                        total_fallback += 1
                else:
                    mask = _mask_from_bbox((h, w))
                    total_fallback += 1

                cv2.imwrite(str(mask_path), mask)
                total_masks += 1

            except Exception as e:
                logger.error("Error processing %s: %s", crop_path.name, e)

    logger.info(
        "Summary: Generated %d masks (%d keypoint-based, %d fallback bbox)",
        total_masks,
        total_masks - total_fallback,
        total_fallback,
    )


if __name__ == "__main__":
    main()
