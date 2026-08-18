#!/usr/bin/env python3
"""
Generate face contour masks for extracted face crops.

Reads face crop images and their sidecar JSON files, extracts the 68 face
landmarks (COCO-133 indices 23-90), and draws a convex hull mask:
  255 (white) inside the face contour, 0 (black) everywhere else.

No GPU required — keypoints are read from existing sidecar JSON files
produced by the extraction pipeline. If sidecar/keypoints are missing for a
crop, no mask file is emitted for that crop.

Supports both video and image modality face crops.

All parameters are read from config.yaml under the 'face_mask_generation' key.
"""

import functools
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

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
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
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

        # Crop sidecars store keypoints at top-level.
        if "keypoints" in data and "keypoint_scores" in data:
            kpts = np.array(data["keypoints"], dtype=np.float32)
            scores = np.array(data["keypoint_scores"], dtype=np.float32)
            if kpts.shape == (133, 2) and scores.shape == (133,):
                return kpts, scores

        # Frame sidecars store detections with keypoints under detections[].
        detections = data.get("detections")
        if isinstance(detections, list) and detections:
            best = max(
                (d for d in detections if isinstance(d, dict)),
                key=lambda d: float(d.get("score", 0.0)),
                default=None,
            )
            if best is not None:
                kpts = np.array(best.get("keypoints", []), dtype=np.float32)
                scores = np.array(best.get("keypoint_scores", []), dtype=np.float32)
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


def _ofiq_quad(det: object) -> np.ndarray | None:
    """The detection's OFIQ face-crop corners, in source-video coordinates.

    ``face_geometry.py`` records these four points when it cuts the identity's face crop
    video, so reusing them makes the mask and the crop the *same* region rather than two
    independent approximations of it — and it needs no tuning parameter. The quad is
    rotated (OFIQ levels the eyes) and covers the whole head, hair included, which boxes
    derived from the face landmarks do not: those stop at the brow line.

    Returns None when the detection produced no face crop. That is the "if a face is
    detected" branch: a subject filmed from behind has a person box and even pose
    keypoints, but no crop and therefore no mask.
    """
    if not isinstance(det, dict):
        return None
    corners = cast(dict[str, Any], det).get("face_crop_corners_ofiq")
    if not (isinstance(corners, list) and len(corners) == 4):
        return None
    try:
        quad = np.array([[float(p[0]), float(p[1])] for p in corners], dtype=np.int32)
    except (TypeError, ValueError, IndexError):
        return None
    return quad


def _detections_with_faces(sidecar_path: Path) -> list[dict]:
    """Detections whose face crop was produced, i.e. the ones a mask can be drawn for."""
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    detections = data.get("detections")
    if not isinstance(detections, list):
        return []
    return [det for det in detections if _ofiq_quad(det) is not None]


def _quad_mask(quad: np.ndarray, h: int, w: int, axis_aligned: bool = False) -> np.ndarray:
    """Filled white region for an OFIQ crop quad, clipped to the frame. Binary {0, 255}.

    With *axis_aligned* the mask is the upright bounding box of the quad — a plain
    rectangle, the literal reading of the request's "bounding box mask". Otherwise it is
    the quad itself, which is rotated (OFIQ levels the eyes) and therefore matches the
    face crop video pixel for pixel. The upright box is the looser of the two: it always
    contains the quad, so it also takes in some background at the corners.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    clipped = quad.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h)

    if axis_aligned:
        x1, y1 = clipped[:, 0].min(), clipped[:, 1].min()
        x2, y2 = clipped[:, 0].max(), clipped[:, 1].max()
        # Slice bounds are inclusive of the quad's extreme pixels: fillConvexPoly paints
        # them, so an exclusive box would not actually contain the quad it bounds.
        if x2 >= x1 and y2 >= y1:
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return mask

    cv2.fillConvexPoly(mask, clipped, 255)
    return mask


def _bbox_mask(bbox: list, h: int, w: int) -> np.ndarray:
    """Filled white rectangle over *bbox*, clipped to the frame. Binary {0, 255}."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = (round(float(v)) for v in bbox)
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _generate_crop_quad_masks(frame_path: Path, axis_aligned: bool = False) -> str:
    """Write one OFIQ face-crop mask per detected identity in *frame_path*.

    Masks are named ``<frame stem>_track<id>_mask.png``. The request asked for "the same
    filename as the frame", but one frame can hold several people and two files cannot
    share a name, so the track id disambiguates while keeping the correspondence by name.
    """
    sidecar = frame_path.with_suffix(".json")
    detections = _detections_with_faces(sidecar)
    if not detections:
        return "no_face"

    written = 0
    image = None
    for det in detections:
        track_id = det.get("track_id", 0)
        mask_path = frame_path.parent / f"{frame_path.stem}_track{int(track_id):03d}_mask.png"
        if mask_path.exists():
            continue

        if image is None:
            # Decode once per frame, and only when a mask actually has to be written.
            image = cv2.imread(str(frame_path))
            if image is None:
                logger.warning("Failed to read image: %s", frame_path.name)
                return "noop"

        quad = _ofiq_quad(det)
        if quad is None:
            continue
        h, w = image.shape[:2]
        mask = _quad_mask(quad, h, w, axis_aligned)
        if mask.max() == 0:
            continue
        cv2.imwrite(str(mask_path), mask)
        written += 1

    return "mask" if written else "noop"


def _generate_one_mask(crop_path: Path) -> str:
    """Write the face mask for one crop.

    Returns the outcome tallied by the caller: ``"mask"`` (written), ``"no_face"``
    (no usable landmarks) or ``"noop"`` (already present, unreadable, or errored).
    """
    mask_path = crop_path.parent / f"{crop_path.stem}_mask.png"
    if mask_path.exists():
        return "noop"

    try:
        # Read the ~300-byte sidecar BEFORE decoding the crop: a frame with
        # no usable keypoints produces no mask, so decoding its PNG first
        # would be pure wasted I/O (crippling over network storage, where
        # a full pass costs tens of GB of reads for zero output).
        kpt_data = _load_keypoints(crop_path.with_suffix(".json"))
        if kpt_data is None:
            return "no_face"

        image = cv2.imread(str(crop_path))
        if image is None:
            logger.warning("Failed to read image: %s", crop_path.name)
            return "noop"

        h, w = image.shape[:2]
        mask = _mask_from_keypoints(kpt_data[0], kpt_data[1], h, w)

        # No usable face landmarks for this crop/frame.
        if mask.max() == 0:
            return "no_face"

        cv2.imwrite(str(mask_path), mask)
        return "mask"

    except Exception as e:
        logger.error("Error processing %s: %s", crop_path.name, e)
        return "noop"


def _run_mask_jobs(
    crop_files: list[Path], modality: str, workers: int, mask_type: str = "face_hull"
) -> Counter[str]:
    """Generate masks for ``crop_files``, threaded when ``workers > 1``.

    Crops are independent — each writes one sibling ``*_mask.png`` and shares no
    state — so threads are safe. They pay off because the work is dominated by
    per-file latency on network storage, and cv2's imread/imwrite release the GIL.
    """
    counts: Counter[str] = Counter()
    desc = f"Generating masks ({modality})"
    if mask_type == "face_hull":
        make_mask = _generate_one_mask
    else:
        upright = mask_type == "ofiq_crop_bbox"
        make_mask = functools.partial(_generate_crop_quad_masks, axis_aligned=upright)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for outcome in tqdm(
                pool.map(make_mask, crop_files),
                total=len(crop_files),
                desc=desc,
                unit="crop",
            ):
                counts[outcome] += 1
    else:
        for crop_path in tqdm(crop_files, desc=desc, unit="crop"):
            counts[make_mask(crop_path)] += 1

    return counts


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

    workers = max(1, int(mask_cfg.get("workers", 1) or 1))
    # The two "ofiq_crop_*" modes implement the video pre-processing request: one filled
    # region per detected identity, derived from the OFIQ face crop cut for that identity
    # (whole head, hair included, no tuning parameter).
    #   ofiq_crop_bbox — upright bounding box of the crop quad; a plain rectangle.
    #   ofiq_crop_quad — the rotated quad itself; matches the crop video pixel for pixel.
    # Default stays "face_hull" (convex hull of face landmarks 23-90) so existing configs
    # are unchanged.
    mask_type = str(mask_cfg.get("mask_type", "face_hull"))
    if mask_type not in {"face_hull", "ofiq_crop_quad", "ofiq_crop_bbox"}:
        logger.error("Unknown face_mask_generation.mask_type: %s", mask_type)
        sys.exit(1)

    total_masks = 0
    total_skipped_no_face = 0
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

        modality_workers = min(workers, len(crop_files))
        logger.info(
            "Generating masks for %d %s face crops (workers: %d, mask_type: %s)",
            len(crop_files),
            modality,
            modality_workers,
            mask_type,
        )

        counts = _run_mask_jobs(crop_files, modality, modality_workers, mask_type)
        total_masks += counts["mask"]
        total_skipped_no_face += counts["no_face"]

    reason = (
        "missing/invalid face keypoints"
        if mask_type == "face_hull"
        else "no OFIQ face crop for that detection"
    )
    logger.info(
        "Summary: Generated %d masks; skipped %d (%s)",
        total_masks,
        total_skipped_no_face,
        reason,
    )


if __name__ == "__main__":
    main()
