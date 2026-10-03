#!/usr/bin/env python3
"""
Generate ArcFace-region masks for extracted face crops.

Reads face crop sidecar JSON files and draws a binary mask per crop frame:
  255 (white) inside the ArcFace quad (``face_crop_corners_arcface`` — the
  yellow rectangle in the viewer, the 112x112 identity region mapped into the
  616x616 OFIQ crop), 0 (black) everywhere else.

Video crops (mp4) get one mask per annotated output frame
(``<stem>_f<index>_mask.png``); image crops get a single ``<stem>_mask.png``.
No decode is needed — the quad is constant per crop and read from the sidecar.
Source frames get one OFIQ quad mask per detected identity
(``<frame stem>_track<id>_mask.png``).

All parameters are read from config.yaml under the 'face_mask_generation' key.
"""

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

from dardcollect.config import (
    FaceCropConfig,
    FrameExtractionConfig,
    _resolve_path_templates,
    get_log_level,
)
from dardcollect.face_geometry import OFIQ_SIZE
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler

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


def _quad_mask(quad: np.ndarray, h: int, w: int) -> np.ndarray:
    """Filled white region for an OFIQ crop quad, clipped to the frame. Binary {0, 255}.

    The quad itself, which is rotated (OFIQ levels the eyes) and therefore matches
    the face crop video pixel for pixel.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    clipped = quad.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h)

    cv2.fillConvexPoly(mask, clipped, 255)
    return mask


def _generate_crop_quad_masks(frame_path: Path) -> Counter[str]:
    """Write one OFIQ face-crop mask per detected identity in *frame_path*.

    Masks are named ``<frame stem>_track<id>_mask.png``. The request asked for "the same
    filename as the frame", but one frame can hold several people and two files cannot
    share a name, so the track id disambiguates while keeping the correspondence by name.
    """
    sidecar = frame_path.with_suffix(".json")
    detections = _detections_with_faces(sidecar)
    if not detections:
        return Counter({"no_face": 1})

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
                return Counter({"noop": 1})

        quad = _ofiq_quad(det)
        if quad is None:
            continue
        h, w = image.shape[:2]
        mask = _quad_mask(quad, h, w)
        if mask.max() == 0:
            continue
        cv2.imwrite(str(mask_path), mask)
        written += 1

    return Counter({"mask": written} if written else {"noop": 1})


def _arcface_corners(holder: object) -> np.ndarray | None:
    """The holder's ArcFace quad as a (4, 2) float array, or None if absent."""
    if not isinstance(holder, dict):
        return None
    corners = cast(dict[str, Any], holder).get("face_crop_corners_arcface")
    if not (isinstance(corners, list) and len(corners) == 4):
        return None
    try:
        quad = np.array([[float(p[0]), float(p[1])] for p in corners], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        return None
    return quad if quad.shape == (4, 2) else None


def _fill_quad_mask(quad: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary {0, 255} mask with the quad filled white, clipped to the frame."""
    mask = np.zeros((h, w), dtype=np.uint8)
    clipped = quad.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [clipped.astype(np.int32)], 255)
    return mask


def _masks_for_video_crop(crop_path: Path, data: dict) -> Counter[str]:
    """One ArcFace mask per annotated output frame of a video face crop."""
    counts: Counter[str] = Counter()
    size = int(data.get("output_size", OFIQ_SIZE))
    frame_data = data.get("frame_data")
    if not isinstance(frame_data, dict):
        return counts
    for key, entries in frame_data.items():
        if not key.isdigit() or not entries:
            continue
        mask_path = crop_path.parent / f"{crop_path.stem}_f{int(key):06d}_mask.png"
        if mask_path.exists():
            counts["noop"] += 1
            continue
        quad = _arcface_corners(entries[0])
        if quad is None:
            counts["no_face"] += 1
            continue
        cv2.imwrite(str(mask_path), _fill_quad_mask(quad, size, size))
        counts["mask"] += 1
    return counts


def _masks_for_image_crop(crop_path: Path, data: dict) -> Counter[str]:
    """Single ArcFace mask for an image face crop."""
    counts: Counter[str] = Counter()
    mask_path = crop_path.parent / f"{crop_path.stem}_mask.png"
    if mask_path.exists():
        counts["noop"] += 1
        return counts
    quad = _arcface_corners(data)
    if quad is None:
        counts["no_face"] += 1
        return counts
    size = int(data.get("output_size", OFIQ_SIZE))
    cv2.imwrite(str(mask_path), _fill_quad_mask(quad, size, size))
    counts["mask"] += 1
    return counts


def _generate_one_mask(crop_path: Path) -> Counter[str]:
    """Write the ArcFace mask(s) for one video/image crop file; tallies per outcome."""
    try:
        data = json.loads(crop_path.with_suffix(".json").read_text(encoding="utf-8"))
    except Exception:
        return Counter({"noop": 1})
    if not isinstance(data, dict):
        return Counter({"noop": 1})
    if isinstance(data.get("frame_data"), dict):
        return _masks_for_video_crop(crop_path, data)
    if "image_path" in data:
        return _masks_for_image_crop(crop_path, data)
    return Counter({"noop": 1})


def _run_mask_jobs(crop_files: list[Path], modality: str, workers: int) -> Counter[str]:
    """Generate masks for ``crop_files``, threaded when ``workers > 1``.

    Crops are independent — each writes its sibling ``*_mask.png`` file(s) and
    shares no state — so threads are safe. They pay off because the work is
    dominated by per-file latency on network storage, and cv2's
    imread/imwrite release the GIL. Video/image crops get ArcFace masks;
    source frames get the OFIQ quad of each detected identity.
    """
    counts: Counter[str] = Counter()
    desc = f"Generating masks ({modality})"
    make_mask = _generate_one_mask if modality in ("video", "image") else _generate_crop_quad_masks

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for outcome in tqdm(
                pool.map(make_mask, crop_files),
                total=len(crop_files),
                desc=desc,
                unit="crop",
            ):
                counts.update(outcome)
    else:
        for crop_path in tqdm(crop_files, desc=desc, unit="crop"):
            counts.update(make_mask(crop_path))

    return counts


@add_timer
def _resolve_mask_dirs():
    """Resolve crop dirs + workers + mask_type from config (fail loud on bad input)."""
    try:
        import yaml

        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    # Resolve {root}/{output_root}/... placeholders so explicit
    # face_mask_generation dirs work the same as every other section.
    config = _resolve_path_templates(config)
    mask_cfg = config.get("face_mask_generation", {})

    # Explicit dirs in face_mask_generation take precedence over the stage
    # output dirs — e.g. point video_crop_dir at filtered_video_face_crops to
    # mask the quality-filtered set instead of the raw crops.
    try:
        _vcfg = FaceCropConfig.from_yaml(str(CONFIG_PATH), section="face_crop_extraction")
        video_crop_dir = Path(_vcfg.output_dir)
    except Exception:
        video_crop_dir = Path(mask_cfg.get("video_crop_dir", "DARD/video_face_crops"))
    if mask_cfg.get("video_crop_dir"):
        video_crop_dir = Path(str(mask_cfg["video_crop_dir"]))

    try:
        _icfg = FaceCropConfig.from_yaml(str(CONFIG_PATH), section="image_face_crop_extraction")
        image_crop_dir = Path(_icfg.output_dir)
    except Exception:
        image_crop_dir = Path(mask_cfg.get("image_crop_dir", "DARD/image_face_crops"))
    if mask_cfg.get("image_crop_dir"):
        image_crop_dir = Path(str(mask_cfg["image_crop_dir"]))

    try:
        fcfg = FrameExtractionConfig.from_yaml(str(CONFIG_PATH))
        frame_dir = Path(fcfg.output_dir)
    except Exception:
        frame_dir = Path(mask_cfg.get("frame_dir", "DARD/extracted_frames"))
    if mask_cfg.get("frame_dir"):
        frame_dir = Path(str(mask_cfg["frame_dir"]))

    modalities = {"video": video_crop_dir, "image": image_crop_dir, "frames": frame_dir}

    workers = max(1, int(mask_cfg.get("workers", 1) or 1))
    # Mask content is fixed per modality (no selection knob): white = the
    # ArcFace quad (face_crop_corners_arcface, the yellow rectangle in the
    # viewer) in crop space — one mask per annotated video-crop frame, one
    # per image crop. Source frames get one OFIQ quad mask per detected
    # identity. A stale mask_type key in the config fails loud below so it
    # cannot silently select a removed behavior.
    if "mask_type" in mask_cfg:
        logger.error(
            "face_mask_generation.mask_type was removed (masks are always the "
            "ArcFace quad for crops, the OFIQ quad for source frames) — delete "
            "the key from the config",
        )
        sys.exit(1)
    return modalities, workers


def _process_one_modality(modality, crop_dir, workers):
    """Generate masks for one modality dir. Returns (masks, skipped_no_face)."""
    if not crop_dir.exists():
        logger.info("Skipping %s (dir not found): %s", modality, crop_dir)
        return 0, 0

    # Video crops are mp4 (one mask per annotated frame); image crops and
    # source frames are still images.
    extensions = {".jpg", ".jpeg", ".png"}
    if modality == "video":
        extensions = extensions | {".mp4"}
    crop_files = [
        f
        for f in crop_dir.rglob("*")
        if f.suffix.lower() in extensions and not f.name.endswith("_mask.png")
    ]

    if not crop_files:
        logger.info("No face crops found in %s: %s", modality, crop_dir)
        return 0, 0

    modality_workers = min(workers, len(crop_files))
    logger.info(
        "Generating masks for %d %s face crops (workers: %d)",
        len(crop_files),
        modality,
        modality_workers,
    )

    counts = _run_mask_jobs(crop_files, modality, modality_workers)
    return counts["mask"], counts["no_face"]


def main():
    """Main entry point."""
    modalities, workers = _resolve_mask_dirs()

    total_masks = 0
    total_skipped_no_face = 0
    for modality, crop_dir in modalities.items():
        masks, no_face = _process_one_modality(modality, crop_dir, workers)
        total_masks += masks
        total_skipped_no_face += no_face

    logger.info(
        "Summary: Generated %d masks; skipped %d (no usable quad for that frame/crop/detection)",
        total_masks,
        total_skipped_no_face,
    )


if __name__ == "__main__":
    main()
