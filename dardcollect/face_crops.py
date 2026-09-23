"""
Face crop extraction functions for images and video clips.

Provides:
  process_image  — extract OFIQ crops from a static image using its detection JSON.
  process_video  — extract OFIQ crop videos from a person-clip video using its sidecar JSON.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dardcollect.config import FaceCropConfig
from dardcollect.face_crop_writers import _CropWriteContext, _write_track_crop
from dardcollect.face_geometry import (
    ARCFACE_CROP_CORNERS_IN_OFIQ,
    OFIQ_SIZE,
    _bbox_iou,
    _corners_to_warp,
    _get_or_compute_corners,
    _transform_keypoints,
)
from dardcollect.fair import add_fair_metadata, reorganize_for_fair, validate_against_schema
from dardcollect.modality_loggers import ImageFaceCropsExtractionLogger
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger
from dardcollect.pipeline_utils import (
    make_tqdm,
)
from dardcollect.provenance import now_iso

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


def process_image(
    image_path: Path,
    detection_json_path: Path,
    face_config: FaceCropConfig,
    output_dir: Path,
    logger_instance: ImageFaceCropsExtractionLogger | None = None,
) -> int:
    """Extract 616×616 OFIQ face crop images from a single source image.

    Reads pre-computed detections from detection_json_path (written by
    extract_persons_from_images.py). Skips persons without a visible face.

    Returns the number of crop images written.
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
            transformed_kpts, transformed_scores, _ = _transform_keypoints(
                keypoints, keypoint_scores, kpts_array, scores_array
            )
        else:
            transformed_kpts, transformed_scores = [], []

        stem = f"{image_path.stem}_face_{person_idx}"
        sidecar_meta = {
            "image_path": image_path.as_posix(),
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
            "face_crop_corners_arcface": arcface_corners_json,
            "extracted_at": now_iso(),
        }

        sidecar_meta = add_fair_metadata(
            sidecar_meta,
            schema_type="face_crop",
            parent_uuid=detection_data.get("uuid", ""),
            parent_file=detection_json_path.name,
        )
        sidecar_meta = reorganize_for_fair(sidecar_meta)

        ofiq_crop_bgr = cv2.cvtColor(ofiq_crop, cv2.COLOR_RGB2BGR)
        crop_path = output_dir / f"{stem}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(crop_path), ofiq_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.warning("Failed to write crop: %s", crop_path)
            continue

        # Validate the FAIR sidecar against the ratified schema before write
        # (per the project's "validate at write" contract).
        validate_against_schema(sidecar_meta, "face_crop")
        json_path = crop_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_meta, f, indent=2)

        if logger_instance:
            bbox = det.get("bbox_tlbr", [None, None, None, None])
            bbox_in_source = f"{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}"
            logger_instance.log_face_crop_extraction(
                source_image_path=str(image_path.absolute()),
                bbox_in_source=bbox_in_source,
                bbox_confidence=float(det.get("bbox_confidence", 0.0)),
                output_path=str(crop_path.absolute()),
            )

        logger.debug("  Wrote crop: %s", crop_path.name)
        written += 1

    return written


def _collect_detection_frames(
    detections: list[dict],
    frame: np.ndarray,
    frame_id: int,
    frame_data_orig: dict,
    start_frame: int,
    face_config: "FaceCropConfig",
    track_frames: dict,
    track_corners: dict,
) -> None:
    """Collect one decoded frame's detections into per-track frame/corner lists.

    Stabilization (issue #9, opt-in): when ``stabilize_face_crops`` is on, the
    SOURCE frame is stored and rendering happens once at write time through the
    track-median quad; default OFF renders per-frame here (unchanged behavior).
    """
    abs_frame = start_frame + frame_id
    detections = frame_data_orig.get(str(abs_frame), detections)

    frame_bboxes = [(d["track_id"], d["bbox"]) for d in detections]

    for det in detections:
        tid = det["track_id"]
        bbox = det["bbox"]

        corners = _get_or_compute_corners(det, face_config)
        track_corners[tid].append(corners)
        if corners is None:
            track_frames[tid].append((frame_id, None))
            continue

        overlapping = any(
            _bbox_iou(bbox, ob) > face_config.max_overlap_iou
            for oid, ob in frame_bboxes
            if oid != tid
        )
        if overlapping:
            track_frames[tid].append((frame_id, None))
            continue

        if face_config.stabilize_face_crops:
            track_frames[tid].append((frame_id, frame))
        else:
            ofiq_crop = _corners_to_warp(frame, corners, OFIQ_SIZE)
            track_frames[tid].append((frame_id, ofiq_crop))


def process_video(
    video_path: Path,
    face_config: FaceCropConfig,
    face_crops_logger: FaceCropsExtractionLogger | None = None,
    encoding: "EncodingConfig | None" = None,
) -> int:
    """Extract 616×616 OFIQ face crop videos from a single person-clip video.

    Reads pre-computed smoothed keypoints and face crop corners from the clip's
    sidecar JSON (written by extract_person_clips_from_videos.py), so no
    re-detection is needed. Produces one .mp4 + .json pair per track.

    Skips tracks with fewer than face_config.min_track_face_frames valid frames.
    Returns the number of crop videos written.
    """
    output_dir = Path(face_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    done_sentinel = output_dir / f"{video_path.stem}.done"
    if done_sentinel.exists():
        logger.info("  SKIP (already done): %s", video_path.name)
        return 0

    json_path = video_path.with_suffix(".json")
    if not json_path.exists():
        logger.error("  No sidecar JSON for %s — skipping", video_path.name)
        return 0

    with open(json_path, encoding="utf-8") as f:
        clip_data = json.load(f)

    start_frame: int = clip_data.get("start_frame", 0)
    frame_data_orig: dict = clip_data.get("frame_data", {})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "  %dx%d  %.1f fps  %d frames  (%.1fs)",
        width,
        height,
        fps,
        total_frames,
        total_frames / fps if fps > 0 else 0,
    )

    # track_id → [(relative_frame_idx, ofiq_crop_or_None), ...]
    track_frames: dict[int, list[tuple[int, np.ndarray | None]]] = defaultdict(list)
    # track_id → [corner arrays or None] (parallel; stabilization, issue #9)
    track_corners: dict[int, list[np.ndarray | None]] = defaultdict(list)

    frame_id = 0

    pbar = make_tqdm(total=total_frames, unit="fr", desc=video_path.name[:40], dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _collect_detection_frames(
            [],
            frame,
            frame_id,
            frame_data_orig,
            start_frame,
            face_config,
            track_frames,
            track_corners,
        )

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write one video per track ─────────────────────────────────────────────
    written = 0
    ctx = _CropWriteContext(
        video_path=video_path,
        clip_data=clip_data,
        frame_data_orig=frame_data_orig,
        start_frame=start_frame,
        face_config=face_config,
        output_dir=output_dir,
        fps=fps,
        encoding=encoding,
        face_crops_logger=face_crops_logger,
        track_corners=track_corners,
        black_ofiq=np.zeros((OFIQ_SIZE, OFIQ_SIZE, 3), dtype=np.uint8),
        arcface_corners_json=[
            [round(float(x), 2), round(float(y), 2)] for x, y in ARCFACE_CROP_CORNERS_IN_OFIQ
        ],
    )

    for tid, frames in track_frames.items():
        valid_frames = [(fid, oc) for fid, oc in frames if oc is not None]
        if _write_track_crop(ctx, tid, frames, valid_frames):
            written += 1

    return written
