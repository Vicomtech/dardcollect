"""
Face crop extraction functions for images and video clips.

Provides:
  process_image  — extract OFIQ crops from a static image using its detection JSON.
  process_video  — extract OFIQ crop videos from a person-clip video using its sidecar JSON.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from dardcollect.audio import _mux_audio
from dardcollect.config import FaceCropConfig
from dardcollect.face_geometry import (
    ARCFACE_CROP_CORNERS_IN_OFIQ,
    OFIQ_SIZE,
    _bbox_iou,
    _corners_to_warp,
    _get_or_compute_corners,
    _transform_bbox,
    _transform_keypoints,
)
from dardcollect.fair import add_fair_metadata, reorganize_for_fair
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger, ImageFaceCropsExtractionLogger
from dardcollect.pipeline_utils import _cleanup_files, _write_video_with_moviepy, check_disk_space
from dardcollect.provenance import now_iso

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
                keypoints, keypoint_scores, kpts_array, scores_array, OFIQ_SIZE
            )
        else:
            transformed_kpts, transformed_scores = [], []

        stem = f"{image_path.stem}_face_{person_idx}"
        sidecar_meta = {
            # UUID from parent image detection; will be replaced by add_fair_metadata
            "uuid": detection_data.get("uuid", ""),
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
            "face_crop_corners_arcface": arcface_corners_json,
            "extracted_at": now_iso(),
        }

        sidecar_meta = add_fair_metadata(
            sidecar_meta,
            schema_type="face_crop",
            parent_uuid=detection_data.get("uuid", ""),
        )
        sidecar_meta = reorganize_for_fair(sidecar_meta, "face_crop")

        ofiq_crop_bgr = cv2.cvtColor(ofiq_crop, cv2.COLOR_RGB2BGR)
        crop_path = output_dir / f"{stem}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(crop_path), ofiq_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            logger.warning("Failed to write crop: %s", crop_path)
            continue

        json_path = crop_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_meta, f, indent=2)

        if logger_instance:
            bbox = det.get("bbox_tlbr", [None, None, None, None])
            face_bbox = f"{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}"
            logger_instance.log_face_crop_extraction(
                source_image_path=str(image_path.absolute()),
                face_bbox=face_bbox,
                confidence=float(det.get("bbox_confidence", 0.0)),
                output_path=str(crop_path.absolute()),
            )

        logger.debug("  Wrote crop: %s", crop_path.name)
        written += 1

    return written


def process_video(
    video_path: Path,
    face_config: FaceCropConfig,
    face_crops_logger: FaceCropsExtractionLogger | None = None,
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

    frame_id = 0
    from tqdm import tqdm

    pbar = tqdm(total=total_frames, unit="fr", desc=video_path.name[:40], dynamic_ncols=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        abs_frame = start_frame + frame_id
        detections = frame_data_orig.get(str(abs_frame), [])

        frame_bboxes = [(d["track_id"], d["bbox"]) for d in detections]

        for det in detections:
            tid = det["track_id"]
            bbox = det["bbox"]

            # Compute or get corners from keypoints
            corners = _get_or_compute_corners(det, face_config)
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

            ofiq_crop = _corners_to_warp(frame, corners, OFIQ_SIZE)
            track_frames[tid].append((frame_id, ofiq_crop))

        frame_id += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # ── Write one video per track ─────────────────────────────────────────────
    written = 0
    black_ofiq = np.zeros((OFIQ_SIZE, OFIQ_SIZE, 3), dtype=np.uint8)

    arcface_corners_json = [
        [round(float(x), 2), round(float(y), 2)] for x, y in ARCFACE_CROP_CORNERS_IN_OFIQ
    ]

    def _is_track_complete(ofiq_path: Path) -> bool:
        return ofiq_path.exists() and ofiq_path.with_suffix(".json").exists()

    for tid, frames in track_frames.items():
        valid_frames = [(fid, oc) for fid, oc in frames if oc is not None]
        if len(valid_frames) < face_config.min_track_face_frames:
            logger.debug(
                "  Track %d: only %d valid face frame(s), skipping",
                tid,
                len(valid_frames),
            )
            continue

        stem = f"{video_path.stem}_face_{tid}"
        ofiq_path = output_dir / f"{stem}.mp4"
        ofiq_sidecar = ofiq_path.with_suffix(".json")

        if _is_track_complete(ofiq_path):
            logger.info("  SKIP (already complete): %s", stem)
            continue

        if ofiq_path.exists():
            logger.info("  Incomplete write detected, cleaning up: %s", stem)
            _cleanup_files(ofiq_path, ofiq_sidecar)

        check_disk_space(output_dir, face_config.min_free_disk_gb)

        frames_to_write = []
        frame_data = {}  # output frame index → detection list (same schema as person clip sidecars)
        if face_config.skip_no_face_frames:
            output_frame_idx = 0
            for fid, oc in valid_frames:
                if oc is not None:
                    frames_to_write.append(oc)
                    # Extract keypoints from original frame data
                    abs_frame = start_frame + fid
                    detections = frame_data_orig.get(str(abs_frame), [])
                    for det in detections:
                        if det.get("track_id") == tid:
                            kpts = det.get("keypoints", [])
                            scores = det.get("keypoint_scores", [])
                            corners = _get_or_compute_corners(det, face_config)
                            if corners is not None and kpts and scores:
                                kpts_array = np.array(kpts, dtype=np.float32)
                                scores_array = np.array(scores, dtype=np.float32)
                                transformed_kpts, _, M = _transform_keypoints(
                                    kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                                )
                                entry: dict = {
                                    "track_id": tid,
                                    "score": det.get("score"),
                                    "keypoints": transformed_kpts,
                                    "keypoint_scores": scores,
                                    "face_crop_corners_arcface": arcface_corners_json,
                                }
                                if M is not None and det.get("bbox"):
                                    entry["bbox"] = _transform_bbox(det["bbox"], M)
                                frame_data[str(output_frame_idx)] = [entry]
                            break
                    output_frame_idx += 1
        else:
            first_fid = frames[0][0]
            last_fid = frames[-1][0]
            ofiq_dict = {fid: oc for fid, oc in frames if oc is not None}
            last_ofiq = black_ofiq
            output_frame_idx = 0
            for fid in range(first_fid, last_fid + 1):
                if fid in ofiq_dict:
                    last_ofiq = ofiq_dict[fid]
                frames_to_write.append(last_ofiq)
                abs_frame = start_frame + fid
                detections = frame_data_orig.get(str(abs_frame), [])
                for det in detections:
                    if det.get("track_id") == tid:
                        kpts = det.get("keypoints", [])
                        scores = det.get("keypoint_scores", [])
                        corners = _get_or_compute_corners(det, face_config)
                        if corners is not None and kpts and scores:
                            kpts_array = np.array(kpts, dtype=np.float32)
                            scores_array = np.array(scores, dtype=np.float32)
                            transformed_kpts, _, M = _transform_keypoints(
                                kpts, scores, kpts_array, scores_array, OFIQ_SIZE
                            )
                            entry = {
                                "track_id": tid,
                                "score": det.get("score"),
                                "keypoints": transformed_kpts,
                                "keypoint_scores": scores,
                                "face_crop_corners_arcface": arcface_corners_json,
                            }
                            if M is not None and det.get("bbox"):
                                entry["bbox"] = _transform_bbox(det["bbox"], M)
                            frame_data[str(output_frame_idx)] = [entry]
                        break
                output_frame_idx += 1

        # Write video using moviepy
        success = _write_video_with_moviepy(frames_to_write, ofiq_path, fps)

        if not success:
            logger.error("Failed to write video for %s — stopping.", stem)
            _cleanup_files(ofiq_path)
            sys.exit(1)

        first_fid, last_fid = frames[0][0], frames[-1][0]
        n_valid = len(valid_frames)
        n_output_frames = last_fid - first_fid + 1
        duration_seconds = round(n_output_frames / fps, 3) if fps > 0 else 0

        if face_config.include_audio and not face_config.skip_no_face_frames:
            start_t = first_fid / fps
            end_t = (last_fid + 1) / fps
            _mux_audio(video_path, ofiq_path, start_t, end_t)

        meta = {
            "source_video": str(video_path),
            "track_id": tid,
            "start_frame": 0,
            "end_frame": n_output_frames - 1,
            "start_seconds": 0.0,
            "end_seconds": duration_seconds,
            "duration_seconds": duration_seconds,
            "video_info": {
                "fps": round(fps, 3),
                "width": OFIQ_SIZE,
                "height": OFIQ_SIZE,
                "duration_seconds": duration_seconds,
            },
            "valid_face_frames": n_valid,
            "crop_format": "ofiq",
            "output_size": OFIQ_SIZE,
            "frame_data": frame_data,
        }

        parent_clip_uuid = clip_data.get("uuid")
        parent_clip_file = video_path.name
        meta = add_fair_metadata(
            meta,
            schema_type="face_crop",
            parent_uuid=parent_clip_uuid,
            parent_file=parent_clip_file,
        )

        try:
            meta = reorganize_for_fair(meta, "face_crop")
            with open(ofiq_sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            logger.error("Cannot write %s (%s) — stopping.", ofiq_sidecar.name, e)
            _cleanup_files(ofiq_path, ofiq_sidecar)
            sys.exit(1)

        if face_crops_logger is not None:
            # Approximate confidence from the first frame's first detection
            avg_confidence = 0.5
            if frame_data and frame_data.get("0"):
                detections = frame_data.get("0", [])
                if detections and isinstance(detections[0], dict):
                    score = detections[0].get("score", 0.5)
                    avg_confidence = float(score) if score else 0.5

            face_crops_logger.log_face_crop_extraction(
                source_type="person_clip",
                source_path=str(video_path),
                face_bbox=f"0,0,{OFIQ_SIZE},{OFIQ_SIZE}",  # Full OFIQ frame
                confidence=avg_confidence,
                output_path=str(ofiq_path),
            )

        logger.info(
            "  Wrote %s  (%d valid face frames / %d span frames)",
            stem,
            n_valid,
            last_fid - first_fid + 1,
        )
        written += 1

    return written
