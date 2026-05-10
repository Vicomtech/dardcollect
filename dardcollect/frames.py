"""
Frame extraction from video clips.

Provides:
  extract_frames  — extract individual PNG frames with FAIR metadata and JSON sidecars.
"""

import json
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from dardcollect.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from dardcollect.pipeline_loggers import FramesExtractionLogger

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    clip_type: str,
    overwrite: bool = False,
    frames_logger: FramesExtractionLogger | None = None,
) -> dict | None:
    """Extract all frames from a video as PNG images with per-frame JSON sidecars.

    Reads detection data from sidecar_path and embeds it in each frame's JSON.
    Resumable: skips frames whose .png and .json already exist unless overwrite=True.

    :param clip_type: String tag embedded in each frame's FAIR metadata
        (e.g. 'person_clip', 'face_crop', 'filtered_face_crop').
    :return: Manifest dict (all frames with UUIDs) or None if skipped/failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sidecar_path.exists():
        logger.warning("No sidecar for %s, skipping", video_path.name)
        return None

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar_data = json.load(f)
    except Exception as e:
        logger.error("Cannot read sidecar %s: %s", sidecar_path.name, e)
        return None

    parent_uuid = sidecar_data.get("uuid")
    parent_file = video_path.name

    frame_data_dict = sidecar_data.get("frame_data", {})

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path.name)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "  %s  %d frames @ %.1f fps",
        video_path.name,
        total_frames,
        fps,
    )

    frame_manifest = {
        "source_video": str(video_path),
        "parent_uuid": parent_uuid,
        "parent_file": parent_file,
        "source_sidecar": sidecar_path.name,
        "clip_type": clip_type,
        "total_frames": total_frames,
        "fps": fps,
        "frames": [],
    }

    frame_count = 0
    pbar = tqdm(total=total_frames, unit="frame", desc=video_path.stem[:40], dynamic_ncols=True)

    try:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_png = output_dir / f"frame_{frame_number:06d}.png"
            frame_json = output_dir / f"frame_{frame_number:06d}.json"

            if frame_png.exists() and frame_json.exists() and not overwrite:
                pbar.update(1)
                frame_number += 1
                continue

            frame_uuid = generate_uuid()

            frame_key = str(frame_number)
            frame_detections = (
                frame_data_dict.get(frame_key, []) if isinstance(frame_data_dict, dict) else []
            )

            frame_meta = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps if fps > 0 else 0.0,
                "detections": frame_detections,
            }

            schema = "face_crop" if "face" in clip_type else "person_clip"
            frame_meta = add_fair_metadata(
                frame_meta,
                schema_type=schema,
                parent_uuid=parent_uuid,
                parent_file=parent_file,
            )
            frame_meta["uuid"] = frame_uuid  # override with frame-specific UUID
            frame_meta = reorganize_for_fair(frame_meta, schema)

            try:
                cv2.imwrite(str(frame_png), frame)
            except Exception as e:
                logger.error("Cannot write frame PNG %s: %s", frame_png.name, e)
                pbar.update(1)
                frame_number += 1
                continue

            try:
                with open(frame_json, "w", encoding="utf-8") as f:
                    json.dump(frame_meta, f, indent=2)
            except Exception as e:
                logger.error("Cannot write frame JSON %s: %s", frame_json.name, e)
                pbar.update(1)
                frame_number += 1
                continue

            frame_manifest["frames"].append(
                {
                    "frame_number": frame_number,
                    "uuid": frame_uuid,
                    "timestamp": frame_meta.get("timestamp", 0.0),
                }
            )

            if frames_logger is not None:
                frames_logger.log_frame_extraction(
                    source_clip_path=str(video_path),
                    frame_number=frame_number,
                    timestamp_seconds=frame_number / fps if fps > 0 else 0.0,
                    output_path=str(frame_png),
                )

            frame_count += 1
            pbar.update(1)
            frame_number += 1

    finally:
        cap.release()
        pbar.close()

    manifest_path = output_dir / "frames_manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(frame_manifest, f, indent=2)
    except Exception as e:
        logger.error("Cannot write manifest %s: %s", manifest_path.name, e)
        return None

    logger.info(
        "  Extracted %d frames → %s",
        frame_count,
        output_dir.name,
    )

    return frame_manifest
