#!/usr/bin/env python3
"""
Extract video frames as PNG images with FAIR-compliant metadata.

Converts video files (from extract_person_clips_from_videos.py,
extract_face_crops_from_videos.py, or filter_face_crops_by_quality.py) into
frame sequences with per-frame JSON sidecars and a frames_manifest.json for
discovery.

Each frame gets:
- frame_XXXXXX.png (zero-padded 6-digit frame number)
- frame_XXXXXX.json (frame metadata with UUID, parent reference, detection data)

Manifest JSON lists all frames with their UUIDs for batch discovery.

Usage:
  python scripts/extract_frames_from_videos.py \
    --input-dir DARD/extracted_person_clips \
    --output-dir DARD/extracted_frames/person_clips \
    --type person_clip

All parameters are read from config.yaml under the 'frame_extraction' key.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from persondet.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from persondet.pipeline_loggers import FramesExtractionLogger
from persondet.script_utilities import _TqdmHandler

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Path to config file
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction."""

    input_dir: str
    output_dir: str
    overwrite: bool = False

    @staticmethod
    def _infer_type_from_folder(input_dir: str) -> str:
        """Infer clip type from folder name."""
        folder_name = Path(input_dir).name.lower()
        if "filtered" in folder_name:
            return "filtered_face_crop"
        if "face" in folder_name:
            return "face_crop"
        return "person_clip"  # default

    def get_type(self) -> str:
        """Get inferred clip type from input_dir folder name."""
        return self._infer_type_from_folder(self.input_dir)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FrameExtractionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        frame_config = config.get("frame_extraction", {})
        return cls(
            input_dir=frame_config.get("input_dir", "DARD/extracted_person_clips"),
            output_dir=frame_config.get("output_dir", "DARD/extracted_frames"),
            overwrite=frame_config.get("overwrite", False),
        )


def extract_frames(
    video_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    clip_type: str,
    overwrite: bool = False,
    frames_logger: FramesExtractionLogger | None = None,
) -> dict | None:
    """Extract frames from a single video with FAIR metadata.

    Returns manifest entry (frame_number → uuid mapping) or None if skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sidecar to get metadata
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

    # Get frame data from sidecar
    frame_data_dict = sidecar_data.get("frame_data", {})

    # Open video
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

            # Check if frame already exists (resumable)
            frame_png = output_dir / f"frame_{frame_number:06d}.png"
            frame_json = output_dir / f"frame_{frame_number:06d}.json"

            if frame_png.exists() and frame_json.exists() and not overwrite:
                pbar.update(1)
                frame_number += 1
                continue

            # Generate frame UUID
            frame_uuid = generate_uuid()

            # Get detection data for this frame (if available)
            frame_key = str(frame_number)
            frame_detections = (
                frame_data_dict.get(frame_key, []) if isinstance(frame_data_dict, dict) else []
            )

            # Create frame metadata
            frame_meta = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps if fps > 0 else 0.0,
                "detections": frame_detections,
            }

            # Add FAIR metadata
            frame_meta = add_fair_metadata(
                frame_meta,
                schema_type="face_crop" if "face" in clip_type else "person_clip",
                parent_uuid=parent_uuid,
                parent_file=parent_file,
            )

            # Override with frame-specific UUID (generate_uuid was called above)
            frame_meta["uuid"] = frame_uuid

            # Reorganize for FAIR
            frame_meta = reorganize_for_fair(
                frame_meta, "face_crop" if "face" in clip_type else "person_clip"
            )

            # Write frame PNG
            try:
                cv2.imwrite(str(frame_png), frame)
            except Exception as e:
                logger.error("Cannot write frame PNG %s: %s", frame_png.name, e)
                pbar.update(1)
                frame_number += 1
                continue

            # Write frame JSON
            try:
                with open(frame_json, "w", encoding="utf-8") as f:
                    json.dump(frame_meta, f, indent=2)
            except Exception as e:
                logger.error("Cannot write frame JSON %s: %s", frame_json.name, e)
                pbar.update(1)
                frame_number += 1
                continue

            # Add to manifest
            frame_manifest["frames"].append(
                {
                    "frame_number": frame_number,
                    "uuid": frame_uuid,
                    "timestamp": frame_meta.get("timestamp", 0.0),
                }
            )

            # Log frame extraction (for traceability)
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

    # Write manifest
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


def main(config_path: str | None = None) -> None:
    """Main entry point."""
    if config_path is None:
        config_path = str(CONFIG_PATH)

    cfg = FrameExtractionConfig.from_yaml(config_path)

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Initialize frames logger
    clips_csv = Path(cfg.input_dir) / "clips_extraction.csv"
    frames_logger = FramesExtractionLogger(output_dir=str(output_dir), clips_csv_path=clips_csv)

    # Find all video files
    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        logger.warning("No MP4 files found in %s", input_dir)
        sys.exit(0)

    clip_type = cfg.get_type()
    logger.info("Extracting frames from %d videos (type: %s)", len(video_files), clip_type)

    for video_path in tqdm(video_files, desc="Extracting frames", unit="video"):
        sidecar_path = video_path.with_suffix(".json")

        # Create output directory for this video
        video_output_dir = output_dir / video_path.stem

        extract_frames(
            video_path,
            sidecar_path,
            video_output_dir,
            clip_type=clip_type,
            overwrite=cfg.overwrite,
            frames_logger=frames_logger,
        )

    logger.info("Frame extraction complete")
    frames_logger.print_summary()


if __name__ == "__main__":
    main()
