#!/usr/bin/env python3
"""
Extract normalized face crop videos from person clip videos.

Reads the sidecar JSONs written by extract_person_clips_from_videos.py (which contain
SG-smoothed keypoints and pre-computed face crop corners) instead of
re-running detection, so face crops use the same smoothed keypoint positions
as the rest of the pipeline.

One crop format is produced per track:

  output_dir/  — 616×616, OFIQ-aligned (BSI-OFIQ convention), written flat (no subdirs).
    Wider framing with eyes at y≈272, nose at y≈336, mouth at y≈402.
    Matches the format expected by all OFIQ quality measures: sharpness,
    expression neutrality, head pose, compression artifacts, background
    uniformity, and face occlusion.
    Input to filter_face_crops_by_quality.py and annotate_face_quality.py.

The sidecar JSON for each OFIQ video includes a 'face_crop_corners_arcface'
field — the 4 corners (TL, TR, BR, BL) of the ArcFace 112×112 region in OFIQ
frame pixel coordinates.  Because both formats align to fixed canonical landmark
positions, this region is constant across all frames and all clips.  Downstream
scripts (filter_face_crops_by_quality.py, annotate_face_quality.py) use it to
extract 112×112 ArcFace crops from OFIQ frames for MagFace scoring.

All parameters are read from config.yaml under the 'face_crop_extraction' key.
"""

import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from dardcollect.face_crops import process_video
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger
from dardcollect.pipeline_utils import _TqdmHandler

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

from dardcollect.config import FaceCropConfig, get_log_level
from dardcollect.encoding_config import EncodingConfig, validate_video_codec

# ── Entry point ───────────────────────────────────────────────────────────────


@dataclass
class _CropJob:
    """Shared state for extracting face crops from one clip video."""

    face_config: FaceCropConfig
    face_crops_logger: FaceCropsExtractionLogger
    enc: EncodingConfig
    input_path: Path
    output_dir: Path
    total_written: int = 0
    skipped_already_done: int = 0


def _load_face_crop_stage():
    """Load face-crop + encoding configs (fail loud on bad config/codec)."""
    try:
        face_config = FaceCropConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    # Issue #8: encoding config + fail-loud codec validation (default libx264
    # needs no probe). The resolved settings flow to the moviepy writer.
    with open(CONFIG_PATH, encoding="utf-8") as _f:
        _cfg_all = yaml.safe_load(_f) or {}
    _enc = EncodingConfig.from_yaml_dict(_cfg_all)
    try:
        from dardcollect.archive import _ffmpeg_exe

        validate_video_codec(_enc.video_codec, _ffmpeg_exe())
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    return face_config, _enc


def _discover_crop_inputs(face_config):
    """Resolve the input path and discover clip videos (fail loud when empty)."""
    input_path = Path(face_config.input_dir)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm", "*.m4v"):
            video_files.extend(input_path.rglob(ext))

    if not video_files:
        logger.error("No video files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d video(s) to process", len(video_files))
    return input_path, video_files


def _process_one_crop_video(job: _CropJob, video_path: Path) -> None:
    """Extract OFIQ crops from one clip video (per-source-subdir layout)."""
    # Mirror input_dir subtree under output_dir so face crops keep the
    # same per-source-subdir layout as the person clips. Clips are
    # discovered via rglob above, so video_path may live in a subdir.
    rel_parent = video_path.relative_to(job.input_path).parent
    video_out_dir = job.output_dir / rel_parent
    video_out_dir.mkdir(parents=True, exist_ok=True)
    done_sentinel = video_out_dir / f"{video_path.stem}.done"
    if done_sentinel.exists():
        job.skipped_already_done += 1
        logger.debug("SKIP (already done): %s", video_path.name)
        return

    # Per-video face_config with output_dir scoped to this subdir. The
    # process_video function reads output_dir from the config to decide
    # where to write the OFIQ crop video and sidecar.
    per_video_config = replace(job.face_config, output_dir=str(video_out_dir))

    logger.info("Processing: %s", video_path.name)
    try:
        n = process_video(video_path, per_video_config, job.face_crops_logger, encoding=job.enc)
        job.total_written += n
        done_sentinel.touch()
    except Exception as e:
        logger.error("Error processing %s: %s", video_path.name, e)


def main() -> None:
    """Extract 616×616 OFIQ-aligned face crop videos from person clip .json sidecars.

    Reads detection data from sidecar JSONs produced by extract_person_clips_from_videos.py,
    extracts normalized OFIQ-format face crops for each track, and writes .mp4 videos
    with companion .json sidecars. The process_video function handles the per-video logic,
    while this entry point manages discovery, progress tracking, and logging.

    Configuration is read from config.yaml under the 'face_crop_extraction' key.
    """
    face_config, _enc = _load_face_crop_stage()
    input_path, video_files = _discover_crop_inputs(face_config)

    output_dir = Path(face_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face crops logger
    clips_csv = Path(face_config.input_dir) / "clips_extraction.csv"
    face_crops_logger = FaceCropsExtractionLogger(
        output_dir=str(output_dir), clips_csv_path=clips_csv
    )

    job = _CropJob(
        face_config=face_config,
        face_crops_logger=face_crops_logger,
        enc=_enc,
        input_path=input_path,
        output_dir=output_dir,
    )
    for video_path in video_files:
        _process_one_crop_video(job, video_path)

    if job.skipped_already_done:
        logger.info("Resume: skipped %d already-processed video(s)", job.skipped_already_done)

    logger.info("\nDone. Wrote %d OFIQ face crop video(s) total.", job.total_written)
    face_crops_logger.print_summary()


if __name__ == "__main__":
    main()
