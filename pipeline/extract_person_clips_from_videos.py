#!/usr/bin/env python3
"""
Extract video clips containing people from downloaded videos.

Uses person detection and tracking to identify segments where
people are visible, then extracts those clips as separate files.

All parameters are read from config.yaml.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock

import yaml

from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler, discover_video_files

# Configure logging — route through tqdm so output doesn't break progress bars


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

# Setup paths BEFORE importing libraries that might load DLLs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

from dardcollect import PersonDetector, PersonTracker, PoseEstimator
from dardcollect.config import ClipExtractionConfig, DetectorConfig, FaceCropConfig, get_log_level
from dardcollect.encoding_config import EncodingConfig, validate_video_codec
from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.person_clips import process_video
from dardcollect.person_clips_run import VideoProcessRequest


@dataclass
class _FilmJob:
    """Shared state for processing one source film (single argument to helpers)."""

    detector: PersonDetector
    poser: PoseEstimator
    det_config: DetectorConfig
    clip_config: ClipExtractionConfig
    face_crop_cfg: FaceCropConfig | None
    clip_logger: ExtractionLogger
    enc: EncodingConfig
    input_path: Path
    output_dir: Path
    results: list = field(default_factory=list)
    skipped_already_done: int = 0
    lock: Lock = field(default_factory=Lock)


def _load_clip_stage_configs():
    """Load detector/clip/face-crop/encoding configs (fail loud on bad config)."""
    try:
        det_config = DetectorConfig.from_yaml(str(CONFIG_PATH))
        clip_config = ClipExtractionConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

    # Issue #8: fail loud when a non-default codec is not available in the
    # resolved ffmpeg (default libx264 needs no probe). Always resolve _enc so
    # the extraction call sites get the (default-equal) encoding settings.
    with open(CONFIG_PATH, encoding="utf-8") as _f:
        _cfg_all = yaml.safe_load(_f) or {}
    _enc = EncodingConfig.from_yaml_dict(_cfg_all)
    try:
        from dardcollect.archive import _ffmpeg_exe

        validate_video_codec(_enc.video_codec, _ffmpeg_exe())
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    face_crop_cfg: FaceCropConfig | None = None
    try:
        face_crop_cfg = FaceCropConfig.from_yaml(str(CONFIG_PATH))
        logger.info("Face crop config loaded — will annotate arcface + ofiq crop corners")
    except Exception:
        logger.info("No face_crop_extraction config found — face_crop_corners will be skipped")
    return det_config, clip_config, face_crop_cfg, _enc


def _collect_source_films(clip_config):
    """Resolve the input dir and discover source videos (fail loud when empty)."""
    input_path = Path(clip_config.input_dir)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    # Collect video files (case-insensitive so uppercase ``.MP4`` downloads
    # from Archive.org are not skipped on Linux).
    video_files = discover_video_files(input_path)

    if not video_files:
        logger.error("No video files found in: %s", input_path)
        sys.exit(1)

    logger.info("Found %d video(s) to process", len(video_files))
    return input_path, video_files


def _init_clip_components(det_config):
    """Load detector + pose models (fail loud when the detector is missing)."""
    # Select model (Updated to YOLOX-Tiny HumanArt for User Request)
    models_dir = Path(det_config.models_path)

    det_filename = "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_filename = "cigpose-m_coco-wholebody_256x192.onnx"

    det_model_path = models_dir / det_filename
    pose_model_path = models_dir / pose_filename

    if not det_model_path.exists():
        logger.error("Detection model not found: %s", det_model_path)
        logger.error("Run pipeline/setup_models.py first!")
        sys.exit(1)

    # Initialize components. The detector and poser ONNX sessions are shared across
    # film workers — ONNX Runtime Run() is thread-safe and they hold no per-call state,
    # so sharing adds NO extra GPU memory. The PersonTracker is stateful (reset per film
    # via init_tracker), so each film gets its OWN cheap tracker instance below.
    logger.info("Initializing detector (%s)...", det_model_path.name)
    detector = PersonDetector(det_config, model_path=str(det_model_path))

    logger.info("Initializing pose estimator (%s)...", pose_model_path.name)
    poser = PoseEstimator(det_config, model_path=str(pose_model_path))

    # Audio Transcriber - Removed from main loop
    # run pipeline/transcribe_video_clips.py instead
    return detector, poser


def _process_one_film(job: _FilmJob, video_path: Path) -> bool:
    """Process a single source film. Returns True if it was processed, False if
    skipped as already done. Runs concurrently when workers > 1: it shares the
    detector/poser/clip_logger and uses its own tracker, so it is thread-safe."""
    # Mirror the input_dir subtree under output_dir so each source video's
    # clips, JSONs, and `.done` sentinel live in a per-source-dir folder.
    # Video stems are unique across input_dir (timestamps + hashes) so
    # clip filenames don't need a subdir prefix.
    rel_parent = video_path.relative_to(job.input_path).parent
    video_out_dir = job.output_dir / rel_parent
    video_out_dir.mkdir(parents=True, exist_ok=True)
    done_sentinel = video_out_dir / f"{video_path.stem}.done"
    if done_sentinel.exists():
        with job.lock:
            job.skipped_already_done += 1
        logger.debug("SKIP (already done): %s", video_path.name)
        return False

    # Per-video clip_config that points output_clips_dir at the per-source
    # subdir. The process_video function reads output_clips_dir from this
    # config to decide where to write clips and the resume progress file.
    per_video_config = replace(job.clip_config, output_clips_dir=str(video_out_dir))

    try:
        results = process_video(
            VideoProcessRequest(
                video_path=video_path,
                detector=job.detector,
                tracker=PersonTracker(),  # own tracker per film (stateful) — thread-safe
                det_config=job.det_config,
                clip_config=per_video_config,
                poser=job.poser,
                face_crop_cfg=job.face_crop_cfg,
                clip_logger=job.clip_logger,
                encoding=job.enc,
            )
        )
        with job.lock:
            job.results.extend(results)
        done_sentinel.touch()
    except Exception as e:
        logger.error("Error processing %s: %s", video_path.name, e)
    return True


def _run_films(job: _FilmJob, video_files: list[Path]) -> None:
    """Process every film serially or with a thread pool (per-film .done resume)."""
    workers = max(1, job.clip_config.workers)
    if workers == 1:
        logger.info("Processing %d film(s) serially", len(video_files))
        for video_path in video_files:
            _process_one_film(job, video_path)
    else:
        logger.info("Processing %d film(s) with %d parallel workers", len(video_files), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one_film, job, v): v for v in video_files}
            for _ in as_completed(futures):
                pass


def _print_clip_summary(job: _FilmJob) -> None:
    """Log totals + resume count + logger summary."""
    # Per-file detection JSONs are saved after each video is processed

    # Summary
    total_clips = len(job.results)
    total_duration = sum(r.get("duration_seconds", 0) for r in job.results)
    if job.skipped_already_done:
        logger.info("Resume: skipped %d already-processed video(s)", job.skipped_already_done)
    logger.info(
        "\nSummary: Extracted %d clips (%.1f seconds total)",
        total_clips,
        total_duration,
    )

    # Print extraction log summary
    job.clip_logger.print_summary()


@add_timer
def main():
    """Main entry point."""
    det_config, clip_config, face_crop_cfg, _enc = _load_clip_stage_configs()
    input_path, video_files = _collect_source_films(clip_config)
    detector, poser = _init_clip_components(det_config)

    # Process videos
    output_dir = Path(clip_config.output_clips_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extraction logger (CSV audit trail). Its append is lock-guarded, so
    # concurrent film workers can log into the single clips_extraction.csv safely.
    downloads_csv = Path(clip_config.input_dir).parent / "downloads.csv"
    clip_logger = ExtractionLogger(output_dir=str(output_dir), downloads_csv_path=downloads_csv)

    job = _FilmJob(
        detector=detector,
        poser=poser,
        det_config=det_config,
        clip_config=clip_config,
        face_crop_cfg=face_crop_cfg,
        clip_logger=clip_logger,
        enc=_enc,
        input_path=input_path,
        output_dir=output_dir,
    )
    _run_films(job, video_files)
    _print_clip_summary(job)


if __name__ == "__main__":
    main()
