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
  python pipeline/extract_frames_from_videos.py \
    --input-dir DARD/extracted_person_clips \
    --output-dir DARD/extracted_frames/person_clips \
    --type person_clip

All parameters are read from config.yaml under the 'frame_extraction' key.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from dardcollect.config import FrameExtractionConfig
from dardcollect.frames import extract_frames
from dardcollect.pipeline_loggers import FramesExtractionLogger
from dardcollect.pipeline_utils import _TqdmHandler, check_disk_space
from dardcollect.source_frames import extract_source_frames_for_clip

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Path to config file
CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG", Path(__file__).parent.parent / "configs" / "config.archive_all.yaml"
    )
)


@dataclass
class _FrameRun:
    """Source-frame run state: the extraction loop's fixed config + running tallies."""

    cfg: FrameExtractionConfig
    output_dir: Path
    total_frames: int = 0
    clips_without_run: int = 0
    clips_already_done: int = 0

    def add(self, result: tuple[int, bool]) -> None:
        written, had_run = result
        self.total_frames += written
        if not had_run:
            self.clips_without_run += 1
        elif written == 0:
            self.clips_already_done += 1

    def check_disk(self) -> None:
        check_disk_space(self.output_dir, self.cfg.min_free_disk_gb)


def _run_sidecars(
    clip_sidecars: list[Path],
    workers: int,
    one,
    run: _FrameRun,
) -> None:
    """Extract each clip's source frames, in parallel or serial, tallying results."""
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, s): s for s in clip_sidecars}
            try:
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Source frames", unit="clip"
                ):
                    try:
                        run.add(future.result())
                    except Exception as e:
                        logger.error("Error on %s: %s", futures[future].name, e)
                    run.check_disk()
            except SystemExit:
                for pending in futures:
                    pending.cancel()
                raise
        return
    for sidecar in tqdm(clip_sidecars, desc="Source frames", unit="clip"):
        run.check_disk()
        run.add(one(sidecar))


def _run_source_video_mode(
    cfg: FrameExtractionConfig,
    input_dir: Path,
    output_dir: Path,
    frames_logger: FramesExtractionLogger,
) -> None:
    """Pull N consecutive source-video frames per person clip, at detected timestamps.

    Reads clip sidecars rather than clip videos: each one names the source video it was
    cut from and keys its detections by absolute source-frame number, so the frames come
    straight out of the original with no re-inference. See
    docs/DESIGN_video_frame_masks.md.
    """
    # A clip sidecar is <stem>.json sitting next to <stem>.mp4. That sibling test is what
    # separates it from the derived sidecars later stages write (.transcription.json,
    # .quality.json): those have no matching .mp4.
    #
    # Do NOT filter on len(Path.suffixes) instead — `suffixes` splits on every dot in the
    # name, so a film titled "Esther and the King (1960).ia" yields
    # ['.ia_04m33s-04m41s', '.json'] and every one of its clips gets silently dropped.
    # That mistake cost 2,694 of 13,406 clips (20%) on the 2026-08-04 run.
    clip_sidecars = sorted(p for p in input_dir.rglob("*.json") if p.with_suffix(".mp4").exists())
    if not clip_sidecars:
        logger.warning("No clip sidecars found in %s", input_dir)
        sys.exit(0)

    workers = min(cfg.workers, len(clip_sidecars))
    logger.info(
        "Extracting %d consecutive source-video frame(s) for each of %d clip(s) (workers: %d)",
        cfg.frames_per_clip,
        len(clip_sidecars),
        workers,
    )

    check_disk_space(output_dir, cfg.min_free_disk_gb)

    def _one(sidecar: Path) -> tuple[int, bool]:
        return extract_source_frames_for_clip(
            sidecar,
            output_dir,
            frames_per_clip=cfg.frames_per_clip,
            overwrite=cfg.overwrite,
            frames_logger=frames_logger,
        )

    run = _FrameRun(cfg=cfg, output_dir=output_dir)
    _run_sidecars(clip_sidecars, workers, _one, run)

    # Clips with no qualifying run are expected, not an error: a clip can track a person
    # from behind for its whole span. Reported so the cost of the face-keypoint gate is
    # visible without re-running anything.
    logger.info(
        "Source-frame extraction complete: %d frame(s) newly written from %d clip(s); "
        "%d clip(s) already had theirs; "
        "%d clip(s) had no run of %d consecutive detected frames",
        run.total_frames,
        len(clip_sidecars) - run.clips_without_run - run.clips_already_done,
        run.clips_already_done,
        run.clips_without_run,
        cfg.frames_per_clip,
    )
    frames_logger.print_summary()


def main(config_path: str | None = None) -> None:
    """Extract PNG frames from video clips with FAIR-compliant metadata sidecars.

    Reads video files from the configured input directory and writes each frame
    as a PNG image with a companion JSON sidecar containing frame-level metadata
    (UUID, timestamp, detection data, parent reference). Also produces a
    frames_manifest.json for batch discovery.

    Args:
        config_path: Path to config.yaml. If None, uses the default config
            file alongside the pipeline scripts.
    """
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

    if cfg.source == "source_video":
        _run_source_video_mode(cfg, input_dir, output_dir, frames_logger)
        return

    # Find all video files (recursively: clips may live in per-source-subdir trees)
    video_files = sorted(input_dir.rglob("*.mp4"))

    if not video_files:
        logger.warning("No MP4 files found in %s", input_dir)
        sys.exit(0)

    clip_type = cfg.get_type()
    workers = min(cfg.workers, len(video_files))
    logger.info(
        "Extracting frames from %d videos (type: %s, workers: %d)",
        len(video_files),
        clip_type,
        workers,
    )

    def _extract_one(video_path: Path) -> None:
        sidecar_path = video_path.with_suffix(".json")

        # Mirror input_dir subtree under output_dir so frames keep the same
        # per-source-subdir layout as the face crops (filtered_face_crops/0c9460bf-.../
        # → extracted_frames/0c9460bf-.../clip_face_1/frame_*.png). Video stems are
        # unique within input_dir, so the final stem dir is unambiguous.
        rel_parent = video_path.relative_to(input_dir).parent
        video_output_dir = output_dir / rel_parent / video_path.stem

        extract_frames(
            video_path,
            sidecar_path,
            video_output_dir,
            clip_type=clip_type,
            overwrite=cfg.overwrite,
            frames_logger=frames_logger,
        )

    # This stage is by far the heaviest writer in the pipeline — a filtered face
    # crop expands to hundreds of lossless PNG frames (~190 MB per crop measured
    # on real data), so a full pass can want hundreds of GB. Without this check it
    # would happily fill the filesystem; check_disk_space exits non-zero instead,
    # which the orchestrator reports as a failed stage. Nothing is lost: frames
    # resume per file, so a later run picks up exactly where this one stopped.
    check_disk_space(output_dir, cfg.min_free_disk_gb)

    if workers > 1:
        # Clips are independent: each writes into its own output subdir and its
        # own manifest, and the only shared state is the CSV logger (lock-guarded
        # in FramesExtractionLogger). Threads rather than processes because the
        # work is dominated by per-file latency on network storage, and cv2's
        # decode/imwrite release the GIL.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one, v): v for v in video_files}
            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Extracting frames",
                    unit="video",
                ):
                    video_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Error extracting frames from %s: %s", video_path.name, e)
                    check_disk_space(output_dir, cfg.min_free_disk_gb)
            except SystemExit:
                # check_disk_space called sys.exit. Cancel what has not started
                # yet: leaving the pool to drain would keep writing into the very
                # filesystem we just declared too full.
                for pending in futures:
                    pending.cancel()
                raise
    else:
        for video_path in tqdm(video_files, desc="Extracting frames", unit="video"):
            check_disk_space(output_dir, cfg.min_free_disk_gb)
            _extract_one(video_path)

    logger.info("Frame extraction complete")
    frames_logger.print_summary()


if __name__ == "__main__":
    main()
