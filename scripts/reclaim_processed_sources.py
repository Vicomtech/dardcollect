"""Reclaim disk space by deleting source videos the clip stage has finished with.

``extract_person_clips_from_videos.py`` touches a ``<video_stem>.done`` sentinel next
to a video's clips once that video is fully processed. Normally no pipeline stage reads
the source file again — audio_clips, face_crops_video, transcribe_video, quality and
filter all consume the extracted clips, not the original download. On a capacity-bound
filesystem those finished sources are the cheapest space to reclaim.

**Except when the frames stage runs in source-video mode.** With
``frame_extraction.source: source_video`` the frames stage seeks back into the original
video to pull frames at the detected timestamps (see docs/DESIGN_video_frame_masks.md),
so deleting a source there costs every frame and mask for that film — silently, since a
missing source is only a warning. This script refuses to run in that configuration
unless ``--force`` says otherwise.

Deleting one is safe for FAIR provenance: ``downloads.csv`` keeps the row, including
``uuid`` and ``archive_org_identifier``, so the file stays fully attributed and can be
re-downloaded from Archive.org. This script therefore REFUSES to delete any video that
has no row in ``downloads.csv`` — an unrecorded file would be unrecoverable.

Dry-run by default; pass ``--apply`` to actually delete.

    # see what could be reclaimed, delete nothing
    uv run python scripts/reclaim_processed_sources.py \
        --config configs/config.archive_download_videos_custom.yaml

    # reclaim now, down to the smallest set that restores 60 GB free
    uv run python scripts/reclaim_processed_sources.py \
        --config configs/config.archive_download_videos_custom.yaml \
        --target-free-gb 60 --apply

    # unattended: check every 10 min, reclaim only when free space drops below 40 GB
    uv run python scripts/reclaim_processed_sources.py \
        --config configs/config.archive_download_videos_custom.yaml \
        --watch 600 --reclaim-below-gb 40 --target-free-gb 60 --apply
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dardcollect.config import _resolve_path_templates

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mpg", ".mpeg", ".ogv", ".webm", ".mov"}
GB = 1024**3


def _load_paths(config_path: Path) -> tuple[Path, Path, Path]:
    """Return (source_videos_dir, clips_dir, downloads_csv) from the config."""
    with open(config_path, encoding="utf-8") as f:
        cfg = _resolve_path_templates(yaml.safe_load(f) or {})
    section = cfg.get("person_extraction", {})
    input_dir = Path(section["input_dir"])
    clips_dir = Path(section["output_clips_dir"])
    return input_dir, clips_dir, input_dir.parent / "downloads.csv"


def _recorded_filenames(downloads_csv: Path) -> set[str]:
    """Filenames that have a provenance row in downloads.csv."""
    if not downloads_csv.exists():
        return set()
    with open(downloads_csv, encoding="utf-8", errors="replace", newline="") as f:
        return {
            row["filename_downloaded"]
            for row in csv.DictReader(f)
            if row.get("filename_downloaded")
        }


def _reclaimable(input_dir: Path, clips_dir: Path, recorded: set[str]) -> list[tuple[Path, int]]:
    """Source videos whose clip extraction finished, largest first.

    A video qualifies only when BOTH hold: its ``.done`` sentinel exists (the clip
    stage completed it, so nothing reads the source again) and downloads.csv has its
    row (so deleting loses no provenance and the file stays re-downloadable).
    """
    out: list[tuple[Path, int]] = []
    for video in input_dir.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        sentinel = clips_dir / video.relative_to(input_dir).parent / f"{video.stem}.done"
        if not sentinel.exists():
            continue
        if video.name not in recorded:
            print(f"  SKIP (no downloads.csv row, would lose provenance): {video.name}")
            continue
        out.append((video, video.stat().st_size))
    return sorted(out, key=lambda pair: pair[1], reverse=True)


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / GB


def _reclaim_once(config_path: Path, target_free_gb: float, apply: bool) -> float:
    """Delete finished sources until *target_free_gb* is free. Returns GB freed."""
    input_dir, clips_dir, downloads_csv = _load_paths(config_path)
    if not input_dir.exists():
        print(f"[reclaim] source dir not found: {input_dir}")
        return 0.0

    free_before = _free_gb(input_dir)
    candidates = _reclaimable(input_dir, clips_dir, _recorded_filenames(downloads_csv))
    total_gb = sum(size for _, size in candidates) / GB
    print(
        f"[reclaim] free: {free_before:.1f} GB | target: {target_free_gb:.1f} GB | "
        f"finished sources: {len(candidates)} ({total_gb:.1f} GB available to reclaim)"
    )

    freed_gb = 0.0
    for video, size in candidates:
        if free_before + freed_gb >= target_free_gb:
            break
        size_gb = size / GB
        if apply:
            try:
                video.unlink()
            except OSError as e:
                print(f"  FAILED to delete {video.name}: {e}")
                continue
            print(f"  deleted {video.name} ({size_gb:.2f} GB)")
        else:
            print(f"  would delete {video.name} ({size_gb:.2f} GB)")
        freed_gb += size_gb

    verb = "freed" if apply else "would free"
    print(f"[reclaim] {verb} {freed_gb:.1f} GB{'' if apply else ' (dry run — pass --apply)'}")
    return freed_gb


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument(
        "--target-free-gb",
        type=float,
        default=60.0,
        help="Stop deleting once this much space is free (default: 60)",
    )
    parser.add_argument(
        "--reclaim-below-gb",
        type=float,
        default=None,
        help="Only reclaim when free space is under this (default: always)",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Re-check every N seconds instead of running once",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reclaim even when the frames stage still needs the source videos",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    # The frames stage reads the originals back in source-video mode, so reclaiming
    # them there silently strips whole films out of the frame/mask output.
    with open(config_path, encoding="utf-8") as f:
        frame_source = (yaml.safe_load(f) or {}).get("frame_extraction", {}).get("source", "clip")
    if str(frame_source) == "source_video" and not args.force:
        print(
            "[reclaim] REFUSING: frame_extraction.source is 'source_video', so the frames "
            "stage still reads the original downloads. Deleting them would silently drop "
            "their frames and masks. Re-run with --force only if you are certain frames "
            "has already finished.",
            flush=True,
        )
        return 1

    input_dir, _, _ = _load_paths(config_path)
    while True:
        free_gb = _free_gb(input_dir)
        if args.reclaim_below_gb is None or free_gb < args.reclaim_below_gb:
            _reclaim_once(config_path, args.target_free_gb, args.apply)
        else:
            print(
                f"[reclaim] free: {free_gb:.1f} GB — above the "
                f"{args.reclaim_below_gb:.1f} GB threshold, nothing to do",
                flush=True,
            )
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
