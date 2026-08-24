"""Re-download source media that ``reclaim_processed_sources.py`` deleted.

Reclaiming source videos removes the file but KEEPS its ``downloads.csv`` row
(``archive_org_identifier`` + ``filename_downloaded`` + ``language``), so every
reclaimed file stays fully attributed and re-fetchable. This script restores the
originals from Archive.org using exactly those recorded identifiers.

It reconstructs each expected path as ``<videos>/<language>/<filename>`` (the same
layout the download stage writes) and re-fetches only the files that are missing on
disk. It does NOT modify ``downloads.csv`` — the provenance row already exists; only
the file is being put back, so UUIDs and lineage are unchanged.

Dry-run by default; pass ``--apply`` to actually download.

    # list what is missing (reclaimed), download nothing
    uv run python scripts/redownload_sources.py \
        --config configs/config.archive_all.yaml

    # restore every missing source video
    uv run python scripts/redownload_sources.py \
        --config configs/config.archive_all.yaml --apply

    # restore a single item by its archive.org identifier
    uv run python scripts/redownload_sources.py \
        --config configs/config.archive_all.yaml --identifier esther-and-the-king-1960 --apply

    # restore missing audio instead of video
    uv run python scripts/redownload_sources.py \
        --config configs/config.archive_all.yaml --media-type audio --apply
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dardcollect.config import _resolve_path_templates

# media_type → the download stage's output subdir under archive_org_public_domain
_MEDIA_SUBDIR = {"video": "videos", "audio": "audio", "image": "images", "text": "texts"}
# media_types organised into per-language subfolders by the download stage
_LANGUAGE_AWARE = {"video", "audio", "text"}


def _load_paths(config_path: Path, media_type: str) -> tuple[Path, Path]:
    """Return (media_dir, downloads_csv) for *media_type* from the config."""
    with open(config_path, encoding="utf-8") as f:
        cfg = _resolve_path_templates(yaml.safe_load(f) or {})
    base = Path(cfg.get("base_output_dir") or f"{cfg.get('root', '.')}/archive_org_public_domain")
    media_dir = base / _MEDIA_SUBDIR[media_type]
    return media_dir, base / "downloads.csv"


def _expected_path(media_dir: Path, filename: str, language: str, media_type: str) -> Path:
    """Reconstruct the on-disk path the download stage would have written."""
    safe = filename.replace("/", "_")
    if media_type in _LANGUAGE_AWARE:
        return media_dir / (language or "und") / safe
    return media_dir / safe


def find_missing_sources(
    media_dir: Path, downloads_csv: Path, media_type: str
) -> list[tuple[str, str, Path]]:
    """Recorded files of *media_type* that are absent on disk (reclaimed).

    Returns (identifier, filename, expected_path) tuples. A file counts as present
    if it exists at its expected language path OR anywhere under media_dir (covers
    a stale/renamed language folder), so only genuinely deleted files are returned.
    """
    if not downloads_csv.exists():
        return []
    missing: list[tuple[str, str, Path]] = []
    seen: set[tuple[str, str]] = set()
    with open(downloads_csv, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("media_type") != media_type:
                continue
            identifier = (row.get("archive_org_identifier") or "").strip()
            filename = (row.get("filename_downloaded") or "").strip()
            language = (row.get("language") or "").strip()
            if not identifier or not filename:
                continue
            key = (identifier, filename)
            if key in seen:
                continue
            seen.add(key)
            expected = _expected_path(media_dir, filename, language, media_type)
            if expected.exists():
                continue
            if media_dir.exists() and any(media_dir.rglob(filename.replace("/", "_"))):
                continue
            missing.append((identifier, filename, expected))
    return missing


def redownload_one(identifier: str, filename: str, target: Path) -> bool:
    """Fetch *filename* from the Archive.org item *identifier* into *target*."""
    from internetarchive import get_item

    target.parent.mkdir(parents=True, exist_ok=True)
    item = get_item(identifier)
    f = item.get_file(filename)
    if f is None or not f.exists:
        print(f"  MISS: {identifier} has no file {filename!r} on Archive.org")
        return False
    f.download(file_path=str(target), ignore_existing=True, retries=3)
    return target.exists() and target.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument(
        "--media-type",
        default="video",
        choices=sorted(_MEDIA_SUBDIR),
        help="Which media type to restore (default: video)",
    )
    parser.add_argument(
        "--identifier",
        default=None,
        help="Restore only this archive.org identifier (default: all missing)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually download (default: dry run)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    media_dir, downloads_csv = _load_paths(config_path, args.media_type)
    if not downloads_csv.exists():
        print(f"[redownload] downloads.csv not found: {downloads_csv}")
        return 1

    missing = find_missing_sources(media_dir, downloads_csv, args.media_type)
    if args.identifier:
        missing = [m for m in missing if m[0] == args.identifier]

    print(
        f"[redownload] media_type={args.media_type} | missing on disk: {len(missing)}"
        f"{' (filtered to --identifier)' if args.identifier else ''}"
    )
    if not missing:
        print("[redownload] nothing to restore.")
        return 0

    restored = 0
    for identifier, filename, target in missing:
        if not args.apply:
            print(f"  would restore {identifier} → {target}")
            continue
        print(f"  restoring {identifier} → {target.name}")
        try:
            if redownload_one(identifier, filename, target):
                restored += 1
            else:
                print(f"  FAILED: {identifier} ({filename})")
        except Exception as e:
            print(f"  ERROR restoring {identifier}: {e}")

    verb = "restored" if args.apply else "would restore"
    print(
        f"[redownload] {verb} {restored if args.apply else len(missing)} file(s)"
        f"{'' if args.apply else ' (dry run — pass --apply)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
