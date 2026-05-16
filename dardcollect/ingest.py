"""
Custom data source ingestion for DARDcollect.

Provides register_source_files() to create a source manifest CSV for files
from any data source — the equivalent of downloads.csv for custom datasets.
This bootstraps the traceability and provenance chain without requiring Archive.org.
"""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from dardcollect.fair import generate_uuid

logger = logging.getLogger(__name__)

_EXTENSIONS: dict[str, list[str]] = {
    "video": [".mp4", ".avi", ".mkv", ".mov", ".webm"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "text": [".pdf", ".txt"],
}

# archive_org_identifier is kept empty for custom sources but retained
# for schema compatibility with downstream loggers that read downloads.csv.
_FIXED_COLUMNS = [
    "uuid",
    "archive_org_identifier",
    "filename_downloaded",
    "media_type",
    "registered_at",
    "source_path",
]


def register_source_files(
    input_dir: Path | str,
    output_csv: Path | str,
    media_type: str,
    extensions: list[str] | None = None,
    extra_metadata: dict[str, str] | None = None,
    overwrite: bool = False,
) -> int:
    """Create a source manifest CSV for files from a custom data source.

    Scans input_dir for media files and assigns each a UUID, producing a
    downloads.csv-compatible manifest that anchors the traceability chain for
    all downstream pipeline stages (face crop extraction, transcription, OCR,
    quality annotation, etc.).

    Use this instead of download_media_from_archive.py when your source files
    were not downloaded from Archive.org. Pass the output CSV path as the
    downloads_csv_path argument to any pipeline logger.

    Args:
        input_dir: Directory containing your source media files.
        output_csv: Path where the manifest CSV will be written.
        media_type: One of "video", "audio", "image", "text". Used to filter
            files by extension (unless extensions is provided) and to populate
            the media_type column.
        extensions: File extensions to include (e.g. [".mp4", ".mov"]). If
            None, uses the default set for media_type.
        extra_metadata: Optional dict of additional columns to add to every
            row (e.g. {"dataset": "MyDataset", "license": "CC-BY-4.0"}).
        overwrite: If True, overwrite an existing CSV. If False (default),
            append only files not already recorded (incremental / resumable).

    Returns:
        Number of files newly registered.

    Raises:
        ValueError: If media_type is not one of the supported values.
        FileNotFoundError: If input_dir does not exist.
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if media_type not in _EXTENSIONS:
        raise ValueError(f"media_type must be one of {list(_EXTENSIONS)}, got {media_type!r}")

    exts = {e.lower() for e in (extensions or _EXTENSIONS[media_type])}
    extra = extra_metadata or {}
    fieldnames = _FIXED_COLUMNS + [k for k in extra if k not in _FIXED_COLUMNS]

    already_registered: set[str] = set()
    if not overwrite and output_csv.exists():
        with open(output_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename_downloaded", "")
                if fn:
                    already_registered.add(fn)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    write_header = overwrite or not output_csv.exists()

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    registered = 0

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for file_path in files:
            if file_path.name in already_registered:
                logger.debug("SKIP (already registered): %s", file_path.name)
                continue
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "archive_org_identifier": "",
                    "filename_downloaded": file_path.name,
                    "media_type": media_type,
                    "registered_at": datetime.now(UTC).isoformat(),
                    "source_path": str(file_path.resolve()),
                    **{k: v for k, v in extra.items() if k not in _FIXED_COLUMNS},
                }
            )
            registered += 1
            logger.debug("Registered: %s", file_path.name)

    logger.info("Registered %d file(s) → %s", registered, output_csv)
    return registered
