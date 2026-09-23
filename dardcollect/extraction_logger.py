"""Incremental CSV logging for extraction pipeline stages.

Provides resumable CSV logging for person clip extractions and downloads.
Logs are written incrementally (append-only) so that progress is preserved
even if the process is interrupted.

All log entries include FAIR-compliant UUIDs, timestamps, and provenance links
to upstream artifacts (e.g., archive.org identifiers).
"""

import csv
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from dardcollect.fair import generate_uuid

# ── CSV helper (consolidated from download_media_from_archive) ────────────────

# Pipeline-specific fields that come first in the downloads CSV. title/creator/
# date/license are Dublin Core terms (the JSON-LD @context in the sidecars maps
# them to dct:*); CSVs stay plain tables — no @context column.
_PIPELINE_FIELDS = [
    "uuid",
    "title",
    "creator",
    "date",
    "license",
    "archive_org_identifier",
    "filename_downloaded",
    "media_type",
    "downloaded_at",
    "download_stage_script",
    "download_stage_timestamp",
]


def _write_to_csv(csv_path: Path, metadata: dict) -> None:
    """Append a metadata row to a CSV file, extending columns dynamically.

    If the CSV does not exist, creates it with all current field names.
    If new fields appear in *metadata* that are not in the existing CSV,
    rewrites the file with the expanded header and all prior rows.

    Args:
        csv_path: Path to the CSV file.
        metadata: Dictionary of column-value pairs to append.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        archive_fields = [k for k in metadata if k not in _PIPELINE_FIELDS]
        fieldnames = _PIPELINE_FIELDS + archive_fields
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerow(metadata)
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        new_fields = [k for k in metadata if k not in existing_fields]
        rows: list[dict] = list(reader) if new_fields else []

    if new_fields:
        fieldnames = existing_fields + new_fields
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(metadata)
    else:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=existing_fields, extrasaction="ignore", restval=""
            )
            writer.writerow(metadata)


logger = logging.getLogger(__name__)


@dataclass
class ClipRecord:
    """The fields of one clips_extraction.csv row (excluding uuid/timestamp)."""

    source_video: str
    fps: float
    start_frame: int
    end_frame: int
    start_seconds: float
    duration_seconds: float
    max_persons_per_frame: int
    detector_model: str
    detector_confidence: float
    output_path: str


class ExtractionLogger:
    """Append-only CSV logger for person clip extractions.

    Writes clips_extraction.csv to the output clips directory. Each row links
    a clip to its source video and, when available, to the archive.org download
    record via archive_org_identifier.

    The log is safe to interrupt and resume because writes are append-only and
    the header is written only if the file does not already exist.
    """

    def __init__(
        self,
        output_dir: str | Path = "DARD/extracted_person_clips",
        downloads_csv_path: str | Path | None = None,
    ):
        """Initialize the extraction logger.

        Args:
            output_dir: Directory where clips and clips_extraction.csv are written.
                Created if it does not exist.
            downloads_csv_path: Path to downloads.csv; used to populate
                archive_org_identifier by matching source video filenames.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = output_dir / "clips_extraction.csv"
        self._header_written = self.log_path.exists() and self.log_path.stat().st_size > 0
        # Guards the append + _header_written mutation below: with person_extraction.workers
        # > 1 several film workers log into this one CSV concurrently. Without the lock two
        # threads can both see _header_written False and write two header rows, or interleave
        # partially-written rows. Mirrors FramesExtractionLogger._write_lock.
        self._write_lock = Lock()

        # Lookup {filename_downloaded → archive_org_identifier} from downloads.csv
        self._source_to_identifier: dict[str, str] = {}
        if downloads_csv_path and Path(downloads_csv_path).exists():
            with open(downloads_csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fn = row.get("filename_downloaded", "")
                    aid = row.get("archive_org_identifier", "")
                    if fn and aid:
                        self._source_to_identifier[fn] = aid

        self.fieldnames = [
            "uuid",
            "archive_org_identifier",
            "timestamp",
            "source_video",
            "fps",
            "start_frame",
            "end_frame",
            "start_seconds",
            "duration_seconds",
            "max_persons_per_frame",
            "detector_model",
            "detector_confidence",
            "output_path",
        ]
        # Note: downloads.csv gains title/creator/date/license columns from
        # _PIPELINE_FIELDS above (written by the download stage);
        # clips_extraction.csv keeps its own fixed fieldnames.

    def log_extraction(self, record: ClipRecord) -> None:
        """Append a clip extraction record to clips_extraction.csv.

        Generates a new UUID and timestamp automatically. Writes are atomic
        (append-only) so the file remains valid even if the process crashes.
        """
        timestamp = datetime.now(UTC).isoformat()

        row = {
            "uuid": generate_uuid(),
            "archive_org_identifier": self._source_to_identifier.get(record.source_video, ""),
            "timestamp": timestamp,
            "source_video": record.source_video,
            "fps": round(record.fps, 3),
            "start_frame": record.start_frame,
            "end_frame": record.end_frame,
            "start_seconds": round(record.start_seconds, 2),
            "duration_seconds": round(record.duration_seconds, 2),
            "max_persons_per_frame": record.max_persons_per_frame,
            "detector_model": record.detector_model,
            "detector_confidence": round(record.detector_confidence, 3),
            "output_path": record.output_path,
        }

        try:
            with self._write_lock, open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True

                writer.writerow(row)

        except Exception as e:
            logger.error("Failed to write extraction log entry: %s", e)

    def print_summary(self) -> None:
        """Read clips_extraction.csv and log aggregate statistics.

        Reports total clips, total duration, total persons, average confidence,
        and clips grouped by source video.
        """
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            logger.info("No extraction log found.")
            return

        try:
            entries = []
            with open(self.log_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                entries = list(reader)

            if not entries:
                logger.info("Extraction log is empty.")
                return

            total_clips = len(entries)
            total_duration = sum(
                float(e["duration_seconds"]) for e in entries if e["duration_seconds"]
            )
            total_persons = sum(
                int(e["max_persons_per_frame"]) for e in entries if e.get("max_persons_per_frame")
            )
            avg_confidence = sum(
                float(e["detector_confidence"]) for e in entries if e["detector_confidence"]
            ) / len(entries)

            # Group by source video
            by_source = {}
            for e in entries:
                src = e["source_video"]
                by_source[src] = by_source.get(src, 0) + 1

            source_summary = "\n".join(
                f"    - {src}: {count} clips" for src, count in sorted(by_source.items())
            )

            logger.info(
                "\n📊 Extraction Summary:\n"
                "  CSV: %s\n"
                "  Total clips extracted: %d\n"
                "  Total duration: %.1f seconds\n"
                "  Total persons: %d\n"
                "  Average detector confidence: %.3f\n"
                "  Clips by source:\n%s",
                self.log_path,
                total_clips,
                total_duration,
                total_persons,
                avg_confidence,
                source_summary,
            )

        except Exception as e:
            logger.error("Failed to compute summary: %s", e)
