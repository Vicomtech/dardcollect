"""Incremental CSV logging for extraction pipeline stages.

Provides resumable CSV logging for person clip extractions and downloads.
Logs are written incrementally (append-only) so that progress is preserved
even if the process is interrupted.

All log entries include FAIR-compliant UUIDs, timestamps, and provenance links
to upstream artifacts (e.g., archive.org identifiers).
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from dardcollect.fair import generate_uuid

# ── CSV helper (consolidated from download_media_from_archive) ────────────────

# Pipeline-specific fields that come first in the downloads CSV
_PIPELINE_FIELDS = [
    "uuid",
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

    def log_extraction(
        self,
        source_video: str,
        fps: float,
        start_frame: int,
        end_frame: int,
        start_seconds: float,
        duration_seconds: float,
        max_persons_per_frame: int,
        detector_model: str,
        detector_confidence: float,
        output_path: str,
    ) -> None:
        """Append a clip extraction record to clips_extraction.csv.

        Generates a new UUID and timestamp automatically. Writes are atomic
        (append-only) so the file remains valid even if the process crashes.

        Args:
            source_video: Filename of the source video.
            fps: Frames per second of the source video.
            start_frame: First frame number of the extracted clip.
            end_frame: Last frame number of the extracted clip.
            start_seconds: Start time in seconds.
            duration_seconds: Clip duration in seconds.
            max_persons_per_frame: Peak simultaneous person count across all frames.
            detector_model: Name/version of the detection model used.
            detector_confidence: Average detection confidence across all frames (0–1).
            output_path: Absolute path to the extracted clip file.
        """
        timestamp = datetime.now(UTC).isoformat()

        row = {
            "uuid": generate_uuid(),
            "archive_org_identifier": self._source_to_identifier.get(source_video, ""),
            "timestamp": timestamp,
            "source_video": source_video,
            "fps": round(fps, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_seconds": round(start_seconds, 2),
            "duration_seconds": round(duration_seconds, 2),
            "max_persons_per_frame": max_persons_per_frame,
            "detector_model": detector_model,
            "detector_confidence": round(detector_confidence, 3),
            "output_path": output_path,
        }

        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
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
                fieldnames = list(reader.fieldnames or [])

            if not entries:
                logger.info("Extraction log is empty.")
                return

            total_clips = len(entries)
            total_duration = sum(
                float(e["duration_seconds"]) for e in entries if e["duration_seconds"]
            )
            person_field = "max_persons_per_frame"
            if person_field not in fieldnames:
                # Keep compatibility with older CSV snapshots.
                person_field = "num_persons"
            total_persons = sum(int(e[person_field]) for e in entries if e.get(person_field))
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
