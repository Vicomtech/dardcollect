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
from pathlib import Path
from threading import Lock

from dardcollect.fair import generate_uuid
from dardcollect.modality_loggers import _build_lookup, _guard_header
from dardcollect.provenance import now_iso

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
    """The fields of one clips_extraction.csv row (excluding uuid/timestamp).

    The clip's fps, frame span, seconds, duration and persons-per-frame are in
    the clip sidecar, not the CSV (one owner per fact — see docs/2-LINEAGE.md).
    """

    source_video: str
    detector_model: str
    detector_confidence: float
    output_path: str


class ExtractionLogger:
    """Append-only CSV logger for person clip extractions.

    Writes clips_extraction.csv to the output clips directory. Each row links a
    clip to its source video and, when a downloads.csv manifest is present, to
    the manifest row via ``download_uuid`` (the upstream artefact uuid).

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
                ``download_uuid`` by matching source video filenames.
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

        # Lookup {filename_downloaded → manifest-row uuid} from downloads.csv.
        # Links carry the upstream *artifact* uuid, the same space a sidecar's
        # parent link uses (see docs/2-LINEAGE.md).
        self._download_lookup = _build_lookup(
            downloads_csv_path, "filename_downloaded", key_transform=lambda p: Path(p).name
        )

        self.fieldnames = [
            "uuid",
            "download_uuid",
            "timestamp",
            "source_video",
            "detector_model",
            "detector_confidence",
            "output_path",
        ]
        # Note: downloads.csv gains title/creator/date/license columns from
        # _PIPELINE_FIELDS above (written by the download stage);
        # clips_extraction.csv keeps its own fixed fieldnames.
        _guard_header(self.log_path, self.fieldnames)

    def log_extraction(self, record: ClipRecord, artifact_uuid: str | None = None) -> None:
        """Append a clip extraction record to clips_extraction.csv.

        The row ``uuid`` is the clip sidecar's uuid (``artifact_uuid``) when the
        caller provides it, so CSV links and sidecar parent links live in one
        identity space. Generates a fresh uuid otherwise. Writes are atomic
        (append-only) so the file remains valid even if the process crashes.
        """
        timestamp = now_iso()

        row = {
            "uuid": artifact_uuid or generate_uuid(),
            "download_uuid": self._download_lookup.get(Path(record.source_video).name, ""),
            "timestamp": timestamp,
            "source_video": record.source_video,
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
        """Read clips_extraction.csv and log aggregate counts.

        Reports total clips, average detector confidence, and clips grouped by
        source video. Per-clip duration / persons live in the sidecars.
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
                "  Average detector confidence: %.3f\n"
                "  Clips by source:\n%s",
                self.log_path,
                total_clips,
                avg_confidence,
                source_summary,
            )

        except Exception as e:
            logger.error("Failed to compute summary: %s", e)
