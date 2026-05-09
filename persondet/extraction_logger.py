"""
Extraction logging for person clips.

Generates a CSV log that tracks each extracted clip, linking it to its source video.
Writes incrementally as clips are extracted, ensuring traceability even if process
is interrupted. Follows FAIR principles (Findable, Accessible, Interoperable, Reusable).
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from persondet.fair import generate_uuid

logger = logging.getLogger(__name__)


class ExtractionLogger:
    """
    CSV logger for clip extractions with incremental writes.

    Each extracted clip is logged to clips_extraction.csv co-located with the
    output clips directory, linking it to its source video.

    Principles:
    - Incremental: Writes each entry immediately to disk
    - Resilient: Survives process interruption
    - Traceable: Links each clip to source video and timestamps
    - FAIR-compliant: Machine-readable, uniquely identifiable, complete metadata
    """

    def __init__(
        self,
        output_dir: str | Path = "DARD/extracted_person_clips",
        downloads_csv_path: str | Path | None = None,
    ):
        """
        Initialize extraction logger.

        Args:
            output_dir: Directory where person clips are written (CSV goes here too)
            downloads_csv_path: Path to downloads.csv for linking clips to their source download
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
            "clip_id",
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
        """
        Log a clip extraction (incremental write to CSV).

        clip_id is derived from output_path stem — not passed separately.

        Args:
            source_video: Source video filename
            fps: Frames per second of the source video
            start_frame: Starting frame number in source
            end_frame: Ending frame number in source
            start_seconds: Starting time in seconds
            duration_seconds: Clip duration in seconds
            max_persons_per_frame: Peak simultaneous person count across all frames
            detector_model: Detector model name (e.g., "yolox-tiny")
            detector_confidence: Average detection confidence across all frames (0-1)
            output_path: Full path to extracted clip file
        """
        timestamp = datetime.now(UTC).isoformat()

        row = {
            "uuid": generate_uuid(),
            "archive_org_identifier": self._source_to_identifier.get(source_video, ""),
            "timestamp": timestamp,
            "clip_id": Path(output_path).stem,
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

        # Incremental write (append mode)
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)

                # Write header only if file is new
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True

                writer.writerow(row)

        except Exception as e:
            logger.error("Failed to write extraction log entry: %s", e)

    def print_summary(self) -> None:
        """
        Print summary statistics from the extraction log.

        Reads CSV file from disk to compute aggregate statistics.
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
            total_persons = sum(int(e["num_persons"]) for e in entries if e["num_persons"])
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
