"""
Pipeline-stage logging for DARDcollect traceability.

Each extraction stage logs incrementally to a CSV co-located with its output
artifacts (FAIR: data and metadata together). All loggers follow the same pattern:
- Incremental append-only CSV writes (survive interruptions)
- ISO 8601 UTC timestamps
- uuid per row = the produced artifact's sidecar uuid (stable identification)
- parent_uuid link to the upstream artifact

CSVs are the **event log**: identity + links + input/output paths + run-level
values. The authoritative per-artifact measurements live in the schema-validated
JSON sidecars; measures the sidecar already stores (frame counts, seconds,
duration, bbox, per-crop confidence, persons-per-frame) are NOT duplicated into
the CSV row (see docs/2-LINEAGE.md).

The image / audio / document loggers live in `dardcollect.modality_loggers`;
the shared `_build_lookup` / `_guard_header` helpers are imported from there.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import ClassVar

from dardcollect.fair import generate_uuid
from dardcollect.modality_loggers import _build_lookup, _guard_header
from dardcollect.provenance import now_iso

__all__ = [
    "FaceCropRecord",
    "FaceCropsExtractionLogger",
    "FilteredFaceCropsLogger",
    "FramesExtractionLogger",
    "TranscriptionRecord",
    "TranscriptionsExtractionLogger",
]


@dataclass
class FaceCropRecord:
    """The fields of one video_face_crops_extraction.csv row (excluding uuid/timestamp).

    ``artifact_uuid`` is the crop sidecar's uuid; when set it becomes the row
    ``uuid`` so CSV links and sidecar links share one identity space.
    """

    source_type: str  # "person_clip" or "image"
    source_path: str
    output_path: str
    artifact_uuid: str | None = None


@dataclass
class TranscriptionRecord:
    """The fields of one transcriptions_extraction.csv row (excluding uuid/timestamp).

    ``artifact_uuid`` is the transcription sidecar's uuid; when set it becomes
    the row ``uuid`` so CSV links and sidecar links share one identity space.
    """

    source_clip_path: str
    language_detected: str
    word_count: int
    output_path: str
    model_version: str = "whisper-small"
    artifact_uuid: str | None = None


class FramesExtractionLogger:
    """Tracks frames extracted from person clips."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "clip_uuid",
        "timestamp",
        "source_clip_path",
        "frame_number",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/extracted_frames",
        clips_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "frames_extraction.csv"
        self._header_written = False
        # extract_frames_from_videos.py can extract several clips concurrently
        # (frame_extraction.workers), and every worker logs into this one CSV.
        # Without the lock two threads can both observe an empty file and each
        # write a header row, and interleave partially-written rows.
        self._write_lock = Lock()
        self.logger = logging.getLogger("FramesExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_lookup = _build_lookup(
            clips_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )
        _guard_header(self.csv_path, self.fieldnames)

    def log_frame_extraction(
        self,
        source_clip_path: str,
        frame_number: int,
        output_path: str,
        artifact_uuid: str | None = None,
    ) -> None:
        with self._write_lock, open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": artifact_uuid or generate_uuid(),
                    "clip_uuid": self._clip_lookup.get(Path(source_clip_path).stem, ""),
                    "timestamp": now_iso(),
                    "source_clip_path": source_clip_path,
                    "frame_number": frame_number,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No frames extracted yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            clips = {Path(r["source_clip_path"]).name for r in rows}
            print("\n📸 Frames Extraction Summary")
            print(f"  Total frames extracted: {len(rows)}")
            print(f"  Clips processed: {len(clips)}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading frames CSV: {e}")


class FaceCropsExtractionLogger:
    """Tracks face crops extracted from person clips or images."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "parent_uuid",
        "timestamp",
        "source_type",
        "source_path",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/face_crops",
        clips_csv_path: Path | str | None = None,
        downloads_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "video_face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("FaceCropsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_lookup = _build_lookup(
            clips_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )
        self._download_lookup = _build_lookup(downloads_csv_path, "filename_downloaded")
        _guard_header(self.csv_path, self.fieldnames)

    def log_face_crop_extraction(self, record: FaceCropRecord) -> None:
        if record.source_type == "person_clip":
            parent_uuid = self._clip_lookup.get(Path(record.source_path).stem, "")
        else:
            parent_uuid = self._download_lookup.get(Path(record.source_path).name, "")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": record.artifact_uuid or generate_uuid(),
                    "parent_uuid": parent_uuid,
                    "timestamp": now_iso(),
                    "source_type": record.source_type,
                    "source_path": record.source_path,
                    "output_path": record.output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No face crops extracted yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            by_type: dict[str, int] = {}
            for r in rows:
                t = r["source_type"]
                by_type[t] = by_type.get(t, 0) + 1
            print("\n👤 Face Crops Extraction Summary")
            print(f"  Total crops extracted: {len(rows)}")
            for stype, count in by_type.items():
                print(f"    {stype}: {count}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading face crops CSV: {e}")


class TranscriptionsExtractionLogger:
    """Tracks transcriptions extracted from person clips."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "clip_uuid",
        "timestamp",
        "source_clip_path",
        "language_detected",
        "word_count",
        "model_version",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/extracted_person_clips",
        clips_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "transcriptions_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("TranscriptionsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_lookup = _build_lookup(
            clips_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )
        _guard_header(self.csv_path, self.fieldnames)

    def log_transcription(self, record: TranscriptionRecord) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": record.artifact_uuid or generate_uuid(),
                    "clip_uuid": self._clip_lookup.get(Path(record.source_clip_path).stem, ""),
                    "timestamp": now_iso(),
                    "source_clip_path": record.source_clip_path,
                    "language_detected": record.language_detected,
                    "word_count": record.word_count,
                    "model_version": record.model_version,
                    "output_path": record.output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No transcriptions extracted yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            languages: dict[str, int] = {}
            total_words = 0
            for r in rows:
                languages[r["language_detected"]] = languages.get(r["language_detected"], 0) + 1
                total_words += int(r["word_count"])
            print("\n🎙️ Transcriptions Extraction Summary")
            print(f"  Total transcriptions: {len(rows)}")
            print(f"  Total words: {total_words}")
            for lang, count in languages.items():
                print(f"    {lang}: {count}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading transcriptions CSV: {e}")


class FilteredFaceCropsLogger:
    """Tracks face crops that pass quality filtering (a decision, no new artifact)."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "crop_uuid",
        "timestamp",
        "source_crop_path",
        "magface_score",
        "filter_threshold",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/filtered_face_crops",
        face_crops_csv_path: Path | str | None = None,
    ):
        # Auto-detect modality based on input CSV name
        modality = (
            "image"
            if face_crops_csv_path and "image_face_crops_extraction.csv" in str(face_crops_csv_path)
            else "video"
        )
        self.csv_path = Path(output_dir) / f"{modality}_filtered_face_crops.csv"
        self._header_written = False
        self.logger = logging.getLogger("FilteredFaceCropsLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # The crop row's output stem is the join key.
        self._crop_lookup = _build_lookup(
            face_crops_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )
        _guard_header(self.csv_path, self.fieldnames)

    def log_filtered_crop(
        self,
        source_crop_path: str,
        magface_score: float,
        filter_threshold: float,
        output_path: str,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    # The filter decision is its own event; it produces no sidecar.
                    "uuid": generate_uuid(),
                    "crop_uuid": self._crop_lookup.get(Path(source_crop_path).stem, ""),
                    "timestamp": now_iso(),
                    "source_crop_path": source_crop_path,
                    "magface_score": round(magface_score, 3),
                    "filter_threshold": round(filter_threshold, 3),
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No filtered crops yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            scores = [float(r["magface_score"]) for r in rows]
            avg = sum(scores) / len(scores) if scores else 0.0
            print("\n✨ Filtered Face Crops Summary")
            print(f"  Total crops passing filter: {len(rows)}")
            print(f"  Avg MagFace score: {avg:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading filtered crops CSV: {e}")
