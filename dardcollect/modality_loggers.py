"""
Modality-track pipeline loggers (image / audio / document) + the shared CSV
helpers.

Split out of `pipeline_loggers.py` so neither file is a god-file. The video-track
loggers stay in `pipeline_loggers.py`, which imports `_build_lookup` from here
(one-way dependency — no circular import).

Each logger follows the same pattern as the video-track loggers: incremental
append-only CSV writes (survive interruptions), ISO 8601 UTC timestamps, uuid
per row, parent_uuid link to the upstream artifact. CSVs are the **event log**
(identity + links + input/output paths + run-level values); the authoritative
per-artifact measurements live in the schema-validated JSON sidecars (see the
`pipeline_loggers` module docstring and docs/2-LINEAGE.md).
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dardcollect.fair import generate_uuid
from dardcollect.provenance import now_iso


def _build_lookup(
    csv_path: Path | str | None,
    key_field: str,
    key_transform=None,
) -> dict[str, str]:
    """Build a {key_field_value: uuid} lookup dict from a pipeline CSV.

    Args:
        csv_path: Path to the CSV file to read.
        key_field: Name of the field to use as lookup key.
        key_transform: Optional callable applied to the raw field value before indexing.

    Returns:
        dict: Mapping from transformed key values to their corresponding UUIDs.
    """
    if not csv_path:
        return {}
    p = Path(csv_path)
    if not p.exists():
        return {}
    result: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get(key_field, "")
            if key_transform:
                key = key_transform(key)
            uid = row.get("uuid", "")
            if key and uid:
                result[key] = uid
    return result


def _guard_header(csv_path: Path, fieldnames: list[str]) -> None:
    """Fail loudly when an existing CSV's header differs from the expected one.

    A logger appends without rewriting the header, so a stale header would land
    values in the wrong columns. Call after ``self.fieldnames`` is set; the
    message names the migration that fixes it.
    """
    if not (csv_path.exists() and csv_path.stat().st_size > 0):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        existing = next(csv.reader(f), [])
    if existing != fieldnames:
        raise ValueError(
            f"{csv_path} has a stale header (expected {fieldnames}, found {existing}). "
            f"Run `uv run python scripts/migrate_csv_schema.py --dard-root <root> --apply` "
            f"to migrate it, or remove the file to regenerate."
        )


@dataclass
class ImageDetectionRecord:
    """The fields of one image_person_detection.csv row (excluding uuid/timestamp).

    ``artifact_uuid`` is the detection sidecar's uuid; when set it becomes the
    row ``uuid`` so CSV links and sidecar links share one identity space.
    """

    source_image_path: str
    detector_model: str
    detector_confidence: float
    output_path: str
    artifact_uuid: str | None = None


class ImagePersonDetectionLogger:
    """Tracks person detections extracted from static images."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "download_uuid",
        "timestamp",
        "source_image_path",
        "detector_model",
        "detector_confidence",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/extracted_image_detections",
        downloads_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "image_person_detection.csv"
        self._header_written = False
        self.logger = logging.getLogger("ImagePersonDetectionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._download_lookup = _build_lookup(downloads_csv_path, "filename_downloaded")
        _guard_header(self.csv_path, self.fieldnames)

    def log_image_detection(self, record: ImageDetectionRecord) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": record.artifact_uuid or generate_uuid(),
                    "download_uuid": self._download_lookup.get(
                        Path(record.source_image_path).name, ""
                    ),
                    "timestamp": now_iso(),
                    "source_image_path": record.source_image_path,
                    "detector_model": record.detector_model,
                    "detector_confidence": round(record.detector_confidence, 3),
                    "output_path": record.output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No image detections yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            confidences = [float(r["detector_confidence"]) for r in rows]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            print("\n📷 Image Person Detection Summary")
            print(f"  Total images processed: {len(rows)}")
            print(f"  Avg detection confidence: {avg_conf:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image detection CSV: {e}")


class ImageFaceCropsExtractionLogger:
    """Tracks face crop extraction from static images."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "detection_uuid",
        "timestamp",
        "source_image_path",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/face_crops",
        image_detection_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "image_face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("ImageFaceCropsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # source_image_path basename in image_person_detection.csv is the join key
        self._detection_lookup = _build_lookup(
            image_detection_csv_path, "source_image_path", key_transform=lambda p: Path(p).name
        )
        _guard_header(self.csv_path, self.fieldnames)

    def log_face_crop_extraction(
        self,
        source_image_path: str,
        output_path: str,
        artifact_uuid: str | None = None,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": artifact_uuid or generate_uuid(),
                    "detection_uuid": self._detection_lookup.get(Path(source_image_path).name, ""),
                    "timestamp": now_iso(),
                    "source_image_path": source_image_path,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No face crops extracted from images yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            print("\n🖼️  Image Face Crops Extraction Summary")
            print(f"  Total crops extracted: {len(rows)}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image face crops CSV: {e}")


class AudioTranscriptionsExtractionLogger:
    """Tracks transcriptions extracted from audio files."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "download_uuid",
        "timestamp",
        "source_audio_path",
        "language_detected",
        "model_version",
        "output_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/audio_transcriptions",
        downloads_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "audio_transcriptions_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("AudioTranscriptionsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._download_lookup = _build_lookup(downloads_csv_path, "filename_downloaded")
        _guard_header(self.csv_path, self.fieldnames)

    def log_audio_transcription(
        self,
        source_audio_path: str,
        language_detected: str,
        model_version: str,
        output_path: str,
        artifact_uuid: str | None = None,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": artifact_uuid or generate_uuid(),
                    "download_uuid": self._download_lookup.get(Path(source_audio_path).name, ""),
                    "timestamp": now_iso(),
                    "source_audio_path": source_audio_path,
                    "language_detected": language_detected,
                    "model_version": model_version,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No audio transcriptions yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            languages: dict[str, int] = {}
            for r in rows:
                languages[r["language_detected"]] = languages.get(r["language_detected"], 0) + 1
            print("\n🎵 Audio Transcriptions Summary")
            print(f"  Total transcriptions: {len(rows)}")
            print(f"  Languages: {languages}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading audio transcriptions CSV: {e}")


@dataclass
class TextExtractionRecord:
    """The fields of one document_text_extraction.csv row (excluding uuid/timestamp)."""

    source_document_path: str
    model_version: str
    output_annotation_path: str
    output_text_path: str


class DocumentTextExtractionLogger:
    """Tracks text extraction from documents."""

    fieldnames: ClassVar[list[str]] = [
        "uuid",
        "download_uuid",
        "timestamp",
        "source_document_path",
        "model_version",
        "output_annotation_path",
        "output_text_path",
    ]

    def __init__(
        self,
        output_dir: str = "DARD/preprocessed_documents",
        downloads_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "document_text_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("DocumentTextExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._download_lookup = _build_lookup(downloads_csv_path, "filename_downloaded")
        _guard_header(self.csv_path, self.fieldnames)

    def log_text_extraction(
        self,
        record: TextExtractionRecord,
        artifact_uuid: str | None = None,
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": artifact_uuid or generate_uuid(),
                    "download_uuid": self._download_lookup.get(
                        Path(record.source_document_path).name, ""
                    ),
                    "timestamp": now_iso(),
                    "source_document_path": record.source_document_path,
                    "model_version": record.model_version,
                    "output_annotation_path": record.output_annotation_path,
                    "output_text_path": record.output_text_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No documents processed yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            print("\n📄 Document Text Extraction Summary")
            print(f"  Total documents processed: {len(rows)}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading document extraction CSV: {e}")
