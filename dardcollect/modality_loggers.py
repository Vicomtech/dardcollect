"""
Modality-track pipeline loggers (image / audio / document) + the shared CSV
lookup helper.

Split out of `pipeline_loggers.py` so neither file is a god-file. The video-track
loggers + the public re-export stay in `pipeline_loggers.py`, which imports
`_build_lookup` from here (one-way dependency — no circular import).

Each logger follows the same pattern as the video-track loggers: incremental
append-only CSV writes (survive interruptions), ISO 8601 UTC timestamps, uuid
per row, parent_uuid link to the upstream CSV row.
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

from dardcollect.fair import generate_uuid


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


class ImagePersonDetectionLogger:
    """Tracks person detections extracted from static images."""

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

    def log_image_detection(
        self,
        source_image_path: str,
        num_persons: int,
        detector_model: str,
        detector_confidence: float,
        output_path: str,
    ) -> None:
        # source_image (= filename) is the lookup key used by ImageFaceCropsExtractionLogger.
        source_image = Path(source_image_path).name
        fieldnames = [
            "uuid",
            "download_uuid",
            "timestamp",
            "source_image",
            "source_image_path",
            "num_persons",
            "detector_model",
            "detector_confidence",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "download_uuid": self._download_lookup.get(source_image, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "source_image": source_image,
                    "source_image_path": source_image_path,
                    "num_persons": num_persons,
                    "detector_model": detector_model,
                    "detector_confidence": round(detector_confidence, 3),
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No image detections yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            total_persons = sum(int(r["num_persons"]) for r in rows)
            confidences = [float(r["detector_confidence"]) for r in rows]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            print("\n📷 Image Person Detection Summary")
            print(f"  Total images processed: {len(rows)}")
            print(f"  Total persons detected: {total_persons}")
            print(f"  Avg detection confidence: {avg_conf:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image detection CSV: {e}")


class ImageFaceCropsExtractionLogger:
    """Tracks face crop extraction from static images."""

    def __init__(
        self,
        output_dir: str = "DARD/face_crops",
        image_detection_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "image_face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("ImageFaceCropsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # source_image column in image_person_detection.csv is the join key
        self._detection_lookup = _build_lookup(image_detection_csv_path, "source_image")

    def log_face_crop_extraction(
        self,
        source_image_path: str,
        face_bbox: str,
        confidence: float,
        output_path: str,
    ) -> None:
        fieldnames = [
            "uuid",
            "detection_uuid",
            "timestamp",
            "source_image_path",
            "face_bbox",
            "confidence",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "detection_uuid": self._detection_lookup.get(Path(source_image_path).name, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "source_image_path": source_image_path,
                    "face_bbox": face_bbox,
                    "confidence": round(confidence, 3),
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
            confidences = [float(r["confidence"]) for r in rows]
            avg = sum(confidences) / len(confidences) if confidences else 0.0
            print("\n🖼️  Image Face Crops Extraction Summary")
            print(f"  Total crops extracted: {len(rows)}")
            print(f"  Avg face confidence: {avg:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image face crops CSV: {e}")


class AudioTranscriptionsExtractionLogger:
    """Tracks transcriptions extracted from audio files."""

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

    def log_audio_transcription(
        self,
        source_audio_path: str,
        language_detected: str,
        confidence: float,
        duration_seconds: float,
        model_version: str,
        output_path: str,
    ) -> None:
        fieldnames = [
            "uuid",
            "download_uuid",
            "timestamp",
            "source_audio_path",
            "language_detected",
            "confidence",
            "duration_seconds",
            "model_version",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "download_uuid": self._download_lookup.get(Path(source_audio_path).name, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "source_audio_path": source_audio_path,
                    "language_detected": language_detected,
                    "confidence": round(confidence, 3),
                    "duration_seconds": round(duration_seconds, 2),
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
            total_duration = sum(float(r["duration_seconds"]) for r in rows)
            languages: dict[str, int] = {}
            for r in rows:
                languages[r["language_detected"]] = languages.get(r["language_detected"], 0) + 1
            print("\n🎵 Audio Transcriptions Summary")
            print(f"  Total transcriptions: {len(rows)}")
            print(f"  Total duration: {total_duration:.1f}s")
            print(f"  Languages: {languages}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading audio transcriptions CSV: {e}")


class DocumentTextExtractionLogger:
    """Tracks text extraction from documents."""

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

    def log_text_extraction(
        self,
        source_document_path: str,
        text_length: int,
        word_count: int,
        model_version: str,
        output_annotation_path: str,
        output_text_path: str,
    ) -> None:
        fieldnames = [
            "uuid",
            "download_uuid",
            "timestamp",
            "source_document_path",
            "text_length",
            "word_count",
            "model_version",
            "output_annotation_path",
            "output_text_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "download_uuid": self._download_lookup.get(Path(source_document_path).name, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "source_document_path": source_document_path,
                    "text_length": text_length,
                    "word_count": word_count,
                    "model_version": model_version,
                    "output_annotation_path": output_annotation_path,
                    "output_text_path": output_text_path,
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
            print(f"  Total characters extracted: {sum(int(r['text_length']) for r in rows)}")
            print(f"  Total words extracted: {sum(int(r['word_count']) for r in rows)}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading document extraction CSV: {e}")
