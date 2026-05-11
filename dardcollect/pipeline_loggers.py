"""
Pipeline-stage logging for DARDcollect traceability.

Each extraction stage logs incrementally to a CSV co-located with its output
artifacts (FAIR: data and metadata together). All loggers follow the same pattern:
- Incremental append-only CSV writes (survive interruptions)
- ISO 8601 UTC timestamps
- uuid per row for stable identification
- parent_uuid link to the upstream CSV row

Derived fields (IDs, short filenames) are computed internally — callers only
pass the authoritative values (full paths, measurements).
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


class FramesExtractionLogger:
    """Tracks frames extracted from person clips."""

    def __init__(
        self,
        output_dir: str = "DARD/extracted_frames",
        clips_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "frames_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("FramesExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_lookup = _build_lookup(
            clips_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )

    def log_frame_extraction(
        self,
        source_clip_path: str,
        frame_number: int,
        timestamp_seconds: float,
        output_path: str,
    ) -> None:
        fieldnames = [
            "uuid",
            "clip_uuid",
            "timestamp",
            "source_clip_path",
            "frame_number",
            "timestamp_seconds",
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
                    "clip_uuid": self._clip_lookup.get(Path(source_clip_path).stem, ""),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "source_clip_path": source_clip_path,
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
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

    def __init__(
        self,
        output_dir: str = "DARD/face_crops",
        clips_csv_path: Path | str | None = None,
        downloads_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("FaceCropsExtractionLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._clip_lookup = _build_lookup(
            clips_csv_path, "output_path", key_transform=lambda p: Path(p).stem
        )
        self._download_lookup = _build_lookup(downloads_csv_path, "filename_downloaded")

    def log_face_crop_extraction(
        self,
        source_type: str,  # "person_clip" or "image"
        source_path: str,
        face_bbox: str,  # "x1,y1,x2,y2"
        confidence: float,
        output_path: str,
    ) -> None:
        # crop_id (= output stem) is the lookup key used by FilteredFaceCropsLogger
        # and FaceQualityAnnotationLogger to resolve parent_uuid.
        crop_id = Path(output_path).stem
        if source_type == "person_clip":
            parent_uuid = self._clip_lookup.get(Path(source_path).stem, "")
        else:
            parent_uuid = self._download_lookup.get(Path(source_path).name, "")

        fieldnames = [
            "uuid",
            "parent_uuid",
            "timestamp",
            "crop_id",
            "source_type",
            "source_path",
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
                    "parent_uuid": parent_uuid,
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "crop_id": crop_id,
                    "source_type": source_type,
                    "source_path": source_path,
                    "face_bbox": face_bbox,
                    "confidence": confidence,
                    "output_path": output_path,
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

    def log_transcription(
        self,
        source_clip_path: str,
        language_detected: str,
        confidence: float,
        word_count: int,
        duration_seconds: float,
        output_path: str,
        model_version: str = "whisper-small",
    ) -> None:
        fieldnames = [
            "uuid",
            "clip_uuid",
            "timestamp",
            "source_clip_path",
            "language_detected",
            "confidence",
            "word_count",
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
                    "clip_uuid": self._clip_lookup.get(Path(source_clip_path).stem, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "source_clip_path": source_clip_path,
                    "language_detected": language_detected,
                    "confidence": confidence,
                    "word_count": word_count,
                    "duration_seconds": duration_seconds,
                    "model_version": model_version,
                    "output_path": output_path,
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


class FaceQualityAnnotationLogger:
    """Tracks quality annotations applied to face crops."""

    def __init__(
        self,
        output_dir: str = "DARD/filtered_face_crops",
        face_crops_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "face_quality_annotation.csv"
        self._header_written = False
        self.logger = logging.getLogger("FaceQualityAnnotationLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._crop_lookup = _build_lookup(face_crops_csv_path, "crop_id")

    def log_quality_annotation(
        self,
        crop_path: str,
        sharpness: float,
        compression_artifacts: float,
        expression_neutrality: float,
        no_head_coverings: float,
        face_occlusion_prevention: float,
        unified_score: float,
        yaw_quality: float,
        pitch_quality: float,
        roll_quality: float,
        passed_filter: bool,
    ) -> None:
        fieldnames = [
            "uuid",
            "crop_uuid",
            "timestamp",
            "crop_id",
            "crop_path",
            "sharpness",
            "compression_artifacts",
            "expression_neutrality",
            "no_head_coverings",
            "face_occlusion_prevention",
            "unified_score",
            "yaw_quality",
            "pitch_quality",
            "roll_quality",
            "passed_filter",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True
            crop_id = Path(crop_path).stem
            writer.writerow(
                {
                    "uuid": generate_uuid(),
                    "crop_uuid": self._crop_lookup.get(crop_id, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "crop_id": crop_id,
                    "crop_path": crop_path,
                    "sharpness": round(sharpness, 2),
                    "compression_artifacts": round(compression_artifacts, 2),
                    "expression_neutrality": round(expression_neutrality, 2),
                    "no_head_coverings": round(no_head_coverings, 2),
                    "face_occlusion_prevention": round(face_occlusion_prevention, 2),
                    "unified_score": round(unified_score, 2),
                    "yaw_quality": round(yaw_quality, 2),
                    "pitch_quality": round(pitch_quality, 2),
                    "roll_quality": round(roll_quality, 2),
                    "passed_filter": passed_filter,
                }
            )

    def print_summary(self) -> None:
        if not self.csv_path.exists():
            print("No quality annotations yet.")
            return
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            passed = sum(1 for r in rows if r["passed_filter"].lower() == "true")
            score_fields = [
                "sharpness",
                "compression_artifacts",
                "expression_neutrality",
                "no_head_coverings",
                "face_occlusion_prevention",
                "unified_score",
                "yaw_quality",
                "pitch_quality",
                "roll_quality",
            ]
            print("\n📊 Face Quality Annotation Summary")
            print(f"  Total annotations: {len(rows)}")
            print(f"  Passed filter: {passed} ({100 * passed / len(rows):.1f}%)")
            print("  Average scores (max per crop):")
            for field in score_fields:
                avg = sum(float(r[field]) for r in rows) / len(rows)
                print(f"    {field}: {avg:.2f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading quality annotation CSV: {e}")


class FilteredFaceCropsLogger:
    """Tracks face crops that pass quality filtering."""

    def __init__(
        self,
        output_dir: str = "DARD/filtered_face_crops",
        face_crops_csv_path: Path | str | None = None,
    ):
        self.csv_path = Path(output_dir) / "filtered_face_crops.csv"
        self._header_written = False
        self.logger = logging.getLogger("FilteredFaceCropsLogger")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._crop_lookup = _build_lookup(face_crops_csv_path, "crop_id")

    def log_filtered_crop(
        self,
        source_crop_path: str,
        magface_score: float,
        filter_threshold: float,
        output_path: str,
    ) -> None:
        fieldnames = [
            "uuid",
            "crop_uuid",
            "timestamp",
            "source_crop_path",
            "magface_score",
            "filter_threshold",
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
                    "crop_uuid": self._crop_lookup.get(Path(source_crop_path).stem, ""),
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
