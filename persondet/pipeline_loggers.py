"""
Pipeline-stage logging for DARD traceability.

Each extraction stage (frames, face_crops, transcriptions, quality_annotation)
logs incrementally to track origin through the complete workflow.

All loggers follow the same pattern:
- Incremental append-only CSV writes (survive interruptions)
- ISO 8601 UTC timestamps
- Links to source artifacts (clip_id, face_crop_id, etc.)
- FAIR-compliant (UUID, source metadata)
"""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path


class FramesExtractionLogger:
    """Tracks frames extracted from person clips."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "frames_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("FramesExtractionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_frame_extraction(
        self,
        frame_id: str,
        source_clip: str,
        source_clip_path: str,
        frame_number: int,
        timestamp_seconds: float,
        output_path: str,
    ) -> None:
        """
        Log a frame extraction.

        Args:
            frame_id: Unique frame identifier (e.g., "clip_001_frame_00042")
            source_clip: Source person clip filename (e.g., "Finger_Man_02m09s-02m12s.mp4")
            source_clip_path: Full path to source clip
            frame_number: Frame index in source clip
            timestamp_seconds: Timestamp in source clip (seconds)
            output_path: Path to saved frame file
        """
        timestamp = datetime.now(UTC).isoformat()

        fieldnames = [
            "timestamp",
            "frame_id",
            "source_clip",
            "source_clip_path",
            "frame_number",
            "timestamp_seconds",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header if file is new
            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "source_clip": source_clip,
                    "source_clip_path": source_clip_path,
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print extraction statistics."""
        if not self.csv_path.exists():
            print("No frames extracted yet.")
            return

        total_frames = 0
        clips_processed = set()

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_frames += 1
                    clips_processed.add(row["source_clip"])

            print("\n📸 Frames Extraction Summary")
            print(f"  Total frames extracted: {total_frames}")
            print(f"  Clips processed: {len(clips_processed)}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading frames CSV: {e}")


class FaceCropsExtractionLogger:
    """Tracks face crops extracted from person clips or images."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("FaceCropsExtractionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_face_crop_extraction(
        self,
        crop_id: str,
        source_type: str,  # "person_clip" or "image"
        source_id: str,  # clip_id or image_id
        source_path: str,  # path to source
        face_bbox: str,  # "x1,y1,x2,y2"
        confidence: float,
        width: int,
        height: int,
        output_path: str,
    ) -> None:
        """
        Log a face crop extraction.

        Args:
            crop_id: Unique crop identifier
            source_type: "person_clip" or "image"
            source_id: Source clip/image identifier
            source_path: Full path to source
            face_bbox: Bounding box as "x1,y1,x2,y2"
            confidence: Detection confidence (0-1)
            width: Crop width in pixels
            height: Crop height in pixels
            output_path: Path to saved crop file
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        fieldnames = [
            "timestamp",
            "crop_id",
            "source_type",
            "source_id",
            "source_path",
            "face_bbox",
            "confidence",
            "width",
            "height",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "crop_id": crop_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_path": source_path,
                    "face_bbox": face_bbox,
                    "confidence": confidence,
                    "width": width,
                    "height": height,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print extraction statistics."""
        if not self.csv_path.exists():
            print("No face crops extracted yet.")
            return

        total_crops = 0
        by_source_type = {}

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_crops += 1
                    source_type = row["source_type"]
                    by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

            print("\n👤 Face Crops Extraction Summary")
            print(f"  Total crops extracted: {total_crops}")
            for stype, count in by_source_type.items():
                print(f"    {stype}: {count}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading face crops CSV: {e}")


class TranscriptionsExtractionLogger:
    """Tracks transcriptions extracted from audio/clips."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "transcriptions_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("TranscriptionsExtractionLogger")

        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_transcription(
        self,
        transcription_id: str,
        source_clip: str,
        source_clip_path: str,
        language_detected: str,
        confidence: float,
        word_count: int,
        duration_seconds: float,
        output_path: str,
        model_version: str = "whisper-small",
    ) -> None:
        """
        Log a transcription extraction.

        Args:
            transcription_id: Unique transcription identifier
            source_clip: Source clip filename
            source_clip_path: Full path to source clip
            language_detected: Detected language code (e.g., "en")
            confidence: Average confidence (0-1)
            word_count: Number of words in transcription
            duration_seconds: Duration of clip transcribed
            output_path: Path to saved transcription file
            model_version: Speech recognition model used
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        fieldnames = [
            "timestamp",
            "transcription_id",
            "source_clip",
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
                    "timestamp": timestamp,
                    "transcription_id": transcription_id,
                    "source_clip": source_clip,
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
        """Print transcription statistics."""
        if not self.csv_path.exists():
            print("No transcriptions extracted yet.")
            return

        total_transcriptions = 0
        languages = {}
        total_words = 0

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_transcriptions += 1
                    lang = row["language_detected"]
                    languages[lang] = languages.get(lang, 0) + 1
                    total_words += int(row["word_count"])

            print("\n🎙️ Transcriptions Extraction Summary")
            print(f"  Total transcriptions: {total_transcriptions}")
            print(f"  Total words: {total_words}")
            print("  Languages:")
            for lang, count in languages.items():
                print(f"    {lang}: {count}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading transcriptions CSV: {e}")


class FaceQualityAnnotationLogger:
    """Tracks quality annotations applied to face crops."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "face_quality_annotation.csv"
        self._header_written = False
        self.logger = logging.getLogger("FaceQualityAnnotationLogger")

        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_quality_annotation(
        self,
        crop_id: str,
        crop_path: str,
        sharpness: float,
        illumination: float,
        contrast: float,
        structure: float,
        completeness: float,
        eye_openness: float,
        mouth_openness: float,
        overall_score: float,
        passed_filter: bool,
    ) -> None:
        """
        Log face quality annotation.

        Args:
            crop_id: Face crop identifier
            crop_path: Path to crop file
            sharpness: OFIQ sharpness score (0-100)
            illumination: OFIQ illumination score (0-100)
            contrast: OFIQ contrast score (0-100)
            structure: OFIQ structure score (0-100)
            completeness: OFIQ completeness score (0-100)
            eye_openness: OFIQ eye openness score (0-100)
            mouth_openness: OFIQ mouth openness score (0-100)
            overall_score: Average of all scores
            passed_filter: Whether crop passed quality threshold
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        fieldnames = [
            "timestamp",
            "crop_id",
            "crop_path",
            "sharpness",
            "illumination",
            "contrast",
            "structure",
            "completeness",
            "eye_openness",
            "mouth_openness",
            "overall_score",
            "passed_filter",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "crop_id": crop_id,
                    "crop_path": crop_path,
                    "sharpness": round(sharpness, 2),
                    "illumination": round(illumination, 2),
                    "contrast": round(contrast, 2),
                    "structure": round(structure, 2),
                    "completeness": round(completeness, 2),
                    "eye_openness": round(eye_openness, 2),
                    "mouth_openness": round(mouth_openness, 2),
                    "overall_score": round(overall_score, 2),
                    "passed_filter": passed_filter,
                }
            )

    def print_summary(self) -> None:
        """Print quality annotation statistics."""
        if not self.csv_path.exists():
            print("No quality annotations yet.")
            return

        total_annotations = 0
        passed = 0
        failed = 0
        avg_scores = {}

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_annotations += 1
                    if row["passed_filter"].lower() == "true":
                        passed += 1
                    else:
                        failed += 1

                    for score_field in [
                        "sharpness",
                        "illumination",
                        "contrast",
                        "structure",
                        "completeness",
                        "eye_openness",
                        "mouth_openness",
                    ]:
                        score = float(row[score_field])
                        if score_field not in avg_scores:
                            avg_scores[score_field] = []
                        avg_scores[score_field].append(score)

            print("\n📊 Face Quality Annotation Summary")
            print(f"  Total annotations: {total_annotations}")
            print(f"  Passed filter: {passed} ({100 * passed / total_annotations:.1f}%)")
            print(f"  Failed filter: {failed} ({100 * failed / total_annotations:.1f}%)")
            print("  Average scores:")
            for field, scores in avg_scores.items():
                avg = sum(scores) / len(scores)
                print(f"    {field}: {avg:.2f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading quality annotation CSV: {e}")


class FilteredFaceCropsLogger:
    """Tracks face crops that pass quality filtering."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "filtered_face_crops.csv"
        self._header_written = False
        self.logger = logging.getLogger("FilteredFaceCropsLogger")

        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_filtered_crop(
        self,
        crop_id: str,
        source_crop_path: str,
        magface_score: float,
        filter_threshold: float,
        output_path: str,
    ) -> None:
        """
        Log a face crop that passed filtering.

        Args:
            crop_id: Face crop identifier
            source_crop_path: Path to source crop
            magface_score: MagFace quality score
            filter_threshold: Threshold used for filtering
            output_path: Path to filtered copy
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        fieldnames = [
            "timestamp",
            "crop_id",
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
                    "timestamp": timestamp,
                    "crop_id": crop_id,
                    "source_crop_path": source_crop_path,
                    "magface_score": round(magface_score, 3),
                    "filter_threshold": round(filter_threshold, 3),
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print filtering statistics."""
        if not self.csv_path.exists():
            print("No filtered crops yet.")
            return

        total_filtered = 0
        avg_magface = 0

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                scores = []
                for row in reader:
                    total_filtered += 1
                    scores.append(float(row["magface_score"]))

                if scores:
                    avg_magface = sum(scores) / len(scores)

            print("\n✨ Filtered Face Crops Summary")
            print(f"  Total crops passing filter: {total_filtered}")
            print(f"  Avg MagFace score: {avg_magface:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading filtered crops CSV: {e}")


class ImagePersonDetectionLogger:
    """Tracks person detections extracted from static images."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "image_person_detection.csv"
        self._header_written = False
        self.logger = logging.getLogger("ImagePersonDetectionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_image_detection(
        self,
        detection_id: str,
        source_image: str,
        source_image_path: str,
        num_persons: int,
        detector_model: str,
        detector_confidence: float,
        output_path: str,
    ) -> None:
        """
        Log person detections from an image.

        Args:
            detection_id: Unique detection identifier (e.g., "image_001")
            source_image: Source image filename
            source_image_path: Full path to source image
            num_persons: Number of persons detected
            detector_model: Model name (e.g., "yolox_tiny")
            detector_confidence: Average detection confidence
            output_path: Path to output detection JSON
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        fieldnames = [
            "timestamp",
            "detection_id",
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
                    "timestamp": timestamp,
                    "detection_id": detection_id,
                    "source_image": source_image,
                    "source_image_path": source_image_path,
                    "num_persons": num_persons,
                    "detector_model": detector_model,
                    "detector_confidence": round(detector_confidence, 3),
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print detection statistics."""
        if not self.csv_path.exists():
            print("No image detections yet.")
            return

        total_images = 0
        total_persons = 0
        avg_confidence = 0

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                confidences = []
                for row in reader:
                    total_images += 1
                    total_persons += int(row["num_persons"])
                    confidences.append(float(row["detector_confidence"]))

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)

            print("\n📷 Image Person Detection Summary")
            print(f"  Total images processed: {total_images}")
            print(f"  Total persons detected: {total_persons}")
            print(f"  Avg detection confidence: {avg_confidence:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image detection CSV: {e}")


class ImageFaceCropsExtractionLogger:
    """Tracks face crop extraction from static images."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "image_face_crops_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("ImageFaceCropsExtractionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_face_crop_extraction(
        self,
        crop_id: str,
        source_image: str,
        source_image_path: str,
        face_bbox: str,
        confidence: float,
        width: int,
        height: int,
        output_path: str,
    ) -> None:
        """
        Log face crop extraction from an image.

        Args:
            crop_id: Unique crop identifier (e.g., "img_001_face_0")
            source_image: Source image filename
            source_image_path: Full path to source image
            face_bbox: Bounding box as "x1,y1,x2,y2"
            confidence: Face detection confidence
            width: Crop width (should be 616)
            height: Crop height (should be 616)
            output_path: Path to output .jpg crop
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        fieldnames = [
            "timestamp",
            "crop_id",
            "source_image",
            "source_image_path",
            "face_bbox",
            "confidence",
            "width",
            "height",
            "output_path",
        ]

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not self._header_written and f.tell() == 0:
                writer.writeheader()
                self._header_written = True

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "crop_id": crop_id,
                    "source_image": source_image,
                    "source_image_path": source_image_path,
                    "face_bbox": face_bbox,
                    "confidence": round(confidence, 3),
                    "width": width,
                    "height": height,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print extraction statistics."""
        if not self.csv_path.exists():
            print("No face crops extracted from images yet.")
            return

        total_crops = 0
        avg_confidence = 0

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                confidences = []
                for row in reader:
                    total_crops += 1
                    confidences.append(float(row["confidence"]))

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)

            print("\n🖼️  Image Face Crops Extraction Summary")
            print(f"  Total crops extracted: {total_crops}")
            print(f"  Avg face confidence: {avg_confidence:.3f}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading image face crops CSV: {e}")


class AudioTranscriptionsExtractionLogger:
    """Tracks transcriptions extracted from audio files."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "audio_transcriptions_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("AudioTranscriptionsExtractionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_audio_transcription(
        self,
        transcription_id: str,
        source_audio: str,
        source_audio_path: str,
        language_detected: str,
        confidence: float,
        duration_seconds: float,
        model_version: str,
        output_path: str,
    ) -> None:
        """
        Log audio file transcription.

        Args:
            transcription_id: Unique transcription identifier
            source_audio: Source audio filename
            source_audio_path: Full path to source audio
            language_detected: Detected language (e.g., "en", "es")
            confidence: Transcription confidence
            duration_seconds: Audio duration in seconds
            model_version: Whisper model version (e.g., "small")
            output_path: Path to output .transcription.json
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        fieldnames = [
            "timestamp",
            "transcription_id",
            "source_audio",
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
                    "timestamp": timestamp,
                    "transcription_id": transcription_id,
                    "source_audio": source_audio,
                    "source_audio_path": source_audio_path,
                    "language_detected": language_detected,
                    "confidence": round(confidence, 3),
                    "duration_seconds": round(duration_seconds, 2),
                    "model_version": model_version,
                    "output_path": output_path,
                }
            )

    def print_summary(self) -> None:
        """Print transcription statistics."""
        if not self.csv_path.exists():
            print("No audio transcriptions yet.")
            return

        total_transcriptions = 0
        total_duration = 0
        languages = {}

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_transcriptions += 1
                    total_duration += float(row["duration_seconds"])
                    lang = row["language_detected"]
                    languages[lang] = languages.get(lang, 0) + 1

            print("\n🎵 Audio Transcriptions Summary")
            print(f"  Total transcriptions: {total_transcriptions}")
            print(f"  Total duration: {total_duration:.1f}s")
            print(f"  Languages: {languages}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading audio transcriptions CSV: {e}")


class DocumentTextExtractionLogger:
    """Tracks text extraction from documents."""

    def __init__(self, dard_root: str = "DARD"):
        self.dard_root = Path(dard_root)
        self.traceability_dir = self.dard_root / "traceability"
        self.csv_path = self.traceability_dir / "document_text_extraction.csv"
        self._header_written = False
        self.logger = logging.getLogger("DocumentTextExtractionLogger")

        # Create directory structure
        self.traceability_dir.mkdir(parents=True, exist_ok=True)

    def log_text_extraction(
        self,
        extraction_id: str,
        source_document: str,
        source_document_path: str,
        text_length: int,
        word_count: int,
        model_version: str,
        output_annotation_path: str,
        output_text_path: str,
    ) -> None:
        """
        Log text extraction from a document.

        Args:
            extraction_id: Unique extraction identifier
            source_document: Source document filename
            source_document_path: Full path to source document
            text_length: Length of extracted text in characters
            word_count: Number of words in extracted text
            model_version: OCR/extraction model version
            output_annotation_path: Path to output .annotation.json
            output_text_path: Path to output .text.txt
        """
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        fieldnames = [
            "timestamp",
            "extraction_id",
            "source_document",
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
                    "timestamp": timestamp,
                    "extraction_id": extraction_id,
                    "source_document": source_document,
                    "source_document_path": source_document_path,
                    "text_length": text_length,
                    "word_count": word_count,
                    "model_version": model_version,
                    "output_annotation_path": output_annotation_path,
                    "output_text_path": output_text_path,
                }
            )

    def print_summary(self) -> None:
        """Print extraction statistics."""
        if not self.csv_path.exists():
            print("No documents processed yet.")
            return

        total_documents = 0
        total_text_length = 0
        total_words = 0

        try:
            with open(self.csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_documents += 1
                    total_text_length += int(row["text_length"])
                    total_words += int(row["word_count"])

            print("\n📄 Document Text Extraction Summary")
            print(f"  Total documents processed: {total_documents}")
            print(f"  Total characters extracted: {total_text_length}")
            print(f"  Total words extracted: {total_words}")
            print(f"  Log file: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error reading document extraction CSV: {e}")
