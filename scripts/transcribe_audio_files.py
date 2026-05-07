#!/usr/bin/env python3
"""
Transcribe standalone audio files from archive.org.

Scans archive_org_public_domain/audio/ for audio files (.mp3, .wav, etc)
with missing transcriptions, transcribes them using Whisper, and writes
.transcription.json sidecars.

Each transcription gets:
- UUID (unique identifier)
- Parent reference (audio filename)
- Transcriber metadata (model size, timestamp)
- Validation against transcription_schema.json

Writes transcription sidecars (.transcription.json) next to source audio files.
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from persondet.audio import AudioTranscriber
from persondet.config import DEFAULT_MODELS_PATH, get_log_level
from persondet.fair import (
    generate_uuid,
    reorganize_for_fair,
    validate_against_schema,
)
from persondet.pipeline_loggers import AudioTranscriptionsExtractionLogger
from persondet.script_utilities import _TqdmHandler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Audio file extensions to recognize
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"}


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for audio file transcription."""

    audio_files_dir: str
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "TranscriptionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        trans_config = config.get("transcription", {})
        return cls(
            audio_files_dir=trans_config.get(
                "audio_files_dir", "DARD/archive_org_public_domain/audio"
            ),
            overwrite=trans_config.get("overwrite", False),
        )


def scan_for_untranscribed_audio(audio_dir: Path, overwrite: bool = False) -> list:
    """Find all audio files that need transcription.

    Returns list of (media_path, trans_path) tuples.
    """
    audio_to_process = []

    if not audio_dir.exists():
        return audio_to_process

    # Recursively find all audio files (including in language subfolders)
    for audio_path in sorted(audio_dir.rglob("*")):
        if audio_path.is_dir():
            continue

        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Check for existing transcription
        trans_path = audio_path.with_stem(audio_path.stem + ".transcription")
        if trans_path.exists() and not overwrite:
            continue

        audio_to_process.append((audio_path, trans_path))

    return audio_to_process


def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting audio file transcription with FAIR integration...")

    cfg = TranscriptionConfig.from_yaml(str(CONFIG_PATH))

    audio_files_dir = Path(cfg.audio_files_dir)

    models_path = Path(DEFAULT_MODELS_PATH)

    # Setup Transcriber (using hardcoded 'small' model)
    model_size = "small"
    try:
        transcriber = AudioTranscriber(model_size=model_size, download_root=str(models_path))
        logger.info("Initialized Whisper model: %s", model_size)
    except Exception as e:
        logger.error("Failed to initialize Whisper: %s", e)
        sys.exit(1)

    # Initialize traceability logger
    transcription_logger = AudioTranscriptionsExtractionLogger(dard_root="DARD")

    # Find audio files needing transcription
    logger.info("Scanning for audio files needing transcription...")
    audio_files_list = scan_for_untranscribed_audio(audio_files_dir, overwrite=cfg.overwrite)
    logger.info(
        "Found %d audio files needing transcription in %s.",
        len(audio_files_list),
        audio_files_dir,
    )

    if not audio_files_list:
        logger.info("All audio files transcribed! Nothing to do.")
        return

    # Process Loop
    success_count = 0
    fail_count = 0

    for media_path, trans_path in tqdm(
        audio_files_list, desc="Transcribing audio files", unit="file"
    ):
        try:
            # Transcribe audio
            text = transcriber.transcribe_file(media_path)
            if not text:
                text = ""  # Ensure non-null string

            # Build transcription metadata with FAIR fields
            trans_meta: dict[str, Any] = {
                "uuid": str(generate_uuid()),
                "schema_version": "1.0",
                "transcription": text,
                "language": "en",  # Whisper language detection could enhance this
            }

            # Add source metadata
            trans_meta["source"] = {
                "archive_org_id": media_path.stem,
                "archive_org_url": "",  # Could be populated from metadata
                "license": "public-domain",
            }
            trans_meta["parent_audio"] = {
                "filename": media_path.name,
            }

            # Add transcriber-specific metadata
            trans_meta["transcriber"] = {
                "method": "openai_whisper",
                "model_size": "small",
            }
            trans_meta["transcribed_at"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017

            # Reorganize for FAIR
            trans_meta = reorganize_for_fair(trans_meta, "transcription")

            # Validate against schema
            try:
                validate_against_schema(trans_meta, "transcription")
            except Exception as e:
                logger.error("Validation failed for %s: %s", media_path.name, e)
                fail_count += 1
                continue

            # Write transcription sidecar
            trans_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trans_path, "w", encoding="utf-8") as f:
                json.dump(trans_meta, f, indent=2)

            # Log transcription to traceability CSV
            transcription_logger.log_audio_transcription(
                transcription_id=trans_meta.get("uuid", media_path.stem),
                source_audio=media_path.name,
                source_audio_path=str(media_path.absolute()),
                language_detected="en",  # Could be enhanced with Whisper lang detection
                confidence=1.0,  # Whisper doesn't provide per-file confidence
                duration_seconds=0.0,  # TODO: get from audio metadata
                model_version=model_size,
                output_path=str(trans_path.absolute()),
            )

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", media_path.name, e)
            fail_count += 1

    logger.info(
        "Audio transcription complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )
    transcription_logger.print_summary()


if __name__ == "__main__":
    main()
