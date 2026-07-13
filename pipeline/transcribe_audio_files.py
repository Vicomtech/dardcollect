#!/usr/bin/env python3
"""
Transcribe standalone audio files from archive.org.

Scans archive_org_public_domain/audio/ for audio files (.mp3, .wav, etc)
with missing transcriptions, transcribes them using Whisper, and writes
.transcription.json sidecars to audio_transcription.output_dir (preserving
language subfolders from the source).

Each transcription gets:
- UUID (unique identifier)
- Parent reference (audio filename)
- Transcriber metadata (model size, timestamp)
- Validation against transcription_schema.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dardcollect.audio import AudioTranscriber, scan_for_untranscribed_audio
from dardcollect.config import DEFAULT_MODELS_PATH, AudioTranscriptionConfig, get_log_level
from dardcollect.fair import (
    generate_uuid,
    reorganize_for_fair,
    validate_against_schema,
)
from dardcollect.pipeline_loggers import AudioTranscriptionsExtractionLogger
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler

CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@add_timer
def main():
    """Transcribe standalone audio files from archive.org downloads.

    Scans the audio directory for files (.mp3, .wav, etc.) with missing
    .transcription.json sidecars, runs Whisper transcription (model: 'small'),
    and writes FAIR-compliant sidecars preserving the language subfolder
    structure from the source.

    Configuration is read from config.yaml under the 'audio_transcription' key.
    """
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting audio file transcription with FAIR integration...")

    cfg = AudioTranscriptionConfig.from_yaml(str(CONFIG_PATH))

    audio_files_dir = Path(cfg.audio_files_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    downloads_csv = audio_files_dir.parent / "downloads.csv"
    transcription_logger = AudioTranscriptionsExtractionLogger(
        output_dir=str(output_dir), downloads_csv_path=downloads_csv
    )

    # Find audio files needing transcription
    logger.info("Scanning for audio files needing transcription...")
    audio_files_list = scan_for_untranscribed_audio(
        audio_files_dir, output_dir, overwrite=cfg.overwrite
    )
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
            # Transcribe audio with timestamps
            result = transcriber.transcribe_with_timestamps(media_path)
            text = str(result.get("text", ""))
            language = str(result.get("language", "")) or "en"
            segments = result.get("segments", [])
            if not isinstance(segments, list):
                segments = []

            # Build transcription metadata with FAIR fields
            trans_meta: dict[str, Any] = {
                "uuid": str(generate_uuid()),
                "schema_version": "1.0",
                "transcription": text,
                "language": language,
                "segments": segments,  # [{start, end, text}, ...]
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
                source_audio_path=str(media_path.absolute()),
                language_detected=language,
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
