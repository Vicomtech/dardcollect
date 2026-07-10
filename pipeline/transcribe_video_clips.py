#!/usr/bin/env python3
"""
Transcribe audio from extracted person clip videos.

Scans extracted_person_clips/ for .mp4 files with missing transcriptions,
transcribes the audio using Whisper, and writes .transcription.json sidecars.

Each transcription gets:
- UUID (unique identifier)
- Parent reference (clip UUID)
- Transcriber metadata (model size, timestamp)
- Validation against transcription_schema.json

Writes transcription sidecars (.transcription.json) next to source video files.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dardcollect.audio import AudioTranscriber, scan_for_untranscribed_clips
from dardcollect.config import DEFAULT_MODELS_PATH, VideoTranscriptionConfig, get_log_level
from dardcollect.fair import (
    add_fair_metadata,
    reorganize_for_fair,
    validate_against_schema,
)
from dardcollect.pipeline_loggers import TranscriptionsExtractionLogger
from dardcollect.pipeline_utils import _TqdmHandler

CONFIG_PATH = Path(
    os.environ.get("DARDCOLLECT_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
)


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    """Transcribe audio from extracted person clip videos using OpenAI Whisper.

    Scans the person clips directory for .mp4 files with missing .transcription.json
    sidecars, runs Whisper transcription (model size: 'small'), and writes
    FAIR-compliant transcription sidecars with UUIDs, parent clip references,
    transcriber metadata, and schema validation.

    All configuration is read from config.yaml under the 'video_transcription' key.
    """
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting video clip transcription with FAIR integration...")

    cfg = VideoTranscriptionConfig.from_yaml(str(CONFIG_PATH))

    person_clips_dir = Path(cfg.person_clips_dir)

    if not person_clips_dir.exists():
        logger.warning("Person clips directory does not exist: %s", person_clips_dir)
        return

    models_path = Path(DEFAULT_MODELS_PATH)

    # Setup Transcriber (using hardcoded 'small' model)
    model_size = "small"
    try:
        transcriber = AudioTranscriber(model_size=model_size, download_root=str(models_path))
        logger.info("Initialized Whisper model: %s", model_size)
    except Exception as e:
        logger.error("Failed to initialize Whisper: %s", e)
        sys.exit(1)

    # Initialize transcriptions logger
    clips_csv = person_clips_dir / "clips_extraction.csv"
    trans_logger = TranscriptionsExtractionLogger(
        output_dir=str(person_clips_dir), clips_csv_path=clips_csv
    )

    # Find clips needing transcription
    logger.info("Scanning for video clips needing transcription...")
    clips_list = scan_for_untranscribed_clips(person_clips_dir, overwrite=cfg.overwrite)
    logger.info(
        "Found %d video clips needing transcription in %s.",
        len(clips_list),
        person_clips_dir,
    )

    if not clips_list:
        logger.info("All video clips transcribed! Nothing to do.")
        return

    # Process Loop
    success_count = 0
    fail_count = 0

    for media_path, json_path, trans_path, parent_sidecar in tqdm(
        clips_list, desc="Transcribing video clips", unit="file"
    ):
        try:
            parent_uuid = parent_sidecar.get("uuid")
            if not parent_uuid:
                logger.warning("No UUID in parent sidecar %s, skipping", json_path.name)
                fail_count += 1
                continue

            # Transcribe audio with timestamps
            result = transcriber.transcribe_with_timestamps(media_path)
            text = str(result.get("text", ""))
            language = str(result.get("language", "")) or "en"
            segments = result.get("segments", [])
            if not isinstance(segments, list):
                segments = []

            # Build transcription metadata with FAIR fields
            trans_meta: dict[str, Any] = {
                "transcription": text,
                "language": language,
                "segments": segments,  # [{start, end, text}, ...]
            }

            # Add FAIR metadata (parent is the clip UUID)
            trans_meta = add_fair_metadata(
                trans_meta,
                schema_type="transcription",
                parent_uuid=parent_uuid,
                parent_file=json_path.name,
            )

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

            # Log transcription extraction (for traceability)
            trans_logger.log_transcription(
                source_clip_path=str(media_path),
                language_detected=language,
                confidence=0.95,  # TODO: get from whisper if available
                word_count=len(text.split()) if text else 0,
                duration_seconds=0.0,  # TODO: get from clip metadata
                output_path=str(trans_path),
                model_version=f"whisper-{model_size}",
            )

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", media_path.name, e)
            fail_count += 1

    logger.info(
        "Video transcription complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )
    trans_logger.print_summary()


if __name__ == "__main__":
    main()
