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
    add_fair_metadata,
    reorganize_for_fair,
    validate_against_schema,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for video clip transcription."""

    person_clips_dir: str
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "TranscriptionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        trans_config = config.get("transcription", {})
        return cls(
            person_clips_dir=trans_config.get("person_clips_dir", "DARD/extracted_person_clips"),
            overwrite=trans_config.get("overwrite", False),
        )


def scan_for_untranscribed_clips(clips_dir: Path, overwrite: bool = False) -> list:
    """Find all .mp4 files in person clips directory that need transcription.

    Returns list of (media_path, json_path, trans_path, parent_sidecar) tuples.
    """
    clips_to_process = []

    # Find all json files (sidecars for person clips)
    json_files = sorted(clips_dir.glob("*.json"))

    for json_path in json_files:
        # Skip if this is already a transcription sidecar
        if json_path.name.endswith(".transcription.json"):
            continue

        # Check if corresponding mp4 exists
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue

        # Check for existing transcription
        trans_path = json_path.with_stem(json_path.stem + ".transcription")
        if trans_path.exists() and not overwrite:
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                sidecar_data = json.load(f)

            clips_to_process.append((mp4_path, json_path, trans_path, sidecar_data))

        except Exception as e:
            logger.warning("Error reading %s: %s", json_path.name, e)

    return clips_to_process


def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting video clip transcription with FAIR integration...")

    cfg = TranscriptionConfig.from_yaml(str(CONFIG_PATH))

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

            # Transcribe audio
            text = transcriber.transcribe_file(media_path)
            if not text:
                text = ""  # Ensure non-null string

            # Build transcription metadata with FAIR fields
            trans_meta: dict[str, Any] = {
                "transcription": text,
                "language": "en",  # Whisper language detection could enhance this
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

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", media_path.name, e)
            fail_count += 1

    logger.info(
        "Video transcription complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )


if __name__ == "__main__":
    main()
