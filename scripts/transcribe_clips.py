#!/usr/bin/env python3
"""
Transcription Script — FAIR-Compliant Audio Transcription

Transcribes audio from two sources:
1. Person clips (videos): Scans extracted_person_clips/ for .mp4 files with missing transcriptions
2. Audio files: Scans archive_org_public_domain/audio/ for .mp3/.wav/.aac/etc files

Each transcription gets:
- UUID (unique identifier)
- Parent reference (clip UUID or audio filename)
- Transcriber metadata (model size, timestamp)
- Validation against transcription_schema.json

Writes transcription sidecars (.transcription.json) next to source files.
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
    generate_uuid,
    reorganize_for_fair,
    validate_against_schema,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Audio file extensions to recognize
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"}


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """Configuration for audio transcription."""

    person_clips_dir: str
    audio_files_dir: str
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "TranscriptionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        trans_config = config.get("transcription", {})
        return cls(
            person_clips_dir=trans_config.get("person_clips_dir", "DARD/extracted_person_clips"),
            audio_files_dir=trans_config.get(
                "audio_files_dir", "DARD/archive_org_public_domain/audio"
            ),
            overwrite=trans_config.get("overwrite", False),
        )


def scan_for_untranscribed_clips(clips_dir: Path, overwrite: bool = False) -> list:
    """Find all .mp4 files in person clips directory that need transcription.

    Returns list of (media_path, json_path, trans_path, parent_sidecar, source_type) tuples.
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

            clips_to_process.append((mp4_path, json_path, trans_path, sidecar_data, "person_clip"))

        except Exception as e:
            logger.warning("Error reading %s: %s", json_path.name, e)

    return clips_to_process


def scan_for_untranscribed_audio(audio_dir: Path, overwrite: bool = False) -> list:
    """Find all audio files that need transcription.

    Returns list of (media_path, trans_path, source_type) tuples.
    """
    audio_to_process = []

    if not audio_dir.exists():
        return audio_to_process

    # Find all audio files
    for audio_path in sorted(audio_dir.glob("*")):
        if audio_path.is_dir():
            continue

        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Check for existing transcription
        trans_path = audio_path.with_stem(audio_path.stem + ".transcription")
        if trans_path.exists() and not overwrite:
            continue

        audio_to_process.append((audio_path, trans_path, "audio_file"))

    return audio_to_process


def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting transcription with FAIR integration...")

    cfg = TranscriptionConfig.from_yaml(str(CONFIG_PATH))

    person_clips_dir = Path(cfg.person_clips_dir)
    audio_files_dir = Path(cfg.audio_files_dir)

    if not person_clips_dir.exists():
        logger.warning("Person clips directory does not exist: %s", person_clips_dir)

    models_path = Path(DEFAULT_MODELS_PATH)

    # Setup Transcriber (using hardcoded 'small' model)
    model_size = "small"
    try:
        transcriber = AudioTranscriber(model_size=model_size, download_root=str(models_path))
        logger.info("Initialized Whisper model: %s", model_size)
    except Exception as e:
        logger.error("Failed to initialize Whisper: %s", e)
        sys.exit(1)

    # Find clips and audio files
    logger.info("Scanning for media needing transcription...")
    work_list = []

    if person_clips_dir.exists():
        person_clips_list = scan_for_untranscribed_clips(person_clips_dir, overwrite=cfg.overwrite)
        logger.info(
            "Found %d person clips needing transcription in %s.",
            len(person_clips_list),
            person_clips_dir,
        )
        work_list.extend(person_clips_list)

    audio_files_list = scan_for_untranscribed_audio(audio_files_dir, overwrite=cfg.overwrite)
    logger.info(
        "Found %d audio files needing transcription in %s.",
        len(audio_files_list),
        audio_files_dir,
    )
    work_list.extend(audio_files_list)

    if not work_list:
        logger.info("All done! Nothing to transcribe.")
        return

    # Process Loop
    success_count = 0
    fail_count = 0

    for item in tqdm(work_list, desc="Transcribing", unit="file"):
        try:
            if len(item) == 5:
                # Person clip: (mp4_path, json_path, trans_path, parent_sidecar, source_type)
                media_path, json_path, trans_path, parent_sidecar, source_type = item
                parent_uuid = parent_sidecar.get("uuid")
                if not parent_uuid:
                    logger.warning("No UUID in parent sidecar %s, skipping", json_path.name)
                    fail_count += 1
                    continue
                parent_file = json_path.name

            else:
                # Audio file: (audio_path, trans_path, source_type)
                media_path, trans_path, source_type = item
                parent_uuid = None
                parent_file = media_path.name

            # Transcribe audio
            text = transcriber.transcribe_file(media_path)
            if not text:
                text = ""  # Ensure non-null string

            # Build transcription metadata with FAIR fields
            trans_meta: dict[str, Any] = {
                "transcription": text,
                "language": "en",  # Whisper language detection could enhance this
            }

            # Add FAIR metadata
            if source_type == "person_clip":
                # Person clip: parent is the clip UUID
                trans_meta = add_fair_metadata(
                    trans_meta,
                    schema_type="transcription",
                    parent_uuid=parent_uuid,
                    parent_file=parent_file,
                )
            else:
                # Audio file: generate new UUID, parent is the audio filename
                trans_meta["uuid"] = str(generate_uuid())
                trans_meta["schema_version"] = "1.0"
                trans_meta["source"] = {
                    "archive_org_id": media_path.stem,
                    "archive_org_url": "",  # Could be populated from metadata
                    "license": "public-domain",
                }
                trans_meta["parent_audio"] = {
                    "filename": parent_file,
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

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", media_path.name, e)
            fail_count += 1

    logger.info(
        "Transcription complete — %d succeeded, %d failed",
        success_count,
        fail_count,
    )


if __name__ == "__main__":
    main()
