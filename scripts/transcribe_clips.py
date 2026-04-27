#!/usr/bin/env python3
"""
Transcription Script
===================

Scans the extraction output directory for clips that are missing transcription
and processes them using OpenAI Whisper (via persondet.audio).

Updates the sidecar JSON files in-place.
"""

import json
import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from persondet.audio import AudioTranscriber
from persondet.config import DEFAULT_MODELS_PATH, get_log_level

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_for_untranscribed(clips_dir: Path) -> list:
    """Find all .mp4 files that need transcription."""
    clips_to_process = []

    # Recursively find all json files (starting point)
    json_files = list(clips_dir.rglob("*.json"))

    for json_path in json_files:
        # Check if corresponding mp4 exists
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check if transcription is missing or empty
            if not data.get("transcription"):
                clips_to_process.append((mp4_path, json_path))

        except Exception as e:
            logger.warning("Error reading %s: %s", json_path.name, e)

    return clips_to_process


def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    logger.info("Starting transcription...")

    config = load_config()

    # Get config values
    clips_dir = Path(config["person_extraction"]["output_clips_dir"])
    if not clips_dir.exists():
        logger.error("Output dir does not exist: %s", clips_dir)
        sys.exit(1)

    model_size = config["person_extraction"].get("transcription_model_size", "small")
    models_path = Path(config["person_extraction"].get("models_path", DEFAULT_MODELS_PATH))

    # Setup Transcriber
    # Re-use logic for local model path
    size_map = {
        "small": "openai_whisper_small.pt",
        "medium": "audio_m.pt",
        "large": "audio_l.pt",
        "large-v3": "audio_l.pt",
    }

    model_file = model_size
    if model_size in size_map:
        local_path = models_path / size_map[model_size]
        if local_path.exists():
            model_file = str(local_path)
            logger.info("Using local Whisper model: %s", local_path.name)

    try:
        transcriber = AudioTranscriber(model_size=model_file, download_root=str(models_path))
    except Exception as e:
        logger.error("Failed to initialize Whisper: %s", e)
        sys.exit(1)

    # Find clips
    logger.info("Scanning %s for clips...", clips_dir)
    work_list = scan_for_untranscribed(clips_dir)
    logger.info("Found %d clips needing transcription.", len(work_list))

    if not work_list:
        logger.info("All done! Nothing to transcribe.")
        return

    # Process Loop
    success_count = 0
    fail_count = 0

    for mp4_path, json_path in tqdm(work_list, desc="Transcribing"):
        try:
            # Transcribe
            text = transcriber.transcribe_file(mp4_path)
            if not text:
                text = ""  # Ensure string

            # Update JSON
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            data["transcription"] = text

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            success_count += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", mp4_path.name, e)
            fail_count += 1

    logger.info("Done — %d succeeded, %d failed", success_count, fail_count)


if __name__ == "__main__":
    main()
