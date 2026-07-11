#!/usr/bin/env python3
"""
Extract audio tracks from person clip videos as WAV files.

Scans extracted_person_clips/ for .mp4 files and extracts the audio track
from each one as a 16kHz mono WAV file — the standard format expected by
Whisper and other speech processing tools.

Output: <clip_stem>.wav written next to each source .mp4 file.
Resumable: skips clips that already have a .wav file alongside them.

All parameters are read from config.yaml under the 'person_extraction' key
(reads output_clips_dir from the existing section).
"""

import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from dardcollect.config import ClipExtractionConfig, get_log_level
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get("DARDCOLLECT_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
)

logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

# Output sample rate expected by Whisper and standard speech tools
_SAMPLE_RATE = 16000


@add_timer
def main() -> None:
    """Extract WAV audio tracks from all person clip .mp4 files."""
    try:
        clip_config = ClipExtractionConfig.from_yaml(str(CONFIG_PATH))
    except Exception as e:
        logger.error("Error loading config: %s", e)
        sys.exit(1)

    clips_dir = Path(clip_config.output_clips_dir)
    if not clips_dir.exists():
        logger.error("Person clips directory not found: %s", clips_dir)
        sys.exit(1)

    # Collect all .mp4 clips that don't yet have a .wav alongside them
    mp4_files = list(clips_dir.rglob("*.mp4"))
    pending = [f for f in mp4_files if not f.with_suffix(".wav").exists()]

    if not pending:
        logger.info("All clips already have WAV files. Nothing to do.")
        return

    logger.info(
        "Found %d clip(s) to process (%d already done)",
        len(pending),
        len(mp4_files) - len(pending),
    )

    # Import moviepy here — it's heavy and only needed at runtime
    try:
        from moviepy import VideoFileClip
    except ImportError:
        logger.error("moviepy is required: uv pip install moviepy")
        sys.exit(1)

    total_ok = 0
    total_fail = 0

    for clip_path in tqdm(pending, desc="Extracting audio", unit="clip"):
        wav_path = clip_path.with_suffix(".wav")
        try:
            with VideoFileClip(str(clip_path)) as video:
                if video.audio is None:
                    logger.warning("No audio track in %s — skipping", clip_path.name)
                    total_fail += 1
                    continue
                video.audio.write_audiofile(
                    str(wav_path),
                    fps=_SAMPLE_RATE,
                    nbytes=2,  # 16-bit PCM
                    ffmpeg_params=["-ac", "1"],  # mono
                    logger=None,  # suppress moviepy progress bar
                )
            total_ok += 1
        except Exception as e:
            logger.error("Failed to extract audio from %s: %s", clip_path.name, e)
            # Remove partial output if any
            if wav_path.exists():
                wav_path.unlink()
            total_fail += 1

    logger.info(
        "Summary: %d WAV file(s) written, %d failed",
        total_ok,
        total_fail,
    )


if __name__ == "__main__":
    main()
