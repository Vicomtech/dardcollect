"""
Audio transcription via OpenAI Whisper and ffmpeg audio mux helpers.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import whisper
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

# Ensure ffmpeg is in PATH for Whisper to use
# Whisper invokes 'ffmpeg' command via subprocess, so it requires it in system PATH.
# MoviePy uses imageio_ffmpeg, so we can borrow that binary.
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
        logger.debug("Added ffmpeg to PATH: %s", ffmpeg_dir)
except Exception as e:
    logger.warning("Failed to add ffmpeg to PATH: %s", e)


class AudioTranscriber:
    """Lazy-loading Whisper transcriber. Model is not loaded until first use."""

    def __init__(self, model_size: str = "base", download_root: str | None = None):
        """
        :param model_size: Whisper model size (tiny, base, small, medium, large).
        :param download_root: Directory to cache/load Whisper model weights.
        """
        self.model_size = model_size
        self.download_root = download_root
        self._model: Any = None

    def _ensure_model_loaded(self):
        if self._model is None:
            logger.info("Loading Whisper model: %s (path=%s)", self.model_size, self.download_root)
            # Whisper's type hints don't properly accept None for download_root
            if self.download_root is not None:
                self._model = whisper.load_model(
                    name=self.model_size,
                    download_root=self.download_root,  # type: ignore[arg-type]
                )
            else:
                self._model = whisper.load_model(self.model_size)

    def transcribe_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> str:
        """Transcribe a time-bounded segment of a video.

        Extracts audio to a temporary WAV via moviepy, then runs Whisper.
        Returns empty string on failure.

        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :return: Transcribed text.
        """
        self._ensure_model_loaded()

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            try:
                with VideoFileClip(str(video_path)) as video:
                    clip = video.subclipped(start_time, end_time)
                    clip.audio.write_audiofile(tmp_audio_path, logger=None)

                result = self._model.transcribe(tmp_audio_path)
                return result["text"].strip()

            finally:
                # Cleanup temp file
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

        except Exception as e:
            logger.error(
                "Error transcribing %s (%.1f-%.1f): %s", video_path.name, start_time, end_time, e
            )
            return ""

    def transcribe_file(self, file_path: Path) -> str:
        """Transcribe an audio or video file.

        Always converts to WAV via moviepy before calling Whisper — ensures
        consistent behaviour across formats (Whisper can decode some formats
        natively but wav avoids edge cases). Returns empty string on failure.
        """
        self._ensure_model_loaded()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            try:
                with VideoFileClip(str(file_path)) as video:
                    video.audio.write_audiofile(tmp_audio_path, logger=None)

                result = self._model.transcribe(tmp_audio_path)
                return result["text"].strip()
            finally:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)

        except Exception as e:
            logger.error("Error transcribing file %s: %s", file_path.name, e)
            return ""


# ── Audio file extensions recognized by scan functions ────────────────────────

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma"}


# ── IO helpers (consolidated from extraction scripts) ─────────────────────────


def _mux_audio(
    source_path: Path,
    face_crop_path: Path,
    start_t: float,
    end_t: float,
) -> None:
    """Replace a video-only face crop file with one that includes audio."""
    from dardcollect.pipeline_utils import _cleanup_files

    tmp_path = face_crop_path.with_suffix(".tmp.mp4")
    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(face_crop_path),
                "-ss",
                str(start_t),
                "-to",
                str(end_t),
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(tmp_path),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "  ffmpeg audio mux failed for %s — keeping video-only file.\n  %s",
                face_crop_path.name,
                result.stderr.decode(errors="replace").strip(),
            )
            _cleanup_files(tmp_path)
            return
        tmp_path.replace(face_crop_path)
    except Exception as e:
        logger.warning("  Audio mux error for %s: %s", face_crop_path.name, e)
        _cleanup_files(tmp_path)


def scan_for_untranscribed_audio(
    audio_dir: Path, output_dir: Path, overwrite: bool = False
) -> list:
    """Scan audio_dir for files lacking a transcription sidecar.

    Returns list of (audio_path, trans_path) tuples. trans_path mirrors
    the language subfolder structure of audio_dir inside output_dir, e.g.
    audio_dir/eng/file.mp3 → output_dir/eng/file.transcription.json.
    """
    audio_to_process = []

    if not audio_dir.exists():
        return audio_to_process

    for audio_path in sorted(audio_dir.rglob("*")):
        if audio_path.is_dir():
            continue

        if audio_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue

        # Mirror the relative subfolder structure (e.g. eng/, spa/) in output_dir
        rel = audio_path.relative_to(audio_dir)
        trans_path = output_dir / rel.parent / (audio_path.stem + ".transcription.json")

        if trans_path.exists() and not overwrite:
            continue

        audio_to_process.append((audio_path, trans_path))

    return audio_to_process


def scan_for_untranscribed_clips(clips_dir: Path, overwrite: bool = False) -> list:
    """Scan clips_dir for .mp4 files lacking a transcription sidecar.

    Returns list of (media_path, json_path, trans_path, parent_sidecar) tuples,
    where parent_sidecar is the already-parsed clip sidecar dict.
    Skips .mp4 files with no sibling .json, and .transcription.json files.
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
