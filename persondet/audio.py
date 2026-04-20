"""
Audio transcription module using OpenAI Whisper.
"""

import logging
import os
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
    """Handles audio extraction and transcription."""

    def __init__(self, model_size: str = "base", download_root: str | None = None):
        """Initialize the transcriber.

        :param model_size: Whisper model size (tiny, base, small, medium, large).
        :param download_root: Directory to store/load models from.
        """
        self.model_size = model_size
        self.download_root = download_root
        self._model: Any = None

    def _ensure_model_loaded(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info("Loading Whisper model: %s (path=%s)", self.model_size, self.download_root)
            # Whisper's type hints don't properly accept None for download_root
            # Call with explicit keyword argument passing
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
        """Transcribe a specific segment of a video.

        :param video_path: Path to the video file.
        :param start_time: Start time in seconds.
        :param end_time: End time in seconds.
        :return: Transcribed text.
        """
        self._ensure_model_loaded()

        try:
            # Extract audio using moviepy
            # Context manager ensures file is closed
            # Create a temp file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            try:
                # Load video clip (subclip) - this avoids loading full video to RAM
                with VideoFileClip(str(video_path)) as video:
                    # Clip boundaries
                    clip = video.subclipped(start_time, end_time)
                    # Write audio
                    clip.audio.write_audiofile(tmp_audio_path, logger=None)

                # Transcribe
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
        """Transcribe an audio/video file directly.

        :param file_path: Path to the media file.
        :return: Transcribed text.
        """
        self._ensure_model_loaded()
        try:
            # If it's a video file, extract audio first using moviepy to temp wav
            # Whisper CAN handle some formats directly but standardizing on wav via moviepy
            # ensures backend consistency.
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
