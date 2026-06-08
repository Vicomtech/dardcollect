"""Audio transcription via openai-whisper and ffmpeg audio mux helpers.

Provides `AudioTranscriber` for transcribing video/audio files using
openai-whisper, with automatic audio extraction via moviepy.

Also includes helper functions for:
    - Replacing video-only face crops with audio-muxed versions.
    - Scanning directories for untranscribed audio and video clips.

ffmpeg from imageio-ffmpeg is automatically added to PATH for audio extraction.
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
from moviepy import AudioFileClip, VideoFileClip

logger = logging.getLogger(__name__)

# Ensure ffmpeg is in PATH for audio extraction
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
    """Lazy-loading openai-whisper transcriber.

    The Whisper model is not loaded into memory until the first transcription
    call, reducing startup time and memory footprint when transcription is
    not needed.
    """

    def __init__(
        self,
        model_size: str = "base",
        download_root: str | None = None,
        device: str | None = None,
    ):
        """Initialize the transcriber without loading the model.

        Args:
            model_size: Whisper model size. Options: "tiny", "base", "small",
                "medium", "large". Larger models are more accurate but slower.
            download_root: Directory to cache downloaded Whisper model weights.
                If None, uses the default cache location.
            device: Device to run inference on. None auto-selects CUDA if available,
                otherwise CPU. Can also be "cpu" or "cuda".
        """
        self.model_size = model_size
        self.download_root = download_root
        self.device = device
        self._model: whisper.Whisper | None = None

    def _ensure_model_loaded(self):
        """Load the Whisper model if not already loaded."""
        if self._model is None:
            logger.info(
                "Loading whisper model: %s (device=%s)",
                self.model_size,
                self.device or "auto",
            )
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=self.download_root,
            )

    def transcribe_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> str:
        """Transcribe a time-bounded segment of a video file.

        Extracts the audio segment to a temporary WAV file via moviepy,
        then runs whisper transcription. Cleans up the temp file afterwards.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            str: Transcribed text, stripped of leading/trailing whitespace.
                Returns empty string on failure.
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
        """Transcribe an entire audio or video file.

        Always converts to WAV via moviepy before calling whisper. This ensures
        consistent behavior across formats.

        Args:
            file_path: Path to the audio or video file.

        Returns:
            str: Transcribed text, stripped of leading/trailing whitespace.
                Returns empty string on failure.
        """
        self._ensure_model_loaded()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name

            try:
                # Audio-only files have no video track — VideoFileClip raises on them.
                # Use AudioFileClip for audio, VideoFileClip (.audio) for video.
                if file_path.suffix.lower() in _AUDIO_EXTENSIONS:
                    with AudioFileClip(str(file_path)) as audio:
                        audio.write_audiofile(tmp_audio_path, logger=None)
                else:
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
    """Replace a video-only face crop with a version that includes audio from the source.

    Uses ffmpeg to mux the original audio track into the face crop video,
    trimming to the clip's time bounds.

    Args:
        source_path: Path to the original full-length video (audio source).
        face_crop_path: Path to the video-only face crop file (will be replaced).
        start_t: Start time in seconds for audio extraction.
        end_t: End time in seconds for audio extraction.
    """
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
    """Scan a directory for audio files without transcription sidecars.

    Mirrors the language subfolder structure from audio_dir into output_dir.
    For example, audio_dir/eng/file.mp3 produces output_dir/eng/file.transcription.json.

    Args:
        audio_dir: Directory containing audio files (may have language subfolders).
        output_dir: Directory where transcription sidecars will be written.
        overwrite: If True, include files that already have transcription sidecars.

    Returns:
        list: Tuples of (audio_path, trans_path) for files needing transcription.
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
    """Scan a directory for video clips without transcription sidecars.

    Finds .mp4 files that have a sibling .json sidecar (but not a .transcription.json).
    Parses the clip sidecar to extract metadata needed for transcription.

    Args:
        clips_dir: Directory containing clip .mp4 and .json sidecar files.
        overwrite: If True, include clips that already have transcription sidecars.

    Returns:
        list: Tuples of (mp4_path, json_path, trans_path, sidecar_data) for
            clips needing transcription. sidecar_data is the parsed JSON dict.
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
