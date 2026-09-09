"""Video encoding configuration (issue #8).

Centralises the codec used by both encoder call sites (`_write_video_with_moviepy`
for frame-sequence renders and `extract_clip`, the direct-ffmpeg clip extractor),
plus startup validation that the configured codec actually exists in the resolved
ffmpeg binary — a missing encoder otherwise surfaces as an opaque broken-pipe
error mid-run.

Defaults preserve historical behavior exactly (libx264 / aac / 8 threads), so the
default path needs no ffmpeg validation and no golden drift.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncodingConfig:
    """Video encoding settings read from the config ``encoding:`` section.

    Defaults reproduce the previously hardcoded values.
    """

    video_codec: str = "libx264"
    audio_codec: str = "aac"
    encoder_threads: int = 8

    @classmethod
    def from_yaml_dict(cls, config: dict) -> EncodingConfig:
        """Read the ``encoding:`` section of a loaded YAML config (defaults when absent)."""
        enc = (config or {}).get("encoding", {}) or {}
        return cls(
            video_codec=str(enc.get("video_codec", "libx264")),
            audio_codec=str(enc.get("audio_codec", "aac")),
            encoder_threads=int(enc.get("encoder_threads", 8)),
        )


def validate_video_codec(video_codec: str, ffmpeg_exe: str | None) -> None:
    """Fail loud when the configured codec is not an available ffmpeg encoder.

    Probes ``ffmpeg -encoders`` for the codec name. The bundled imageio-ffmpeg
    build lacks hardware encoders (no NVENC) — point ``FFMPEG_BINARY`` /
    ``IMAGEIO_FFMPEG_EXE`` at a system ffmpeg to use them.
    """
    if video_codec == "libx264":
        return  # default path — always available, no probe needed

    if not ffmpeg_exe:
        raise RuntimeError(
            f"encoding.video_codec is {video_codec!r} but no ffmpeg binary was found "
            "(set FFMPEG_BINARY or IMAGEIO_FFMPEG_EXE to a system ffmpeg that "
            "provides it). The bundled imageio-ffmpeg lacks hardware encoders."
        )

    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not run ffmpeg ({ffmpeg_exe}) to validate encoding.video_codec="
            f"{video_codec!r}: {exc}"
        ) from exc

    encoders = result.stdout or ""
    if video_codec not in encoders:
        raise RuntimeError(
            f"ffmpeg ({ffmpeg_exe}) has no encoder {video_codec!r} — "
            f"check `ffmpeg -encoders`. Note: the bundled imageio-ffmpeg lacks "
            f"NVENC/hardware encoders; set FFMPEG_BINARY or IMAGEIO_FFMPEG_EXE "
            f"to a system ffmpeg build that supports it."
        )
    logger.info("Encoding codec %s validated against %s", video_codec, ffmpeg_exe)
