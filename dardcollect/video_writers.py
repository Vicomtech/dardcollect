"""Video/sidecar writers — clip + frame-sequence encoding helpers.

Extracted from pipeline_utils.py (god-file ratchet) so the encoding-config
plumbing (issue #8) lands in a cohesive, focused module. Contains:

- ``extract_clip`` — direct-ffmpeg clip extractor (input seek, frame-exact)
- ``_write_video_with_moviepy`` — frame-sequence renderer (moviepy)
- ``save_clip_sidecar_json`` — atomic sidecar writer for clips
- ``_cleanup_files`` — partial-file removal helper

Both encoder call sites take an optional :class:`~dardcollect.encoding_config.EncodingConfig`
(issue #8); defaults reproduce the previously hardcoded libx264/aac/8-thread
behaviour, so the default path is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


def _cleanup_files(*paths: Path) -> None:
    """Remove partially-written files so they are not mistaken for valid output."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.info("  Removed incomplete file: %s", path.name)
        except OSError as e:
            logger.warning("  Could not remove %s: %s", path.name, e)


def save_clip_sidecar_json(
    clip_path: Path,
    metadata: dict,
) -> None:
    """Save metadata for a single clip as a sidecar JSON file.

    Writes to a sibling ``.json.partial`` temp file and atomically renames it
    into place, so concurrent downstream readers (audio_clips, face_crops_video)
    that discover clips via ``rglob("*.json")`` never observe a partially-written
    sidecar (Windows file-lock race during clip extraction). The ``.partial``
    suffix ensures ``rglob("*.json")`` does not match the in-progress file.
    """
    sidecar_path = clip_path.with_suffix(".json")
    tmp_path = sidecar_path.with_name(sidecar_path.name + ".partial")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_path, sidecar_path)
    except OSError as e:
        logger.error(
            "Cannot write %s (%s) — removing incomplete file and stopping.",
            sidecar_path.name,
            e,
        )
        _cleanup_files(tmp_path, sidecar_path)
        sys.exit(1)


def _write_video_with_moviepy(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    encoding: EncodingConfig | None = None,
) -> bool:
    """Write frames to MP4 using moviepy (same as extracted_person_clips).

    Args:
        frames: List of BGR numpy arrays (H, W, 3)
        output_path: Output MP4 file path
        fps: Frames per second
        encoding: Encoding settings (issue #8). Defaults reproduce the
            previously hardcoded libx264/aac/8-thread behaviour.

    Returns:
        True if successful, False otherwise
    """
    from dardcollect.encoding_config import EncodingConfig

    enc = encoding or EncodingConfig()
    if not frames:
        logger.error("No frames to write")
        return False

    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Convert BGR to RGB (moviepy uses RGB)
        rgb_frames = [cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in frames]

        # Create a VideoClip from the frames using ImageSequenceClip
        clip = ImageSequenceClip(rgb_frames, durations=[1.0 / fps] * len(rgb_frames))

        clip.write_videofile(
            str(output_path),
            codec=enc.video_codec,
            audio_codec=enc.audio_codec,
            logger=None,
            preset="veryfast",
            threads=enc.encoder_threads,
        )

        success = output_path.exists() and output_path.stat().st_size > 0
        if not success:
            logger.error("Output file is missing or empty")
            return False

        return True

    except Exception as e:
        logger.error("Error writing video with moviepy: %s", e)
        return False


@dataclass
class ClipSpec:
    """One clip to extract: source range + destination + encoding (single argument)."""

    input_path: Path
    output_path: Path
    start_frame: int
    end_frame: int
    fps: float
    encoding: EncodingConfig | None = None


def extract_clip(spec: ClipSpec) -> bool:
    """Extract a clip from a video file with audio.

    Runs the bundled ffmpeg (imageio-ffmpeg, same binary moviepy uses, so no new
    dependency and portable across Linux/Windows/macOS) directly rather than
    through moviepy's Python frame loop. ffmpeg decodes and re-encodes in one
    native process with **input seeking** (``-ss`` before ``-i``), which jumps to
    the source position instead of decoding the whole film up to it — measured
    ~4.6x faster on SD source than moviepy, which decodes every preceding frame
    through Python. Encoding uses ``-preset veryfast`` (same size, ~3x faster
    than the x264 default).

    Frame-exactness is a hard requirement: downstream stages map a clip's frames
    back to source detections by position (``abs_frame = start_frame + frame_id``
    in dardcollect/face_crops.py), so the clip MUST start exactly at
    ``start_frame`` and contain exactly ``end_frame - start_frame + 1`` frames. A
    single off-by-one would misalign the whole ``frame_data`` mapping. ``-ss``
    before ``-i`` is frame-accurate when re-encoding (ffmpeg decodes from the
    preceding keyframe and discards up to the exact timestamp), and ``-frames:v``
    pins the output to exactly the expected frame count regardless of any
    duration rounding.

    Contract note — ``end_frame`` is INCLUSIVE, so the clip length is
    ``end_frame - start_frame + 1`` (== ``Segment.frame_count`` in tracker.py),
    NOT ``end_frame - start_frame``. ``end_frame`` is the last frame the tracker
    saw the person (dardcollect/person_clips.py sets ``end_frame = frame_id``
    while the person stays visible, and the max-duration split uses
    ``start + max_frames - 1``), and ``frame_data`` carries a detection entry for
    that frame. Clip length is therefore never a round number — it follows the
    tracked segment, bounded by ``max_clip_duration_seconds`` (seconds, not a
    frame count). The previous moviepy path produced one frame too few
    (``end - start``), silently dropping the ``end_frame`` detection downstream;
    this path emits the full ``frame_count``.

    Writes to a sibling ``.partial`` temp file and atomically renames it into place on
    success, so concurrent downstream readers (audio_clips, face_crops_video) that scan the
    clips dir via ``rglob("*.mp4")`` never observe a partially-written, moov-less MP4. On
    Windows, a reader opening the in-progress file can lock it and prevent ffmpeg from
    finalizing the moov atom, leaving a corrupt clip (ftyp + mdat, no moov) that blocks the
    pipeline indefinitely; the temp+rename pattern breaks that race.

    The temp uses a ``.partial`` suffix (not ``.tmp.mp4``) specifically so ``rglob("*.mp4")``
    does not match it, and ffmpeg is forced to the mp4 muxer via ``-f mp4`` since the
    extension no longer signals the format. ``os.replace`` also overwrites any stale corrupt
    clip left by a prior interrupted run, self-healing the output dir.

    Args:
        spec: The clip to extract (source range + destination + encoding).
    """
    input_path, output_path = spec.input_path, spec.output_path
    start_frame, end_frame, fps = spec.start_frame, spec.end_frame, spec.fps
    encoding = spec.encoding
    temp_clip = output_path.with_name(output_path.name + ".partial")

    if fps <= 0:
        logger.error("Cannot extract clip %s: invalid fps %s", output_path.name, fps)
        return False

    n_frames = end_frame - start_frame + 1
    if n_frames <= 0:
        logger.error(
            "Cannot extract clip %s: empty frame range [%d, %d]",
            output_path.name,
            start_frame,
            end_frame,
        )
        return False

    start_seconds = start_frame / fps
    try:
        import imageio_ffmpeg

        from dardcollect.encoding_config import EncodingConfig

        enc = encoding or EncodingConfig()
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",  # input seek: fast + frame-accurate on re-encode
            "-i",
            str(input_path),
            "-frames:v",
            str(n_frames),  # pin exact output frame count (alignment contract)
            "-c:v",
            enc.video_codec,
            "-preset",
            "veryfast",
            "-c:a",
            enc.audio_codec,
            "-threads",
            str(enc.encoder_threads),
            "-f",
            "mp4",  # extension is .partial, so name the muxer explicitly
            str(temp_clip),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if not temp_clip.exists() or temp_clip.stat().st_size == 0:
            logger.error(
                "Clip extraction produced empty/missing output for %s",
                output_path.name,
            )
            _cleanup_files(temp_clip)
            return False

        os.replace(temp_clip, output_path)
        return True

    except subprocess.CalledProcessError as e:
        # ffmpeg exited non-zero (unsupported codec, malformed source, write
        # denied…). Per-clip: log stderr, clean up, continue with the next clip.
        # We deliberately do NOT sys.exit: one bad clip must not abort the whole
        # batch (e.g. one VP8 source libx264 can't transcode must not kill the rest).
        logger.error(
            "Cannot extract clip %s: ffmpeg failed (%s) — removing incomplete files.",
            output_path.name,
            (e.stderr or "").strip() or e,
        )
        _cleanup_files(temp_clip)
        return False

    except Exception as e:
        logger.error("Cannot extract clip %s: %s — removing incomplete files.", output_path.name, e)
        _cleanup_files(temp_clip)
        return False
