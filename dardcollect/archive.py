"""
Archive.org download primitives.

Provides functions for downloading files from archive.org items,
building FAIR-compliant metadata, and recording downloads in CSV.

Shared state (DOWNLOAD_STATE, size_lock, csv_lock, _cancel) is
initialized by the calling script (download_media_from_archive.py)
after the module is imported.
"""

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from internetarchive import get_item
from tqdm import tqdm

from dardcollect.extraction_logger import _write_to_csv
from dardcollect.fair import _build_fair_metadata, _get_metadata_value

logger = logging.getLogger(__name__)

# ── Shared state — set by the calling script ──────────────────────────────────

DOWNLOAD_STATE: dict = {"size": 0}
MAX_TOTAL_SIZE_BYTES: int = 100 * 1024 * 1024 * 1024
MAX_TOTAL_SIZE_GB: float = 100.0
DOWNLOAD_STARTED_AT: str = ""
RETRY_DELAY: float = 5.0

size_lock = threading.Lock()
csv_lock = threading.Lock()
_cancel = threading.Event()


# ── Download primitives ───────────────────────────────────────────────────────


# ── Codec probing + AV1 policy (issue #10) ────────────────────────────────────


def _ffmpeg_exe() -> str | None:
    """Resolve an ffmpeg binary: env override first, then imageio-ffmpeg's.

    Returns None when no ffmpeg is available (probe then reports "unknown").
    """
    import os

    for env_var in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        exe = os.environ.get(env_var)
        if exe and Path(exe).exists():
            return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_video_codec(video_path: Path) -> str | None:
    """Return the codec name of a video's first video stream, or None on failure.

    Uses ffprobe when a sibling of the resolved ffmpeg binary exists; otherwise
    falls back to parsing ``ffmpeg -i`` stderr (logged per the runtime-fallback
    policy — the bundled imageio-ffmpeg ships without ffprobe).
    """
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        logger.warning("No ffmpeg available — cannot probe codec of %s", video_path.name)
        return None

    ffprobe = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
    if ffprobe.exists():
        try:
            result = subprocess.run(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            codec = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else None
            return codec or None
        except Exception as exc:
            logger.warning(
                "ffprobe failed for %s (%s) — falling back to ffmpeg parse", video_path.name, exc
            )

    # Fallback (logged): parse `ffmpeg -i` stderr, e.g. "Stream #0:0... Video: av1 ..."
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(video_path), "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        stderr = result.stderr or ""
        for line in stderr.splitlines():
            if "Video:" in line:
                after = line.split("Video:", 1)[1].strip()
                # ffmpeg prints "Video: av1 (Main), yuv420p" — the codec token
                # ends at the first space or comma.
                codec = after.split()[0].split(",")[0].strip() if after else ""
                logger.info(
                    "Codec probe via ffmpeg-parse fallback (no ffprobe): %s → %s",
                    video_path.name,
                    codec,
                )
                return codec or None
        return None
    except Exception as exc:
        logger.warning("ffmpeg parse failed for %s: %s", video_path.name, exc)
        return None


def _apply_av1_policy(video_path: Path, policy: str) -> bool:
    """Probe a downloaded video's codec and apply the configured AV1 policy.

    Args:
        video_path: Path to the downloaded video file.
        policy: "warn" (default — log a loud warning, keep the file) or
            "skip" (delete the file and report the item as not downloaded).

    Returns:
        True when the file should be kept (or policy is unknown), False when the
        file was removed by the skip policy.
    """
    if policy not in ("warn", "skip"):
        logger.warning("Unknown av1_policy %r — falling back to 'warn'", policy)
        policy = "warn"

    codec = probe_video_codec(video_path)
    if codec is None:
        logger.warning(
            "Codec probe failed — proceeding without AV1 handling "
            "(downstream OpenCV stages may read 0 frames if this is AV1)"
        )
        return True

    if codec.strip().lower() != "av1":
        logger.debug("Codec %s — no AV1 handling needed", codec)
        return True

    if policy == "skip":
        try:
            video_path.unlink()
            logger.warning(
                "AV1 codec detected and av1_policy=skip — deleted %s (downstream "
                "stages cannot decode AV1 on this build; transcode is a planned "
                "follow-up, see issue #10)",
                video_path.name,
            )
            return False
        except OSError as exc:
            logger.error("AV1 skip policy could not delete file: %s", exc)
            return True

    logger.warning(
        "AV1 codec detected — OpenCV-based stages (person detection, face crops, "
        "frame extraction) will read 0 frames unless this build supports AV1 "
        "decode (install a system ffmpeg with libdav1d and point "
        "FFMPEG_BINARY/IMAGEIO_FFMPEG_EXE at it, or set av1_policy: skip)"
    )
    return True


def _download_with_progress(
    identifier: str, filename: str, dest_path: Path, file_size: int
) -> None:
    """Download a single file from archive.org with a tqdm progress bar.

    Args:
        identifier: archive.org item identifier.
        filename: Name of the file to download within the item.
        dest_path: Local path where the file will be saved.
        file_size: Expected file size in bytes (for progress bar total).

    Raises:
        InterruptedError: If the global cancellation event is set during download.
        requests.HTTPError: If the HTTP request fails.
    """
    url = f"https://archive.org/download/{identifier}/{filename}"
    label = f"[{identifier[:25]}] {Path(filename).name[:30]}"
    # connect timeout 30s, stall timeout 60s (no bytes received)
    with requests.get(url, stream=True, timeout=(30, 60)) as r:
        r.raise_for_status()
        with tqdm(
            total=file_size or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            leave=True,
        ) as bar:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if _cancel.is_set():
                        raise InterruptedError("Cancelled")
                    f.write(chunk)
                    bar.update(len(chunk))


_FILE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "video": (".mp4", ".avi", ".mkv", ".mov", ".webm"),
    "audio": (".mp3", ".wav"),
    "image": (".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp", ".webp"),
    "text": (".pdf", ".txt"),
}
# Substrings that mark archive.org derivatives, not the original upload.
_DERIVATIVE_MARKERS = ("thumb", "preview", "derivative")
# Media types organized into per-language subfolders.
_LANGUAGE_AWARE_TYPES = frozenset({"video", "audio", "text"})


def _download_result(
    identifier: str,
    *,
    success: bool,
    limit_reached: bool = False,
    metadata: dict | None = None,
) -> dict:
    """Uniform result dict so every return path reports the same keys."""
    return {
        "identifier": identifier,
        "success": success,
        "limit_reached": limit_reached,
        "metadata": metadata,
    }


def _is_original_file(entry: dict, extensions: tuple[str, ...]) -> bool:
    """True if *entry* is a real original upload of one of *extensions*."""
    name = entry.get("name", "")
    lower = name.lower()
    return (
        bool(entry.get("size"))
        and int(entry.get("size", 0)) > 100
        and not entry.get("private")
        and "__" not in name
        and not any(marker in lower for marker in _DERIVATIVE_MARKERS)
        and any(lower.endswith(ext) for ext in extensions)
    )


def _select_original(item, media_type: str) -> dict | None:
    """Largest suitable original file of *media_type*, or None if none exists."""
    extensions = _FILE_EXTENSIONS.get(media_type, ())
    originals = [f for f in item.files if _is_original_file(f, extensions)]
    if not originals:
        return None
    originals.sort(key=lambda f: int(f.get("size", 0)), reverse=True)
    return originals[0]


def _too_short(file_info: dict, min_duration_mins: float) -> bool:
    """True if the file's reported duration is below *min_duration_mins*."""
    if min_duration_mins <= 0:
        return False
    length_str = file_info.get("length")
    if not length_str:
        return False
    try:
        return float(length_str) < min_duration_mins * 60
    except ValueError:
        return False


def _target_path(dest_dir: Path, filename: str, language: str, media_type: str) -> Path:
    """Destination path, in a per-language subfolder for language-aware types."""
    safe_name = filename.replace("/", "_")
    if media_type in _LANGUAGE_AWARE_TYPES:
        lang_subfolder = dest_dir / (language or "und")
        lang_subfolder.mkdir(parents=True, exist_ok=True)
        return lang_subfolder / safe_name
    return dest_dir / safe_name


def _stamp_download_metadata(metadata: dict) -> dict:
    """Add the download-stage provenance fields shared by both write paths."""
    metadata["download_stage_script"] = "pipeline/download_media_from_archive.py"
    metadata["download_stage_timestamp"] = DOWNLOAD_STARTED_AT
    return metadata


@dataclass
class DownloadRequest:
    """One archive.org item to download (single argument to download_item)."""

    identifier: str
    dest_dir: Path
    history_file: Path
    min_duration_mins: float = 0
    media_type: str = "video"
    av1_policy: str = "warn"


def download_item(req: DownloadRequest):
    """Download the original file from a single archive.org item.

    For the given identifier, selects the largest suitable file of the requested
    media type, checks duration and size limits, downloads it, and writes FAIR
    metadata to the history CSV.

    Args:
        req: The item + destination + policy (see DownloadRequest).

    Returns:
        dict: Result dictionary with keys:
            - "identifier": The archive.org identifier.
            - "success": True if the file was downloaded or already exists.
            - "limit_reached": True if skipped due to global size limit.
            - "metadata": FAIR metadata dict if successful, None otherwise.
    """
    identifier, dest_dir, history_file = req.identifier, req.dest_dir, req.history_file
    min_duration_mins, media_type = req.min_duration_mins, req.media_type
    av1_policy = req.av1_policy
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        item = get_item(identifier)

        file_info = _select_original(item, media_type)
        if file_info is None:
            logger.debug(
                "[%s] Skipped: no suitable %s file (%d files checked)",
                identifier,
                media_type,
                len(item.files),
            )
            return _download_result(identifier, success=False)

        filename = file_info["name"]
        file_size = int(file_info.get("size", 0))

        if _too_short(file_info, min_duration_mins):
            logger.debug(
                "[%s] SKIP: duration %.1fm < %.0fm",
                identifier,
                float(file_info["length"]) / 60,
                min_duration_mins,
            )
            return _download_result(identifier, success=False)

        language = _get_metadata_value(item, "language", "").strip()
        target_path = _target_path(dest_dir, filename, language, media_type)

        if target_path.exists():
            logger.debug("[%s] Already exists → %s", identifier, target_path.name)
            metadata = _stamp_download_metadata(
                _build_fair_metadata(identifier, item, filename, media_type)
            )
            with csv_lock:
                _write_to_csv(history_file, metadata)
            return _download_result(identifier, success=True, metadata=metadata)

        with size_lock:
            if DOWNLOAD_STATE["size"] + file_size > MAX_TOTAL_SIZE_BYTES:
                logger.info(
                    "[%s] SKIP: size limit reached (%.2f GB / %g GB)",
                    identifier,
                    DOWNLOAD_STATE["size"] / 1024**3,
                    MAX_TOTAL_SIZE_GB,
                )
                return _download_result(identifier, success=False, limit_reached=True)
            DOWNLOAD_STATE["size"] += file_size

        if not _download_to(identifier, filename, file_size, target_path):
            return _download_result(identifier, success=False)

        metadata = _stamp_download_metadata(
            _build_fair_metadata(identifier, item, filename, media_type)
        )
        with csv_lock:
            _write_to_csv(history_file, metadata)

        # AV1 handling (issue #10): probe the codec and apply the configured
        # policy. warn = loud log, keep file (default). skip = delete + report.
        if media_type == "video" and not _apply_av1_policy(target_path, av1_policy):
            metadata["download_skipped_reason"] = "av1_policy_skip"
            with csv_lock:
                _write_to_csv(history_file, metadata)
            return _download_result(identifier, success=False)

        return _download_result(identifier, success=True, metadata=metadata)

    except Exception as e:
        logger.warning("[%s] Error: %s", identifier, e)
        time.sleep(RETRY_DELAY)
        return _download_result(identifier, success=False)


def _download_to(identifier: str, filename: str, file_size: int, target_path: Path) -> bool:
    """Download to a temp file then atomically rename into *target_path*.

    Returns True on success; on failure logs, cleans the partial file, backs off
    and returns False (no partial file is left behind).
    """
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / filename.split("/")[-1]
            _download_with_progress(identifier, filename, temp_file, file_size)
            if not temp_file.exists():
                raise FileNotFoundError(f"Missing after download: {filename}")
            shutil.move(str(temp_file), str(temp_path))
            temp_path.rename(target_path)
            logger.info("[%s] Done → %s", identifier, target_path.name)
        return True
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        logger.warning("[%s] Download failed: %s", identifier, e)
        time.sleep(RETRY_DELAY)
        return False
