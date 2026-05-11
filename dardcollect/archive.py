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
import tempfile
import threading
import time
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


def download_item(
    identifier: str,
    dest_dir: Path,
    seen_titles: set,
    history_file: Path,
    min_duration_mins: float = 0,
    media_type: str = "video",
):
    """Download the original file from a single archive.org item.

    For the given identifier, selects the largest suitable file of the requested
    media type, checks duration and size limits, downloads it, and writes FAIR
    metadata to the history CSV.

    Args:
        identifier: archive.org item identifier.
        dest_dir: Directory where the file will be saved.
        seen_titles: Set of already-downloaded titles (legacy, no longer used).
        history_file: Path to the CSV file for recording download metadata.
        min_duration_mins: Minimum duration in minutes (video/audio only).
            Files shorter than this are skipped.
        media_type: One of "video", "audio", "image", or "text".

    Returns:
        dict: Result dictionary with keys:
            - "identifier": The archive.org identifier.
            - "success": True if the file was downloaded or already exists.
            - "limit_reached": True if skipped due to global size limit.
            - "metadata": FAIR metadata dict if successful, None otherwise.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        item = get_item(identifier)

        title = item.metadata.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""

        file_extensions = {
            "video": (".mp4", ".avi", ".mkv", ".mov", ".webm"),
            "audio": (".mp3", ".wav"),
            "image": (".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp", ".webp"),
            "text": (".pdf", ".txt"),
        }
        extensions = file_extensions.get(media_type, ())

        originals = [
            f
            for f in item.files
            if (
                f.get("size")
                and int(f.get("size", 0)) > 100
                and not f.get("private")
                and "__" not in f.get("name", "")
                and not any(
                    x in f.get("name", "").lower() for x in ("thumb", "preview", "derivative")
                )
                and any(f.get("name", "").lower().endswith(ext) for ext in extensions)
            )
        ]
        originals.sort(key=lambda f: int(f.get("size", 0)), reverse=True)
        if originals:
            logger.debug(
                "[%s] %s: Selected %s (%s)",
                identifier,
                media_type,
                originals[0].get("name"),
                originals[0].get("size"),
            )
        if not originals:
            logger.debug(
                "[%s] Skipped: no suitable %s file (%d files checked)",
                identifier,
                media_type,
                len(item.files),
            )
            return {
                "identifier": identifier,
                "success": False,
                "limit_reached": False,
                "metadata": None,
            }

        file_info = originals[0]
        filename = file_info["name"]
        file_size = int(file_info.get("size", 0))

        if min_duration_mins > 0:
            length_str = file_info.get("length")
            if length_str:
                try:
                    if float(length_str) < min_duration_mins * 60:
                        logger.debug(
                            "[%s] SKIP: duration %.1fm < %.0fm",
                            identifier,
                            float(length_str) / 60,
                            min_duration_mins,
                        )
                        return {
                            "identifier": identifier,
                            "success": False,
                            "limit_reached": False,
                            "metadata": None,
                        }
                except ValueError:
                    pass

        # Organize by language for media types that have language metadata
        language = _get_metadata_value(item, "language", "").strip()
        language_aware_types = {"video", "audio", "text"}

        if media_type in language_aware_types:
            lang_subfolder = dest_dir / (language if language else "und")
            lang_subfolder.mkdir(parents=True, exist_ok=True)
            target_path = lang_subfolder / filename.replace("/", "_")
        else:
            target_path = dest_dir / filename.replace("/", "_")

        if target_path.exists():
            logger.debug("[%s] Already exists → %s", identifier, target_path.name)
            metadata = _build_fair_metadata(identifier, item, filename, media_type)
            metadata["download_stage_script"] = "pipeline/download_media_from_archive.py"
            metadata["download_stage_timestamp"] = DOWNLOAD_STARTED_AT
            with csv_lock:
                _write_to_csv(history_file, metadata)
            return {
                "identifier": identifier,
                "success": True,
                "limit_reached": False,
                "metadata": metadata,
            }

        with size_lock:
            if DOWNLOAD_STATE["size"] + file_size > MAX_TOTAL_SIZE_BYTES:
                logger.info(
                    "[%s] SKIP: size limit reached (%.2f GB / %g GB)",
                    identifier,
                    DOWNLOAD_STATE["size"] / 1024**3,
                    MAX_TOTAL_SIZE_GB,
                )
                return {
                    "identifier": identifier,
                    "success": False,
                    "limit_reached": True,
                    "metadata": None,
                }
            DOWNLOAD_STATE["size"] += file_size

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
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            logger.warning("[%s] Download failed: %s", identifier, e)
            time.sleep(RETRY_DELAY)
            return {
                "identifier": identifier,
                "success": False,
                "limit_reached": False,
                "metadata": None,
            }

        metadata = _build_fair_metadata(identifier, item, filename, media_type)
        metadata["download_stage_script"] = "pipeline/download_media_from_archive.py"
        metadata["download_stage_timestamp"] = DOWNLOAD_STARTED_AT
        with csv_lock:
            _write_to_csv(history_file, metadata)

        return {
            "identifier": identifier,
            "success": True,
            "limit_reached": False,
            "metadata": metadata,
        }

    except Exception as e:
        logger.warning("[%s] Error: %s", identifier, e)
        time.sleep(RETRY_DELAY)
        return {
            "identifier": identifier,
            "success": False,
            "limit_reached": False,
            "metadata": None,
        }
