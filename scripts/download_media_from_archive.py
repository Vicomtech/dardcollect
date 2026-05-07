#!/usr/bin/env python3
"""
Mass download public-domain historical media files from archive.org

Downloads videos, images, audio, and texts into language-based subfolders.
All downloads are recorded in a single unified CSV: dataset.csv
Resumable: skips files already in dataset.csv (filtered by media_type).
"""

import csv
import logging
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from internetarchive import get_item, search_items
from tqdm import tqdm

from persondet.config import get_log_level
from persondet.fair import add_fair_metadata, generate_uuid
from persondet.provenance import now_iso


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIGURATION — loaded from config.yaml
# ----------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

try:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("Configuration file not found at %s", CONFIG_PATH)
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error("Error parsing configuration file: %s", e)
    sys.exit(1)

logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))

SEARCH_SORT = config.get("search_sort", None)
MAX_WORKERS = config.get("max_workers", 10)
MAX_TOTAL_SIZE_GB = config.get("max_total_size_gb", 100)
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_SIZE_GB * 1024 * 1024 * 1024
RETRY_DELAY = config.get("retry_delay", 5)
BASE_OUTPUT_DIR = Path(config.get("base_output_dir", "./archive_org_public_domain"))
MEDIA_DOWNLOAD_CONFIG = config.get("media_download", {})
ACTIVE_TYPES = set(config.get("media_types", ["video"]))

# Global state shared across all media types
DOWNLOAD_STATE = {"size": 0}
DOWNLOAD_STARTED_AT = now_iso()  # Timestamp for all downloads in this run
_cancel = threading.Event()
size_lock = threading.Lock()
csv_lock = threading.Lock()  # Unified CSV write lock


# ----------------------------------------------------------------------
def get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _download_with_progress(
    identifier: str, filename: str, dest_path: Path, file_size: int
) -> None:
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


def _build_fair_metadata(identifier: str, item, filename: str, media_type: str) -> dict:
    """Build FAIR metadata dict for a downloaded item."""
    metadata = {
        "uuid": generate_uuid(),
        "archive_org_identifier": identifier,
        "archive_org_url": f"https://archive.org/details/{identifier}",
        "filename_downloaded": filename,
        "media_type": media_type,
        "title": _get_metadata_value(item, "title", ""),
        "creator": _get_metadata_value(item, "creator", ""),
        "date": _get_metadata_value(item, "date", ""),
        "year": _get_metadata_value(item, "year", ""),
        "description": _get_metadata_value(item, "description", "")[:500],
        "licenseurl": _get_metadata_value(item, "licenseurl", ""),
        "subject": _get_metadata_value(item, "subject", ""),
        "collection": _get_metadata_value(item, "collection", ""),
        "language": _get_metadata_value(item, "language", ""),
        "downloaded_at": now_iso(),
    }

    # Add FAIR metadata with archive.org source tracking
    return add_fair_metadata(
        metadata,
        schema_type="download",
        archive_org_id=identifier,
        archive_org_url=f"https://archive.org/details/{identifier}",
    )


def _get_metadata_value(item, key: str, default="") -> str:
    """Safely extract metadata value, handling lists and None."""
    val = item.metadata.get(key, default)
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v)
    return str(val) if val else default


def _write_to_csv(csv_path: Path, metadata: dict) -> None:
    """Write metadata row to unified dataset CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "uuid",
        "archive_org_identifier",
        "archive_org_url",
        "filename_downloaded",
        "media_type",
        "title",
        "creator",
        "date",
        "year",
        "description",
        "licenseurl",
        "subject",
        "collection",
        "language",
        "downloaded_at",
        "source",
        "download_stage_script",
        "download_stage_timestamp",
    ]

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        # Flatten "source" dict if it exists
        row = metadata.copy()
        if "source" in row and isinstance(row["source"], dict):
            row["source"] = str(row["source"])

        writer.writerow(row)


def download_item(
    identifier: str,
    dest_dir: Path,
    seen_titles: set,
    history_file: Path,
    min_duration_mins: float = 0,
    media_type: str = "video",
):
    """Download the original file from an archive.org item.

    Returns:
        dict: {
            'identifier': str,
            'success': bool,
            'limit_reached': bool,
            'metadata': dict (FAIR metadata if successful, None otherwise)
        }
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

        if language and media_type in language_aware_types:
            # Create language subfolder (e.g., "eng", "spa", "fra")
            lang_subfolder = dest_dir / language
            lang_subfolder.mkdir(parents=True, exist_ok=True)
            target_path = lang_subfolder / filename.replace("/", "_")
        else:
            target_path = dest_dir / filename.replace("/", "_")

        if target_path.exists():
            logger.debug("[%s] Already exists → %s", identifier, target_path.name)
            metadata = _build_fair_metadata(identifier, item, filename, media_type)
            metadata["download_stage_script"] = "scripts/download_media_from_archive.py"
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
        metadata["download_stage_script"] = "scripts/download_media_from_archive.py"
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
        return identifier, False, False


def main():
    """Search all active media types then download all items interleaved."""
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DOWNLOAD_STATE["size"] = get_dir_size(BASE_OUTPUT_DIR)
    logger.info(
        "Base output dir: %.2f GB used / %g GB limit",
        DOWNLOAD_STATE["size"] / 1024**3,
        MAX_TOTAL_SIZE_GB,
    )

    sorts = [SEARCH_SORT] if SEARCH_SORT else None

    # Phase 1: search all active types and collect pending tasks
    Task = tuple  # (ident, media_type, output_dir, seen_titles, dataset_csv, min_dur)
    tasks: list[Task] = []

    for media_type, type_cfg in MEDIA_DOWNLOAD_CONFIG.items():
        if media_type not in ACTIVE_TYPES:
            logger.info("Skipping %s (not in media_types)", media_type)
            continue

        search_query = type_cfg.get("search_query", "").strip()
        if not search_query:
            logger.warning("No search_query for '%s', skipping", media_type)
            continue

        output_dir = BASE_OUTPUT_DIR / type_cfg.get("output_subdir", media_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        min_duration_mins = type_cfg.get("min_duration_minutes", 0)
        max_results = type_cfg.get("max_results", 1000)  # Limit search scope

        logger.info("=== %s: searching archive.org... ===", media_type.upper())
        try:
            search = search_items(search_query, fields=["identifier"], sorts=sorts)
            identifiers = []
            for i, r in enumerate(search):
                if i >= max_results:
                    logger.info("%s: reached max_results limit (%d)", media_type, max_results)
                    break
                if "identifier" in r:
                    identifiers.append(r["identifier"])
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            logger.error(
                "%s: search timed out after %ds (query too complex?). Skipping this type.",
                media_type.upper(),
                30,
            )
            logger.debug("Search query: %s", search_query)
            continue
        except Exception as e:
            logger.error("%s: search failed: %s. Skipping this type.", media_type.upper(), e)
            continue

        completed_ids: set[str] = set()
        dataset_csv = BASE_OUTPUT_DIR / "dataset.csv"
        if dataset_csv.exists():
            with open(dataset_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                completed_ids = {
                    row["archive_org_identifier"]
                    for row in reader
                    if row and row.get("media_type") == media_type
                }

        seen_titles: set[str] = set()  # No longer used, but kept for compatibility

        pending = [i for i in identifiers if i not in completed_ids]
        logger.info(
            "%s: %d new, %d already downloaded",
            media_type,
            len(pending),
            len(identifiers) - len(pending),
        )
        for ident in pending:
            tasks.append(
                (
                    ident,
                    media_type,
                    output_dir,
                    seen_titles,
                    dataset_csv,
                    min_duration_mins,
                )
            )

    if not tasks:
        logger.info("Nothing to download.")
        sys.exit(0)

    logger.info("Starting download of %d items across %s types", len(tasks), len(ACTIVE_TYPES))

    # Phase 2: download all types interleaved in one shared executor
    all_newly_downloaded: list[dict] = []
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures: dict = {}

    # Organize tasks by media type for round-robin submission to ensure balanced load
    tasks_by_type: dict = {}
    for ident, media_type, output_dir, seen_titles, dataset_csv, min_dur in tasks:
        if media_type not in tasks_by_type:
            tasks_by_type[media_type] = []
        tasks_by_type[media_type].append(
            (ident, media_type, output_dir, seen_titles, dataset_csv, min_dur)
        )

    # Submit tasks in round-robin fashion by media type for balanced concurrent downloads
    max_tasks_per_type = max(len(t) for t in tasks_by_type.values()) if tasks_by_type else 0
    for task_idx in range(max_tasks_per_type):
        for media_type in ACTIVE_TYPES:
            if media_type not in tasks_by_type or task_idx >= len(tasks_by_type[media_type]):
                continue
            ident, media_type, output_dir, seen_titles, dataset_csv, min_dur = tasks_by_type[
                media_type
            ][task_idx]
            fut = executor.submit(
                download_item,
                ident,
                output_dir,
                seen_titles,
                dataset_csv,
                min_dur,
                media_type,
            )
            futures[fut] = (ident, media_type)

    success = 0
    success_by_type: dict = {}
    try:
        for future in as_completed(futures):
            ident, media_type = futures[future]
            try:
                result = future.result()
                ok = result["success"]
                limit_reached = result["limit_reached"]
                success_by_type[media_type] = success_by_type.get(media_type, 0) + (1 if ok else 0)
                if ok:
                    success += 1
                    all_newly_downloaded.append(
                        {
                            "type": media_type,
                            "identifier": ident,
                            "metadata": result["metadata"],
                        }
                    )
                if limit_reached:
                    logger.warning(
                        "Size limit reached (%.2f GB / %g GB) — "
                        "letting in-flight downloads finish.",
                        DOWNLOAD_STATE["size"] / 1024**3,
                        MAX_TOTAL_SIZE_GB,
                    )
                    for fut in futures:
                        fut.cancel()  # no-op for already-running futures
                    break
            except Exception as e:
                logger.error("[%s] Unhandled exception: %s", ident, e)
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping downloads.")
        _cancel.set()
        for fut in futures:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return
    executor.shutdown(wait=True)

    logger.info("Done — %d/%d items downloaded", success, len(tasks))
    logger.info(
        "By media type: %s",
        " | ".join(f"{t}={success_by_type.get(t, 0)}" for t in ACTIVE_TYPES),
    )
    logger.info("Dataset CSV: %s", BASE_OUTPUT_DIR / "dataset.csv")


if __name__ == "__main__":
    main()
