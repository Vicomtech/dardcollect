#!/usr/bin/env python3
"""
Mass download public-domain historical media files from archive.org

Downloads videos, images, audio, and texts into language-based subfolders.
All downloads are recorded in a single unified CSV: downloads.csv
Resumable: skips files already in downloads.csv (filtered by media_type).
"""

import csv
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml
from internetarchive import search_items

import dardcollect.archive as _archive
from dardcollect.archive import download_item
from dardcollect.config import get_log_level
from dardcollect.pipeline_timer import add_timer
from dardcollect.pipeline_utils import _TqdmHandler, get_dir_size
from dardcollect.provenance import now_iso

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIGURATION — loaded from config.yaml
# ----------------------------------------------------------------------
CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)

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

# Global state shared across all media types — wire into dardcollect.archive
DOWNLOAD_STATE = {"size": 0}
DOWNLOAD_STARTED_AT = now_iso()  # Timestamp for all downloads in this run
_cancel = threading.Event()
size_lock = threading.Lock()
csv_lock = threading.Lock()  # Unified CSV write lock

# Push shared state into archive module so the moved functions see it
_archive.DOWNLOAD_STATE = DOWNLOAD_STATE
_archive.MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_SIZE_BYTES
_archive.MAX_TOTAL_SIZE_GB = MAX_TOTAL_SIZE_GB
_archive.DOWNLOAD_STARTED_AT = DOWNLOAD_STARTED_AT
_archive.RETRY_DELAY = RETRY_DELAY
_archive.size_lock = size_lock
_archive.csv_lock = csv_lock
_archive._cancel = _cancel


def _collect_download_tasks(media_type: str, type_cfg: dict, sorts: list | None) -> list:
    """Search archive.org for one media type and return its pending download
    tasks (skipping already-downloaded identifiers). Returns [] if the type is
    inactive, has no search_query, or the search fails/times out."""
    if media_type not in ACTIVE_TYPES:
        logger.info("Skipping %s (not in media_types)", media_type)
        return []
    search_query = type_cfg.get("search_query", "").strip()
    if not search_query:
        logger.warning("No search_query for '%s', skipping", media_type)
        return []

    output_dir = BASE_OUTPUT_DIR / type_cfg.get("output_subdir", media_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_duration_mins = type_cfg.get("min_duration_minutes", 0)
    max_results = type_cfg.get("max_results", 1000)

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
        return []
    except Exception as e:
        logger.error("%s: search failed: %s. Skipping this type.", media_type.upper(), e)
        return []

    completed_ids: set[str] = set()
    downloads_csv = BASE_OUTPUT_DIR / "downloads.csv"
    if downloads_csv.exists():
        with open(downloads_csv, encoding="utf-8") as f:
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
    return [
        (i, media_type, output_dir, seen_titles, downloads_csv, min_duration_mins) for i in pending
    ]


@add_timer
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
    Task = tuple  # (ident, media_type, output_dir, seen_titles, downloads_csv, min_dur)
    tasks: list[Task] = []
    for media_type, type_cfg in MEDIA_DOWNLOAD_CONFIG.items():
        tasks.extend(_collect_download_tasks(media_type, type_cfg, sorts))

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
    for ident, media_type, output_dir, seen_titles, downloads_csv, min_dur in tasks:
        if media_type not in tasks_by_type:
            tasks_by_type[media_type] = []
        tasks_by_type[media_type].append(
            (ident, media_type, output_dir, seen_titles, downloads_csv, min_dur)
        )

    # Submit tasks in round-robin fashion by media type for balanced concurrent downloads
    max_tasks_per_type = max(len(t) for t in tasks_by_type.values()) if tasks_by_type else 0
    for task_idx in range(max_tasks_per_type):
        for media_type in ACTIVE_TYPES:
            if media_type not in tasks_by_type or task_idx >= len(tasks_by_type[media_type]):
                continue
            ident, media_type, output_dir, seen_titles, downloads_csv, min_dur = tasks_by_type[
                media_type
            ][task_idx]
            fut = executor.submit(
                download_item,
                ident,
                output_dir,
                seen_titles,
                downloads_csv,
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
    logger.info("Downloads CSV: %s", BASE_OUTPUT_DIR / "downloads.csv")


if __name__ == "__main__":
    main()
