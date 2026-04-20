#!/usr/bin/env python3
"""
Mass download public-domain historical videos (1900-1955) from archive.org
Perfect for face datasets — almost everyone in these videos is deceased.
"""

import re
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

from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage

# ----------------------------------------------------------------------
# CONFIGURATION — loaded from config.yaml
# ----------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

try:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file: {e}")
    sys.exit(1)

SEARCH_QUERY = config.get("search_query", "")
SEARCH_SORT = config.get("search_sort", None)
GLOB_PATTERN = config.get("glob_pattern", "*.mp4")
MIN_DURATION_MINS = config.get("min_duration_minutes", 0)
MAX_WORKERS = config.get("max_workers", 10)
OUTPUT_DIR = Path(config.get("output_dir", "./archive_org_faces_1900_1955"))
RETRY_DELAY = config.get("retry_delay", 5)

MAX_TOTAL_SIZE_GB = config.get("max_total_size_gb", 100)
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_SIZE_GB * 1024 * 1024 * 1024

# Global state for size tracking and cancellation
DOWNLOAD_STATE = {"size": 0}
_cancel = threading.Event()
size_lock = threading.Lock()
history_lock = threading.Lock()
HISTORY_FILE = OUTPUT_DIR / "download_history.txt"
TITLE_HISTORY_FILE = OUTPUT_DIR / "title_history.txt"
title_lock = threading.Lock()


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


def clean_title(title: str) -> str:
    """Normalize title for deduplication."""
    if not title:
        return ""
    # Lowercase first
    t = title.lower()
    # Remove common junk suffixes
    t = re.sub(r"\bstar\b.*", "", t)  # "starring..."
    t = re.sub(r"\bfull movie\b.*", "", t)
    t = re.sub(r"\brestored\b.*", "", t)
    t = re.sub(r"\bcolorized\b.*", "", t)
    # Remove things in brackets/parentheses
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^\)]*\)", "", t)
    # Remove years 1900-2099
    t = re.sub(r"19\d{2}", "", t)
    t = re.sub(r"20\d{2}", "", t)
    # Keeping only alphanumeric
    t = re.sub(r"[^a-z0-9]", "", t)
    return t.strip()


def clean_identifier(identifier: str) -> str:
    """Normalize identifier by removing version suffixes (e.g. _202405)."""
    # Remove trailing date/version stamps like _202312 or _202505
    # Many re-uploads just append _YYYYMM to original ID
    return re.sub(r"_\d{4,8}.*$", "", identifier)


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


def download_item(identifier: str, dest_dir: Path, seen_titles: set):
    """Download the best MP4 file for a single archive.org item.

    Returns:
        tuple: (identifier, success, limit_reached)
            - success: True if file was downloaded, False otherwise
            - limit_reached: True if download stopped due to size limit
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        item = get_item(identifier)

        # --- DEDUPLICATION CHECKS ---
        title = item.metadata.get("title", "")
        if isinstance(title, list):
            title = title[0] if title else ""

        norm_title = clean_title(str(title))
        norm_id = clean_identifier(identifier)

        with title_lock:
            # Check Title
            if norm_title and norm_title in seen_titles:
                print(f"[{identifier}] SKIP: Title collision '{title}' (norm: {norm_title})")
                return identifier, False, False

            # Check Normalized ID
            if norm_id in seen_titles:  # Re-using seen_titles for IDs too to save complex logic?
                # Actually better to have separate set, but for now let's use a
                # "seen_bases" set
                pass

        # Better: use a seen_keys set that stores both titles and base IDs
        # To avoid significant refactoring, let's just use seen_titles for titles
        # And add a local check for IDs if needed.
        # But wait, we need to persist this.

        # Let's simplify: We will treat 'norm_id' as a 'title' for the sake of the
        # history file.
        # It's unique string.

        dedup_keys = []
        dedup_keys = []
        if norm_title:
            dedup_keys.append(norm_title)
        if norm_id:
            dedup_keys.append(norm_id)  # ALWAYS check/claim the base ID

        with title_lock:
            for key in dedup_keys:
                if key in seen_titles:
                    print(f"[{identifier}] SKIP: Duplicate detected via '{key}'")
                    return identifier, False, False

            # RACE CONDITION FIX: Claim valid keys immediately
            for key in dedup_keys:
                seen_titles.add(key)

        files = item.files

        # Find MP4 files and sort by size (largest first = usually highest
        # quality)
        mp4_files = sorted(
            [f for f in files if f["name"].lower().endswith(".mp4")],
            key=lambda f: int(f.get("size", 0)),
            reverse=True,
        )

        if not mp4_files:
            print(f"[{identifier}] No MP4 found, skipping")
            return identifier, False, False

        target_file = mp4_files[0]["name"]

        # Check duration if configured
        if MIN_DURATION_MINS > 0:
            length_str = mp4_files[0].get("length")
            if length_str:
                try:
                    length_sec = float(length_str)
                    if length_sec < (MIN_DURATION_MINS * 60):
                        print(
                            f"[{identifier}] SKIP: Duration {length_sec / 60:.1f}m "
                            f"< {MIN_DURATION_MINS}m"
                        )
                        return identifier, False, False
                except ValueError:
                    pass  # If format is weird, we download anyway to be safe

        target_path = dest_dir / target_file.replace("/", "_")  # flatten
        file_size = int(mp4_files[0].get("size", 0))

        if target_path.exists():
            print(f"[{identifier}] Already downloaded → {target_path.name}")
            # Ensure it's in history so we skip metadata fetch next time
            with history_lock:
                with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{identifier}\n")
            # Record dedup keys for existing file
            with title_lock:
                if norm_title:
                    seen_titles.add(norm_title)
                    with open(TITLE_HISTORY_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{norm_title}\n")
                if norm_id:
                    seen_titles.add(norm_id)
                    with open(TITLE_HISTORY_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{norm_id}\n")
            return identifier, True, False

        # Check for incomplete temp file (from interrupted download)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if temp_path.exists():
            print(f"[{identifier}] Removing incomplete temp file: {temp_path.name}")
            temp_path.unlink()

        # Check size limit
        with size_lock:
            if DOWNLOAD_STATE["size"] + file_size > MAX_TOTAL_SIZE_BYTES:
                print(
                    f"[{identifier}] SKIP: Total size limit reached "
                    f"({DOWNLOAD_STATE['size'] / 1024**3:.2f} GB + "
                    f"{file_size / 1024**3:.2f} GB > {MAX_TOTAL_SIZE_GB} GB)"
                )
                return identifier, False, True
            # Reserve space optimistically
            DOWNLOAD_STATE["size"] += file_size

        # Download to temporary directory, then move to final location
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / target_file.split("/")[-1]
                _download_with_progress(identifier, target_file, temp_file, file_size)

                if not temp_file.exists():
                    raise FileNotFoundError(
                        f"Download failed: {target_file} not found in temp directory"
                    )

                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_file), str(temp_path))

                # Rename from .tmp to final filename (atomic operation)
                temp_path.rename(target_path)
                tqdm.write(f"[{identifier}] Done → {target_path.name}")

        except Exception as e:
            # Clean up incomplete temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            print(f"[{identifier}] Download failed: {e}")
            time.sleep(RETRY_DELAY)
            return identifier, False, False

        # Record success
        with history_lock:
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(f"{identifier}\n")

        # Record success keys
        with title_lock:
            with open(TITLE_HISTORY_FILE, "a", encoding="utf-8") as f:
                if norm_title:
                    f.write(f"{norm_title}\n")
                if norm_id:
                    f.write(f"{norm_id}\n")

        return identifier, True, False

    except Exception as e:
        print(f"[{identifier}] Error: {e}")
        time.sleep(RETRY_DELAY)
        return identifier, False, False


def main():
    """
    Main entry point for the archive.org video downloader.

    Searches for videos matching the configured query and downloads them
    using multithreading.
    """
    started_at = now_iso()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize current size
    print("Calculating current output directory size...")
    DOWNLOAD_STATE["size"] = get_dir_size(OUTPUT_DIR)
    print(
        f"Current size: {DOWNLOAD_STATE['size'] / 1024**3:.2f} GB (Limit: {MAX_TOTAL_SIZE_GB} GB)"
    )

    print(f"Searching archive.org with query:\n  {SEARCH_QUERY}\n")

    # Stream search results (does not load everything into memory)
    sorts = [SEARCH_SORT] if SEARCH_SORT else None
    search = search_items(SEARCH_QUERY, fields=["identifier"], sorts=sorts)

    identifiers = []
    print("Collecting identifiers...")
    for result in search:
        ident = result["identifier"]
        identifiers.append(ident)
        print(f"  → {ident}")

    print(f"\nFound {len(identifiers)} items. Starting download with {MAX_WORKERS} workers...\n")

    # Load history
    completed_ids = set()
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8") as f:
            completed_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(completed_ids)} already downloaded items from history.")

    # Load title history
    seen_titles = set()
    if TITLE_HISTORY_FILE.exists():
        with open(TITLE_HISTORY_FILE, encoding="utf-8") as f:
            seen_titles = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(seen_titles)} unique titles from history.")

    # Filter identifiers
    pending_identifiers = [i for i in identifiers if i not in completed_ids]
    skipped_count = len(identifiers) - len(pending_identifiers)
    print(f"Skipping {skipped_count} items (already in history).")
    identifiers = pending_identifiers

    if not identifiers:
        print("All items already downloaded! Exiting.")
        sys.exit(0)

    # Multithreaded download
    newly_downloaded: list[str] = []
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {
        executor.submit(download_item, ident, OUTPUT_DIR, seen_titles): ident
        for ident in identifiers
    }

    success = 0
    try:
        for future in as_completed(futures):
            ident = futures[future]
            try:
                _, ok, limit_reached = future.result()
                if ok:
                    success += 1
                    newly_downloaded.append(ident)
                if limit_reached:
                    print(
                        f"\n⚠️  Size limit reached ({DOWNLOAD_STATE['size'] / 1024**3:.2f} GB / "
                        f"{MAX_TOTAL_SIZE_GB} GB). Stopping downloads."
                    )
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                print(f"[{ident}] Unhandled exception: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted — stopping downloads...")
        _cancel.set()
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return
    executor.shutdown(wait=True)

    print(f"\nFinished! Successfully downloaded {success}/{len(identifiers)} items.")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")

    record_stage(
        OUTPUT_DIR.parent / PROVENANCE_FILENAME,
        {
            "stage": "download",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {"script": "scripts/download_videos_from_archive.py"},
            "collection": {
                "search_query": SEARCH_QUERY,
                "search_sort": SEARCH_SORT,
                "source": "archive.org",
            },
            "sources": [
                {
                    "identifier": ident,
                    "url": f"https://archive.org/details/{ident}",
                }
                for ident in newly_downloaded
            ],
            "stats": {
                "items_attempted": len(identifiers),
                "items_downloaded": success,
            },
        },
    )


if __name__ == "__main__":
    main()
