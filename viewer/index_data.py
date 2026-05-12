"""Data indexing utility for viewer web server.

Builds and maintains a data index JSON file that lists all extracted datasets
and allows the web viewer to discover and link to pipeline outputs.

Provides functions for:
    - Loading configuration from YAML.
    - Resolving data paths relative to configuration.
    - Syncing symbolic links or NTFS junctions to data directories.
    - Generating index metadata for web viewer discovery.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DATA_LINK = Path(__file__).resolve().parent / "data_link"
OUTPUT_FILE = Path(__file__).resolve().parent / "data_index.json"


def _load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = CONFIG_PATH.parent / p
    return p.resolve()


def _sync_symlink(target: Path) -> None:
    """Keep data_link pointing at *target* so the web server can serve the files.

    Uses symlinks if permissions allow, falls back to NTFS junctions via PowerShell.
    """
    if DATA_LINK.exists() or DATA_LINK.is_symlink():
        if DATA_LINK.is_symlink() and DATA_LINK.resolve() == target.resolve():
            return

        # Remove existing link/directory - try unlink first (works for symlinks/junctions)
        try:
            DATA_LINK.unlink()
        except OSError:
            # If unlink fails, try rmtree (for regular directories)
            try:
                shutil.rmtree(DATA_LINK)
            except Exception as e:
                raise RuntimeError(f"{DATA_LINK} exists and could not be removed: {e}")

    # Try to create symlink first
    try:
        DATA_LINK.symlink_to(target)
        print(f"Symlink created: {DATA_LINK} → {target}")
        return
    except OSError as e:
        # If symlink fails due to permissions, try junction via PowerShell
        if hasattr(e, "winerror") and e.winerror == 1314:  # ERROR_PRIVILEGE_NOT_HELD
            try:
                ps_cmd = (
                    f'New-Item -ItemType Junction -Path "{DATA_LINK}" '
                    f'-Target "{target}" -Force -ErrorAction Stop'
                )
                subprocess.run(
                    ["powershell", "-Command", ps_cmd], check=True, capture_output=True, text=True
                )
                print(f"Junction created: {DATA_LINK} → {target}")
                return
            except Exception as junction_error:
                raise RuntimeError(
                    f"Failed to create symlink or junction at {DATA_LINK}. "
                    f"Symlink error: {e}. Junction error: {junction_error}. "
                    f"Please run as administrator or enable Developer Mode."
                ) from junction_error
        raise


_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".flac", ".wav", ".m4a", ".aac", ".opus"}


def _scan_video_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for all sidecar JSON + video pairs in *dir_path*."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for json_path in sorted(dir_path.rglob("*.json")):
        name = json_path.name
        if (
            name.endswith("_progress.json")
            or name.endswith(".done")
            or name.endswith(".quality.json")
            or name.endswith(".transcription.json")
        ):
            continue
        rel_in_dir = json_path.relative_to(dir_path)
        rel_json = f"{prefix}/{rel_in_dir.as_posix()}"
        video_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".mp4")
        if not video_real.exists():
            continue
        rel_video = rel_json.rsplit(".", 1)[0] + ".mp4"
        entry: dict = {"json_path": rel_json, "video_path": rel_video}
        quality_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".quality.json")
        if quality_real.exists():
            entry["quality_path"] = rel_json.rsplit(".", 1)[0] + ".quality.json"
        trans_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".transcription.json")
        if trans_real.exists():
            entry["transcription_path"] = rel_json.rsplit(".", 1)[0] + ".transcription.json"
        items.append(entry)
    return items


def _scan_audio_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for audio files, with optional transcription sidecars."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for audio_path in sorted(dir_path.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        rel_in_dir = audio_path.relative_to(dir_path)
        rel_audio = f"{prefix}/{rel_in_dir.as_posix()}"
        entry: dict = {"type": "audio", "audio_path": rel_audio}
        trans_real = audio_path.parent / (audio_path.stem + ".transcription.json")
        if trans_real.exists():
            trans_rel = trans_real.relative_to(dir_path).as_posix()
            entry["transcription_path"] = f"{prefix}/{trans_rel}"
        items.append(entry)
    return items


def _scan_image_detections_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for image detection JSONs (no corresponding video/image files needed)."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for json_path in sorted(dir_path.glob("*.json")):
        name = json_path.name
        if (
            name.endswith("_progress.json")
            or name.endswith(".done")
            or name.endswith(".quality.json")
            or name.endswith(".transcription.json")
        ):
            continue
        rel_in_dir = json_path.relative_to(dir_path)
        rel_json = f"{prefix}/{rel_in_dir.as_posix()}"
        items.append({"type": "image_detection", "json_path": rel_json})
    return items


def _scan_documents_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for annotation + text pairs from document preprocessing."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for ann_path in sorted(dir_path.rglob("*.annotation.json")):
        stem = ann_path.name[: -len(".annotation.json")]
        text_real = ann_path.parent / f"{stem}.text.txt"
        if not text_real.exists():
            continue
        rel_ann = ann_path.relative_to(dir_path).as_posix()
        rel_txt = text_real.relative_to(dir_path).as_posix()
        items.append(
            {
                "type": "document",
                "annotation_path": f"{prefix}/{rel_ann}",
                "text_path": f"{prefix}/{rel_txt}",
            }
        )
    return items


def index_data() -> None:
    cfg = _load_cfg()

    base_output_dir = cfg.get("base_output_dir", "DARD/archive_org_public_domain")

    # (raw_path, label, scan_fn)
    from collections.abc import Callable

    dir_specs: list[tuple[str | None, str, str, Callable]] = [
        (
            cfg.get("person_extraction", {}).get("output_clips_dir"),
            "extracted_person_clips",
            "video",
            _scan_video_dir,
        ),
        (
            cfg.get("image_extraction", {}).get("output_detections_dir"),
            "extracted_image_detections",
            "image",
            _scan_image_detections_dir,
        ),
        (
            cfg.get("face_crop_extraction", {}).get("output_dir"),
            "face_crops",
            "video",
            _scan_video_dir,
        ),
        (
            cfg.get("face_quality_filtering", {}).get("output_dir"),
            "filtered_face_crops",
            "video",
            _scan_video_dir,
        ),
    ]
    # Only index audio files if transcription output directory exists (sidecars written there)
    audio_trans_cfg = cfg.get("audio_transcription", {})
    audio_trans_output = audio_trans_cfg.get("output_dir")
    if audio_trans_output:
        dir_specs.append(
            (
                audio_trans_output,
                "audio_transcriptions",
                "audio",
                _scan_audio_dir,
            )
        )
    docs_raw = cfg.get("document_preprocessing", {}).get("output_dir")
    if docs_raw:
        dir_specs.append((docs_raw, "documents", "document", _scan_documents_dir))

    existing_dirs: dict[str, tuple[Path, str, Callable]] = {}
    for raw, label, dtype, scan_fn in dir_specs:
        if not raw:
            continue
        p = _resolve(raw)
        if p.exists():
            existing_dirs[label] = (p, dtype, scan_fn)

    if not existing_dirs:
        raise RuntimeError("None of the configured output directories exist yet.")

    # Point data_link at the common ancestor so all sub-folders are reachable
    all_paths = [p for p, _, _ in existing_dirs.values()]
    common = Path(os.path.commonpath([str(p) for p in all_paths]))
    _sync_symlink(common)

    print(f"data_link → {common}")
    print(f"Indexing {len(existing_dirs)} folder(s)...")

    folders: dict[str, dict] = {}
    for label, (dir_path, dtype, scan_fn) in existing_dirs.items():
        link_subpath = dir_path.relative_to(common).as_posix()
        items = scan_fn(dir_path, link_subpath)
        folders[label] = {"type": dtype, "items": items}
        print(f"  {label} ({dtype}): {len(items)} items")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"folders": folders}, f, indent=2)

    total = sum(len(v.get("items", [])) for v in folders.values())
    print(f"Index written to {OUTPUT_FILE} — {total} total items across {len(folders)} folder(s).")


if __name__ == "__main__":
    index_data()
