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


def _is_unc_path(path: Path) -> bool:
    """Check if path is a UNC network path (\\\\server\\share or //server/share)."""
    path_str = str(path)
    return path_str.startswith("\\\\") or path_str.startswith("//")


def _sync_symlink(target: Path) -> None:
    """Keep data_link pointing at *target* so the web server can serve the files.

    Uses symlinks if permissions allow, falls back to NTFS junctions via PowerShell.
    Raises RuntimeError on Windows if target is a UNC path (junctions don't support UNC).
    """
    # Windows junctions/symlinks don't work reliably with UNC paths
    if os.name == "nt" and _is_unc_path(target):
        raise RuntimeError(
            f"Cannot create junction to UNC path: {target}\n\n"
            "Windows junctions do not support network paths. Workarounds:\n"
            "  1. Map a network drive: net use Z: \\\\server\\share\n"
            "     Then update config.yaml to use Z:\\... paths\n"
            "  2. Copy data to a local directory\n"
            "  3. Run the viewer from the network location directly:\n"
            f"     cd {target} && python -m http.server 8000\n"
            "     (requires copying viewer/ folder there)"
        )

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
            or name.endswith(".magface.json")
            or name.endswith(".ofiq_attr.json")
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
        magface_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".magface.json")
        if magface_real.exists():
            entry["magface_path"] = rel_json.rsplit(".", 1)[0] + ".magface.json"
        ofiq_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".ofiq_attr.json")
        if ofiq_real.exists():
            entry["ofiq_attr_path"] = rel_json.rsplit(".", 1)[0] + ".ofiq_attr.json"
        trans_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".transcription.json")
        if trans_real.exists():
            entry["transcription_path"] = rel_json.rsplit(".", 1)[0] + ".transcription.json"
        items.append(entry)
    return items


def _scan_audio_transcription_dir(
    dir_path: Path, link_subpath: str, audio_files_dir: Path | None, common: Path
) -> list[dict]:
    """Return index entries for audio transcription JSONs.
    
    Scans for .transcription.json files and reads parent_audio.filename to locate
    the original audio file in audio_files_dir (preserving language subfolders).
    """
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for trans_path in sorted(dir_path.rglob("*.transcription.json")):
        rel_in_dir = trans_path.relative_to(dir_path)
        rel_trans = f"{prefix}/{rel_in_dir.as_posix()}"
        entry: dict = {"type": "audio_transcription", "transcription_path": rel_trans}
        
        # Try to read parent_audio.filename and locate the audio file
        if audio_files_dir:
            try:
                with open(trans_path, encoding="utf-8") as f:
                    trans_data = json.load(f)
                parent_audio = trans_data.get("parent_audio", {})
                audio_filename = parent_audio.get("filename")
                if audio_filename:
                    # Preserve language subfolder structure (e.g., eng/file.mp3)
                    lang_subdir = rel_in_dir.parent
                    audio_path = audio_files_dir / lang_subdir / audio_filename
                    if audio_path.exists():
                        # Construct path relative to common ancestor
                        audio_link = f"data_link/{audio_path.relative_to(common).as_posix()}"
                        entry["audio_path"] = audio_link
            except (json.JSONDecodeError, OSError):
                pass
        
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
    """Return index entries for image detection JSONs (no video/image file required)."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for json_path in sorted(dir_path.glob("*.json")):
        name = json_path.name
        if (
            name.endswith("_progress.json")
            or name.endswith(".done")
            or name.endswith(".magface.json")
            or name.endswith(".ofiq_attr.json")
            or name.endswith(".quality.json")
            or name.endswith(".transcription.json")
        ):
            continue
        rel_in_dir = json_path.relative_to(dir_path)
        rel_json = f"{prefix}/{rel_in_dir.as_posix()}"
        items.append({"type": "image_detection", "json_path": rel_json})
    return items


def _scan_image_face_crops_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for image face crops: 616×616 OFIQ-aligned .jpg + sidecar .json."""
    items = []
    prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
    for json_path in sorted(dir_path.glob("*.json")):
        name = json_path.name
        if (
            name.endswith("_progress.json")
            or name.endswith(".done")
            or name.endswith(".magface.json")
            or name.endswith(".ofiq_attr.json")
            or name.endswith(".transcription.json")
        ):
            continue
        jpg_real = json_path.with_suffix(".jpg")
        if not jpg_real.exists():
            continue
        rel_json = f"{prefix}/{json_path.relative_to(dir_path).as_posix()}"
        rel_jpg = f"{prefix}/{jpg_real.relative_to(dir_path).as_posix()}"
        entry = {"type": "image_face_crop", "json_path": rel_json, "image_path": rel_jpg}
        magface_real = json_path.with_suffix(".magface.json")
        if magface_real.exists():
            entry["magface_path"] = rel_json.rsplit(".", 1)[0] + ".magface.json"
        ofiq_real = json_path.with_suffix(".ofiq_attr.json")
        if ofiq_real.exists():
            entry["ofiq_attr_path"] = rel_json.rsplit(".", 1)[0] + ".ofiq_attr.json"
        items.append(entry)
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
            "video_face_crops",
            "video",
            _scan_video_dir,
        ),
        (
            cfg.get("face_quality_filtering", {}).get("output_dir"),
            "filtered_video_face_crops",
            "video",
            _scan_video_dir,
        ),
        (
            cfg.get("image_face_crop_extraction", {}).get("output_dir"),
            "image_face_crops",
            "image_face_crop",
            _scan_image_face_crops_dir,
        ),
        (
            cfg.get("image_face_quality_filtering", {}).get("output_dir"),
            "filtered_image_face_crops",
            "image_face_crop",
            _scan_image_face_crops_dir,
        ),
    ]
    # Audio transcriptions handled separately (needs audio_files_dir + common path)
    audio_trans_cfg = cfg.get("audio_transcription", {})
    audio_trans_output = audio_trans_cfg.get("output_dir")
    audio_files_dir_raw = audio_trans_cfg.get("audio_files_dir")
    
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

    # Resolve audio transcription paths (handled separately but included in common ancestor)
    audio_trans_dir: Path | None = None
    audio_files_dir: Path | None = None
    if audio_trans_output:
        audio_trans_dir = _resolve(audio_trans_output)
        if not audio_trans_dir.exists():
            audio_trans_dir = None
    if audio_files_dir_raw:
        audio_files_dir = _resolve(audio_files_dir_raw)
        if not audio_files_dir.exists():
            audio_files_dir = None

    if not existing_dirs and not audio_trans_dir:
        raise RuntimeError("None of the configured output directories exist yet.")

    # Point data_link at the common ancestor so all sub-folders are reachable
    all_paths = [p for p, _, _ in existing_dirs.values()]
    if audio_trans_dir:
        all_paths.append(audio_trans_dir)
    if audio_files_dir:
        all_paths.append(audio_files_dir)
    common = Path(os.path.commonpath([str(p) for p in all_paths]))

    # For UNC paths on Windows, skip junction and use serve.py instead
    use_server_proxy = os.name == "nt" and _is_unc_path(common)
    if use_server_proxy:
        print(f"UNC path detected: {common}")
        print("Skipping junction (not supported for network paths on Windows)")
        print("Use 'python viewer/serve.py' to serve files via HTTP proxy")
    else:
        _sync_symlink(common)
        print(f"data_link → {common}")

    total_folders = len(existing_dirs) + (1 if audio_trans_dir else 0)
    print(f"Indexing {total_folders} folder(s)...")

    folders: dict[str, dict] = {}
    for label, (dir_path, dtype, scan_fn) in existing_dirs.items():
        link_subpath = dir_path.relative_to(common).as_posix()
        items = scan_fn(dir_path, link_subpath)
        folders[label] = {"type": dtype, "items": items}
        print(f"  {label} ({dtype}): {len(items)} items")

    # Handle audio transcriptions with special function (needs audio_files_dir + common)
    if audio_trans_dir:
        link_subpath = audio_trans_dir.relative_to(common).as_posix()
        items = _scan_audio_transcription_dir(
            audio_trans_dir, link_subpath, audio_files_dir, common
        )
        folders["audio_transcriptions"] = {"type": "audio_transcription", "items": items}
        print(f"  audio_transcriptions (audio_transcription): {len(items)} items")

    # Store data_root for serve.py to use when proxying requests
    index_data_out = {
        "folders": folders,
        "data_root": str(common),
        "use_server_proxy": use_server_proxy,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data_out, f, indent=2)

    total = sum(len(v.get("items", [])) for v in folders.values())
    print(f"Index written to {OUTPUT_FILE} — {total} total items across {len(folders)} folder(s).")


if __name__ == "__main__":
    index_data()
