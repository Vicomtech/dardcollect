import json
import os
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
    """Keep data_link pointing at *target* so the web server can serve the files."""
    if DATA_LINK.is_symlink():
        if DATA_LINK.resolve() == target.resolve():
            return
        DATA_LINK.unlink()
    elif DATA_LINK.exists():
        raise RuntimeError(f"{DATA_LINK} exists and is not a symlink — remove it manually.")
    DATA_LINK.symlink_to(target)
    print(f"Symlink updated: {DATA_LINK} → {target}")


def _scan_dir(dir_path: Path, link_subpath: str) -> list[dict]:
    """Return index entries for all sidecar JSON + video pairs in *dir_path*.

    Paths are relative to the viewer dir, routed through
    data_link/<link_subpath>/...
    """
    items = []
    for json_path in sorted(dir_path.rglob("*.json")):
        name = json_path.name
        if (
            name.endswith("_progress.json")
            or name.endswith(".done")
            or name.endswith(".quality.json")
        ):
            continue
        rel_in_dir = json_path.relative_to(dir_path)
        prefix = f"data_link/{link_subpath}" if link_subpath else "data_link"
        rel_json = f"{prefix}/{rel_in_dir.as_posix()}"
        rel_video = rel_json.rsplit(".", 1)[0] + ".mp4"
        video_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".mp4")
        if not video_real.exists():
            continue
        entry: dict = {"json_path": rel_json, "video_path": rel_video}
        quality_real = dir_path / rel_in_dir.parent / (rel_in_dir.stem + ".quality.json")
        if quality_real.exists():
            entry["quality_path"] = rel_json.rsplit(".", 1)[0] + ".quality.json"
        items.append(entry)
    return items


def index_data() -> None:
    cfg = _load_cfg()

    dir_specs = [
        (cfg.get("person_extraction", {}).get("output_clips_dir"), "extracted_person_clips"),
        (cfg.get("face_crop_extraction", {}).get("output_dir"), "face_crops"),
        (cfg.get("face_quality_filtering", {}).get("output_dir"), "filtered_face_crops"),
    ]

    existing_dirs: dict[str, Path] = {}
    for raw, label in dir_specs:
        if not raw:
            continue
        p = _resolve(raw)
        if p.exists():
            existing_dirs[label] = p

    if not existing_dirs:
        raise RuntimeError("None of the configured output directories exist yet.")

    # Point data_link at the common ancestor so all sub-folders are reachable
    all_paths = list(existing_dirs.values())
    common = Path(os.path.commonpath([str(p) for p in all_paths]))
    _sync_symlink(common)

    print(f"data_link → {common}")
    print(f"Indexing {len(existing_dirs)} folder(s)...")

    folders: dict[str, list[dict]] = {}
    for label, dir_path in existing_dirs.items():
        link_subpath = dir_path.relative_to(common).as_posix()
        items = _scan_dir(dir_path, link_subpath)
        folders[label] = items
        print(f"  {label}: {len(items)} items")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"folders": folders}, f, indent=2)

    total = sum(len(v) for v in folders.values())
    print(f"Index written to {OUTPUT_FILE} — {total} total items across {len(folders)} folder(s).")


if __name__ == "__main__":
    index_data()
