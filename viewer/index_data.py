import glob
import json
import os
import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DATA_LINK = Path(__file__).resolve().parent / "data_link"
OUTPUT_FILE = Path(__file__).resolve().parent / "data_index.json"


def _load_clips_dir() -> Path:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("person_extraction", {}).get("output_clips_dir")
    if not raw:
        raise ValueError("person_extraction.output_clips_dir is not set in config.yaml")
    return Path(raw)


def _sync_symlink(target: Path) -> None:
    """Keep data_link pointing at *target* so the web server can serve the files."""
    if DATA_LINK.is_symlink() or DATA_LINK.is_dir():
        if DATA_LINK.resolve() == target.resolve():
            return
        subprocess.run(["cmd", "/c", "rmdir", str(DATA_LINK)], check=True)
    elif DATA_LINK.exists():
        raise RuntimeError(f"{DATA_LINK} exists and is not a symlink — remove it manually.")

    subprocess.run(["cmd", "/c", "mklink", "/D", str(DATA_LINK), str(target)], check=True)
    print(f"Junction updated: {DATA_LINK} → {target}")


def index_data() -> None:
    clips_dir = _load_clips_dir()
    if not clips_dir.exists():
        raise FileNotFoundError(f"output_clips_dir does not exist: {clips_dir}")

    _sync_symlink(clips_dir)

    print(f"Scanning {clips_dir} ...")

    json_files = glob.glob(str(DATA_LINK / "**" / "*.json"), recursive=True)
    # Exclude progress sentinels and .done markers
    json_files = [
        f for f in json_files if not f.endswith("_progress.json") and not f.endswith(".done")
    ]

    print(f"Found {len(json_files)} sidecar JSON files.")

    data = []
    for json_path in json_files:
        try:
            rel_json = os.path.relpath(json_path, start=OUTPUT_FILE.parent)
            rel_video = rel_json.rsplit(".", 1)[0] + ".mp4"
            if Path(OUTPUT_FILE.parent / rel_video).exists():
                data.append({"json_path": rel_json, "video_path": rel_video})
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Index written to {OUTPUT_FILE} with {len(data)} items.")


if __name__ == "__main__":
    index_data()
