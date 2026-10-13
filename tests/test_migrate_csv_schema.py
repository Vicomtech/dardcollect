"""CPU-only tests for scripts/migrate_csv_schema.py.

The migration applies renames + drops to bring old traceability CSVs onto the
current (lean event-log) headers, resolving the clips ``download_uuid`` from the
manifest. It must be idempotent so it is safe to re-run.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "migrate_csv_schema.py"
    spec = importlib.util.spec_from_file_location("migrate_csv_schema", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_csv_schema"] = module
    spec.loader.exec_module(module)
    return module


mcs = _load()


def _write(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def test_clips_rename_drops_and_resolves_manifest_uuid(tmp_path):
    clips = tmp_path / "clips" / "clips_extraction.csv"
    _write(
        clips,
        ["uuid", "archive_org_identifier", "source_video", "fps", "duration_seconds"],
        [
            {
                "uuid": "c",
                "archive_org_identifier": "id-a",
                "source_video": "a.mp4",
                "fps": "24",
                "duration_seconds": "2.0",
            }
        ],
    )

    _, changed = mcs._migrate_one(
        clips, mcs.MIGRATIONS["clips_extraction.csv"], {"a.mp4": "dl-1"}, apply=True
    )

    header, out = _read(clips)
    assert changed
    assert header == ["uuid", "download_uuid", "source_video"]
    assert out[0]["download_uuid"] == "dl-1"
    assert out[0]["uuid"] == "c"


def test_drop_only_spec_removes_column(tmp_path):
    frames = tmp_path / "frames_extraction.csv"
    _write(
        frames,
        ["uuid", "clip_uuid", "timestamp_seconds", "output_path"],
        [{"uuid": "f", "clip_uuid": "c", "timestamp_seconds": "1.5", "output_path": "p.png"}],
    )

    _, changed = mcs._migrate_one(frames, mcs.MIGRATIONS["frames_extraction.csv"], {}, apply=True)

    header, out = _read(frames)
    assert changed
    assert header == ["uuid", "clip_uuid", "output_path"]
    assert "timestamp_seconds" not in out[0]


def test_dry_run_does_not_write(tmp_path):
    frames = tmp_path / "frames_extraction.csv"
    _write(frames, ["uuid", "timestamp_seconds"], [{"uuid": "f", "timestamp_seconds": "1.5"}])

    _, changed = mcs._migrate_one(frames, mcs.MIGRATIONS["frames_extraction.csv"], {}, apply=False)

    header, _out = _read(frames)
    assert changed
    assert header == ["uuid", "timestamp_seconds"]  # untouched on a dry run


def test_idempotent_on_current_header(tmp_path):
    frames = tmp_path / "frames_extraction.csv"
    _write(
        frames,
        ["uuid", "clip_uuid", "output_path"],
        [{"uuid": "f", "clip_uuid": "c", "output_path": "p.png"}],
    )

    _, changed = mcs._migrate_one(frames, mcs.MIGRATIONS["frames_extraction.csv"], {}, apply=True)

    header, _out = _read(frames)
    assert not changed
    assert header == ["uuid", "clip_uuid", "output_path"]
