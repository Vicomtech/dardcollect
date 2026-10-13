"""The provenance check must verify each identity space against itself.

Pooling CSV row uuids and sidecar uuids into one set (the behaviour before
2026-10-11) let a link that crossed the two spaces pass as valid. These tests
plant that defect and require it to fail, and require a consistent tree to pass.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gs = _load("golden_snapshot", "scripts/golden_snapshot.py")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sidecar(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _surface(root: Path):
    return gs.Surface(csvs=sorted(root.rglob("*.csv")), sidecars=sorted(root.rglob("*.json")))


def test_unified_identity_passes(tmp_path):
    """CSV row uuid == sidecar uuid, and a crop links to its clip in both spaces."""
    _write_csv(
        tmp_path / "clips_extraction.csv",
        ["uuid", "download_uuid"],
        [{"uuid": "clip-1", "download_uuid": ""}],
    )
    _write_sidecar(tmp_path / "clips" / "clip-1.json", {"uuid": "clip-1"})
    _write_csv(
        tmp_path / "video_face_crops_extraction.csv",
        ["uuid", "parent_uuid"],
        [{"uuid": "crop-1", "parent_uuid": "clip-1"}],
    )
    _write_sidecar(
        tmp_path / "crops" / "crop-1.json",
        {"uuid": "crop-1", "parent_clip": {"uuid": "clip-1"}},
    )
    assert gs._provenance_check(tmp_path, _surface(tmp_path)) == []


def test_csv_link_to_sidecar_only_uuid_is_flagged(tmp_path):
    """A CSV link pointing at a uuid that exists only as a sidecar must fail.

    This is the cross-space defect: the pooled check would accept it because the
    uuid appears somewhere in the tree.
    """
    _write_csv(
        tmp_path / "crops.csv",
        ["uuid", "parent_uuid"],
        [{"uuid": "crop-row", "parent_uuid": "clip-sidecar-only"}],
    )
    _write_sidecar(tmp_path / "clip-sidecar-only.json", {"uuid": "clip-sidecar-only"})
    breaks = gs._provenance_check(tmp_path, _surface(tmp_path))
    assert breaks, "cross-space CSV link was not flagged"
    assert "parent_uuid" in breaks[0]
    assert "CSV row uuids" in breaks[0]


def test_sidecar_parent_to_csv_row_only_uuid_is_flagged(tmp_path):
    """A sidecar parent link resolving only against a CSV row uuid must fail."""
    _write_csv(tmp_path / "clips.csv", ["uuid"], [{"uuid": "clip-row-only"}])
    _write_sidecar(
        tmp_path / "crop.json",
        {"uuid": "crop-1", "parent_clip": {"uuid": "clip-row-only"}},
    )
    breaks = gs._provenance_check(tmp_path, _surface(tmp_path))
    assert breaks, "cross-space sidecar parent link was not flagged"
    assert "parent_clip" in breaks[0]
    assert "sidecar uuids" in breaks[0]
