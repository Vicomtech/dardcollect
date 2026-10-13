"""CPU-only tests for scripts/golden_snapshot._volume_check.

The volume bound must be robust to input-driven stage presence (a downstream
stage that produces nothing when its input is empty and hundreds of per-frame
sidecars when it runs) while still failing on a real artifact collapse.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "golden_snapshot.py"
    spec = importlib.util.spec_from_file_location("golden_snapshot", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["golden_snapshot"] = module
    spec.loader.exec_module(module)
    return module


gs = _load()


def _manifest(csvs, sidecar_keys):
    return {"csv": dict.fromkeys(csvs, "h"), "sidecars": dict.fromkeys(sidecar_keys, "h")}


def test_extra_csv_is_not_a_volume_failure():
    """A stage that found input this run adds its CSV — data variation, not a regression."""
    baseline = _manifest(["a.csv"], ["video_face_crops/x.json"])
    current = _manifest(["a.csv", "frames_extraction.csv"], ["video_face_crops/x.json"])
    assert gs._volume_check(baseline, current) == []


def test_per_frame_sidecars_are_excluded_from_the_volume_bound():
    baseline = _manifest(["a.csv"], ["video_face_crops/x.json"])
    current = _manifest(
        ["a.csv", "frames_extraction.csv"],
        ["video_face_crops/x.json"]
        + [f"extracted_frames/c/frame_{i:06d}.json" for i in range(500)],
    )
    assert gs._volume_check(baseline, current) == []


def test_artifact_collapse_still_fails():
    baseline = _manifest(["a.csv"], [f"video_face_crops/x{i}.json" for i in range(10)])
    current = _manifest(["a.csv"], ["video_face_crops/x0.json"])
    issues = gs._volume_check(baseline, current)
    assert issues and "out of bounds" in issues[0]
