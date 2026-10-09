"""The quality filter moves a passing crop together with its OFIQ annotation.

Regression: the move trio (crop, .json, .magface.json) left
``<crop>.ofiq_attr.json`` behind in input_dir, orphaning the OFIQ annotation of every
crop that passed the filter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent


def _filter_module():
    # The stage script imports its sibling helper the way it does when run directly.
    sys.path.insert(0, str(REPO / "pipeline"))
    spec = importlib.util.spec_from_file_location(
        "filter_face_crops_by_quality", REPO / "pipeline" / "filter_face_crops_by_quality.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("passes", [True, False])
def test_passing_crop_takes_its_ofiq_sidecar(tmp_path, monkeypatch, passes):
    mod = _filter_module()
    input_dir = tmp_path / "video_face_crops" / "Actor_01"
    output_dir = tmp_path / "filtered" / "Actor_01"
    input_dir.mkdir(parents=True)
    crop = input_dir / "clip_face_1.mp4"
    crop.write_bytes(b"mp4")
    crop.with_suffix(".json").write_text("{}", encoding="utf-8")
    crop.with_suffix(".magface.json").write_text("{}", encoding="utf-8")
    ofiq = crop.with_suffix(".ofiq_attr.json")
    ofiq.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mod, "_get_max_score", lambda p, ctx: 30.0 if passes else 1.0)
    ctx = SimpleNamespace(
        modality="video",
        input_dir=input_dir.parent,
        output_dir=output_dir.parent,
        cfg=SimpleNamespace(quality_threshold=8.0),
        session=None,
        filter_logger=MagicMock(),
    )
    status, _ = mod._process_crop(crop, ctx)

    if passes:
        assert status == "assessed_pass"
        assert (output_dir / "clip_face_1.ofiq_attr.json").exists()
        assert not ofiq.exists(), "OFIQ annotation must not stay in the source folder"
    else:
        assert status == "assessed_fail"
        assert ofiq.exists(), "failing crops keep their annotation in place"
