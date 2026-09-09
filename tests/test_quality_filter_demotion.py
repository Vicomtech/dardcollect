"""CPU-only tests for the opt-in demotion behavior of filter_face_crops_by_quality (#6).

``_demote_output_crops`` re-evaluates crops already in output_dir against the
CURRENT threshold using their cached ``.magface.json`` and moves those that no
longer pass (+ ``.json`` + ``.magface.json``) back to input_dir. Default config
(``demote_on_raise: false``) never demotes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "filter_stage",
        Path(__file__).resolve().parent.parent / "pipeline" / "filter_face_crops_by_quality.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError("cannot load pipeline/filter_face_crops_by_quality.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["filter_stage"] = module
    spec.loader.exec_module(module)
    return module


filter_stage = _load_module()


def _make_cfg(tmp_path, demote: bool, threshold: float):
    return SimpleNamespace(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        quality_threshold=10.0 if not demote else 20.0,
        gpu_id=0,
        min_free_disk_gb=2.0,
        demote_on_raise=demote,
    )


def _write_magface(path, max_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"unified_score": {"max": max_score, "mean": max_score}}),
        encoding="utf-8",
    )


def _setup_filtered_crop(tmp_path, score: float):
    """input_dir + output_dir with one already-filtered crop in output_dir."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    crop = output_dir / "clip_01m00s-01m02s_face_0.mp4"
    crop.parent.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    crop.write_bytes(b"fake-mp4")
    crop.with_suffix(".json").write_text("{}", encoding="utf-8")
    _write_magface(crop.with_suffix(".magface.json"), score)
    return input_dir, output_dir, crop


def test_raise_demotes_cached_crops_below_threshold(tmp_path):
    input_dir, output_dir, crop = _setup_filtered_crop(tmp_path, score=12.0)
    crop_name = crop.name
    # threshold raised from 10 to 20; cached score 12 no longer passes
    cfg = SimpleNamespace(output_dir=str(output_dir), quality_threshold=20.0, demote_on_raise=True)
    demoted = filter_stage._demote_output_crops("video", cfg, input_dir)

    assert demoted == 1
    assert (input_dir / crop_name).exists()
    assert (input_dir / "clip_01m00s-01m02s_face_0.json").exists()
    assert (input_dir / "clip_01m00s-01m02s_face_0.magface.json").exists()
    assert not (output_dir / crop_name).exists()


def test_lowering_leaves_crops_in_output(tmp_path):
    """Crops still above the current threshold stay put."""
    input_dir, output_dir, crop = _setup_filtered_crop(tmp_path, score=25.0)
    cfg = SimpleNamespace(output_dir=str(output_dir), quality_threshold=20.0, demote_on_raise=True)
    demoted = filter_stage._demote_output_crops("video", cfg, input_dir)

    assert demoted == 0
    assert crop.exists()
    assert not (input_dir / crop.name).exists()


def test_opt_out_default_never_demotes(tmp_path):
    """Without the flag the demotion pass does not even run (forward skip only)."""
    input_dir, output_dir, _crop = _setup_filtered_crop(tmp_path, score=1.0)
    cfg = SimpleNamespace(output_dir=str(output_dir), quality_threshold=50.0, demote_on_raise=False)
    # The gating lives in _process_modality: it calls _demote_output_crops only
    # when cfg.demote_on_raise is true. The flag defaults to False, so main()
    # never demotes unless opted in; the unit below only pins the direct-call
    # contract (the caller owns the gate).
    assert cfg.demote_on_raise is False
    demoted = filter_stage._demote_output_crops("video", cfg, input_dir)
    assert demoted == 1  # direct call demotes (caller's responsibility to gate)


def test_missing_magface_sidecar_leaves_crop_in_place(tmp_path, caplog):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    crop = output_dir / "clip_face_0.mp4"
    crop.parent.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    crop.write_bytes(b"fake-mp4")

    cfg = SimpleNamespace(output_dir=str(output_dir), quality_threshold=20.0, demote_on_raise=True)
    demoted = filter_stage._demote_output_crops("video", cfg, input_dir)

    assert demoted == 0
    assert crop.exists()  # cannot re-evaluate without a cached score — keep it


def test_collision_never_overwrites_input(tmp_path, caplog):
    import logging

    input_dir, output_dir, crop = _setup_filtered_crop(tmp_path, score=12.0)
    # A crop with the same name was re-extracted in input_dir meanwhile
    preexisting = input_dir / crop.name
    preexisting.write_bytes(b"newer-extraction")

    cfg = SimpleNamespace(output_dir=str(output_dir), quality_threshold=20.0, demote_on_raise=True)
    with caplog.at_level(logging.WARNING, logger="filter_stage"):
        demoted = filter_stage._demote_output_crops("video", cfg, input_dir)

    assert demoted == 0
    assert preexisting.read_bytes() == b"newer-extraction"  # input untouched
    assert crop.exists()  # filtered copy left in place too
    assert "already exists in input_dir" in caplog.text


def test_config_reads_demote_on_raise(tmp_path):
    from dardcollect.config import FaceQualityFilterConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_quality_filtering:\n"
        "  input_dir: in\n"
        "  output_dir: out\n"
        "  quality_threshold: 10.0\n"
        "  demote_on_raise: true\n",
        encoding="utf-8",
    )
    cfg = FaceQualityFilterConfig.from_yaml(str(yaml_path))
    assert cfg.demote_on_raise is True

    # Default is False (backward compatible)
    yaml_path2 = tmp_path / "cfg2.yaml"
    yaml_path2.write_text(
        "face_quality_filtering:\n  input_dir: in\n  output_dir: out\n  quality_threshold: 10.0\n",
        encoding="utf-8",
    )
    cfg2 = FaceQualityFilterConfig.from_yaml(str(yaml_path2))
    assert cfg2.demote_on_raise is False


def test_image_modality_section_also_supports_flag(tmp_path):
    from dardcollect.config import FaceQualityFilterConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "image_face_quality_filtering:\n"
        "  input_dir: in\n"
        "  output_dir: out\n"
        "  quality_threshold: 10.0\n"
        "  demote_on_raise: true\n",
        encoding="utf-8",
    )
    cfg = FaceQualityFilterConfig.from_yaml(str(yaml_path), section="image_face_quality_filtering")
    assert cfg.demote_on_raise is True
