"""Unit tests for `dardcollect.config` — config parsing + log-level resolution.

Pure CPU tests: no GPU, no models. They write minimal YAML to a pytest tmp_path
and assert the dataclass loaders + `get_log_level` behave per their contracts.
"""

import logging

import pytest
import yaml

from dardcollect.config import DEFAULT_MODELS_PATH, DetectorConfig, get_log_level


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


# ── get_log_level ─────────────────────────────────────────────────────────────


def test_get_log_level_resolves_named_level(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", {"log_level": "DEBUG"})
    assert get_log_level(cfg) == logging.DEBUG


def test_get_log_level_is_case_insensitive(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", {"log_level": "warning"})
    assert get_log_level(cfg) == logging.WARNING


def test_get_log_level_defaults_to_info_when_absent(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", {})
    assert get_log_level(cfg) == logging.INFO


def test_get_log_level_falls_back_to_info_on_unknown_level(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", {"log_level": "NONSENSE"})
    assert get_log_level(cfg) == logging.INFO


# ── DetectorConfig.from_yaml ──────────────────────────────────────────────────


_MINIMAL_PERSON_EXTRACTION = {
    "person_extraction": {
        "detection_threshold": 0.5,
        "tracking_score_threshold": 0.4,
        "tracking_min_hits": 3,
        "tracking_max_time_lost": 30,
        "pose_keypoint_threshold": 0.3,
    }
}


def test_detector_config_from_yaml_loads_required_fields(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", _MINIMAL_PERSON_EXTRACTION)
    dc = DetectorConfig.from_yaml(cfg)
    assert dc.detection_threshold == 0.5
    assert dc.tracking_min_hits == 3
    assert dc.pose_keypoint_threshold == 0.3
    # Optional fields fall back to documented defaults.
    assert dc.models_path == DEFAULT_MODELS_PATH
    assert dc.detection_model_type == 0
    assert dc.gpu_id == 0


def test_detector_config_from_yaml_uses_gpu_id_override(tmp_path):
    data = copy_with(_MINIMAL_PERSON_EXTRACTION, gpu_id=2)
    cfg = _write_yaml(tmp_path / "config.yaml", data)
    assert DetectorConfig.from_yaml(cfg).gpu_id == 2


def test_detector_config_from_yaml_missing_section_raises(tmp_path):
    cfg = _write_yaml(tmp_path / "config.yaml", {"other_section": {}})
    with pytest.raises(ValueError, match="person_extraction"):
        DetectorConfig.from_yaml(cfg)


def test_detector_config_from_yaml_missing_required_key_raises(tmp_path):
    data = copy_with(_MINIMAL_PERSON_EXTRACTION, drop=("detection_threshold",))
    cfg = _write_yaml(tmp_path / "config.yaml", data)
    with pytest.raises(ValueError, match="detection_threshold"):
        DetectorConfig.from_yaml(cfg)


def test_detector_config_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DetectorConfig.from_yaml(str(tmp_path / "absent.yaml"))


# ── helpers ────────────────────────────────────────────────────────────────────


def copy_with(base: dict, *, drop=None, **overrides) -> dict:
    """Deep-ish copy of `base` with top-level overrides applied to
    `person_extraction` and optional key removal."""
    import copy

    out = copy.deepcopy(base)
    pe = out.setdefault("person_extraction", {})
    if drop:
        for key in drop:
            pe.pop(key, None)
    for k, v in overrides.items():
        if k in (
            "detection_threshold",
            "tracking_score_threshold",
            "tracking_min_hits",
            "tracking_max_time_lost",
            "pose_keypoint_threshold",
            "models_path",
        ):
            pe[k] = v
        else:
            out[k] = v
    return out
