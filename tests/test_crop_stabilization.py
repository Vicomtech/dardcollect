"""CPU-only tests for corner-only crop stabilization (issue #9).

Synthetic frames: a textured rectangle warped from jittered corners vs its
static median-corner render. Stabilization OFF (default) must reproduce the
per-frame rendering; ON must make the crop background constant across frames.
Design doc: docs/DESIGN_crop_stabilization.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / rel_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {rel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


face_geometry = _load_module("fg_mod", "dardcollect/face_geometry.py")


def _jittered_corners(base: np.ndarray, jitter: float, n: int, seed: int = 7) -> list:
    """n corner quads: base + uniform sub-pixel jitter (None sprinkled in)."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray | None] = []
    for i in range(n):
        if i % 7 == 3:  # some frames have no valid corners (gap handling)
            out.append(None)
            continue
        noise = rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
        out.append((base + noise).astype(np.float32))
    return out


def test_median_corners_constant_for_jittered_track():
    """The median quad of jittered corners is stable (not the mean of extremes)."""
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = _jittered_corners(base, jitter=2.0, n=30)
    median = face_geometry.compute_track_mean_corners(corners, min_frames=5)
    assert median is not None
    assert median.shape == (4, 2)
    # Every component within 1.5px of the base quad (robustness vs outliers)
    assert np.abs(median - base).max() < 1.5


def test_median_is_median_not_mean():
    """One big outlier must not move the median (it would move the mean)."""
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners: list = [base.copy() for _ in range(10)]
    outlier = base + 50.0  # one landmark failure gone wild
    corners[5] = outlier.astype(np.float32)
    median = face_geometry.compute_track_mean_corners(corners, min_frames=5)
    assert np.abs(median - base).max() < 1e-6


def test_fewer_than_min_frames_returns_none():
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = [base.copy() for _ in range(4)]
    assert face_geometry.compute_track_mean_corners(corners, min_frames=5) is None


def test_all_none_returns_none():
    assert face_geometry.compute_track_mean_corners([None, None, None], min_frames=5) is None


def test_stabilized_render_stops_background_jitter():
    """Render a gradient-textured frame from jittered corners vs the median
    quad: the stabilized render is identical across frames (no wobble), while
    per-frame jittered corners produce visibly shifting rows (the wobble)."""
    from itertools import pairwise

    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = _jittered_corners(base, jitter=3.0, n=20)
    median = face_geometry.compute_track_mean_corners(corners, min_frames=5)
    # Fine vertical gradient: every row differs, so sub-pixel warp shifts show
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[:, :, 0] = (np.arange(300)[:, None] * 3 % 256).astype(np.uint8)

    stabilized = [face_geometry._corners_to_warp(frame, median, 64) for _ in range(10)]
    # All stabilized renders are identical (same source frame, same quad)
    for a, b in pairwise(stabilized):
        assert np.array_equal(a, b)

    per_frame = [face_geometry._corners_to_warp(frame, c, 64) for c in corners if c is not None]
    # The per-frame renders actually differ (the jitter was real)
    assert any(not np.array_equal(a, b) for a, b in pairwise(per_frame))


def test_sidecar_corners_stay_raw():
    """Design invariant: stabilization is render-time only — the function
    never mutates its inputs."""
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = [base.copy() for _ in range(8)]
    before = [c.copy() for c in corners]
    face_geometry.compute_track_mean_corners(corners, min_frames=5)
    for orig, now in zip(corners, before, strict=True):
        assert np.array_equal(orig, now)


def test_config_reads_stabilization_keys(tmp_path):
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n"
        "  input_dir: in\n"
        "  output_dir: out\n"
        "  stabilize_face_crops: true\n"
        "  stabilization_min_frames: 7\n",
        encoding="utf-8",
    )
    cfg = FaceCropConfig.from_yaml(str(yaml_path))
    assert cfg.stabilize_face_crops is True
    assert cfg.stabilization_min_frames == 7


def test_config_stabilization_defaults_off(tmp_path):
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n  input_dir: in\n  output_dir: out\n",
        encoding="utf-8",
    )
    cfg = FaceCropConfig.from_yaml(str(yaml_path))
    assert cfg.stabilize_face_crops is False
    assert cfg.stabilization_min_frames == 5
