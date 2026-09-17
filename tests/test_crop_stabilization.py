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
from types import SimpleNamespace

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
    for orig, now in zip(corners, before):
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


# ── 2-pass stabilization (2026-09-16 fix: O(1) source-frame memory) ─────────
# The single-pass design retained every full-resolution source frame in memory
# (~11 GB for a 60 s 1080p clip). The fix plans corners from the sidecar JSON
# (no pixels), then re-decodes once and renders through the track-median quad.


CFG = SimpleNamespace(
    max_overlap_iou=0.3,
    stabilization_min_frames=5,
    pose_keypoint_threshold=0.5,
    min_eye_distance_px=10.0,
)

_BASE = np.array([[100, 60], [220, 60], [220, 180], [100, 180]], dtype=np.float32)


def _det(tid: int, corners: np.ndarray, bbox: list | None = None) -> dict:
    return {
        "track_id": tid,
        "bbox": bbox or [100, 60, 220, 180],
        "face_crop_corners_ofiq": [[float(x), float(y)] for x, y in corners],
    }


def test_plan_marks_overlap_and_gap_frames():
    """Pass 1 (JSON-only) applies the same inclusion rule as the per-frame
    path: overlapping detections (both tracks, the check is symmetric) and
    corner-less frames contribute None."""
    det_a = _det(0, _BASE)
    det_b = _det(1, np.array(_BASE) + 5.0)  # IoU ≈ 0.82 > max_overlap_iou
    no_corners = {"track_id": 2, "bbox": [300, 60, 320, 180]}
    frame_data = {
        "0": [det_a, det_b],  # overlap -> both tracks None on this frame
        "1": [det_a, no_corners],  # track 2 has no corners -> None
        "2": [det_a],
        "3": [det_a],
        "4": [det_a],
        "5": [det_a],
    }
    plan, medians = face_geometry.plan_stabilized_track_crops(frame_data, 0, CFG, 7)
    assert plan[0][0] is None  # overlap (symmetric)
    assert plan[0][1] is None  # overlap (symmetric)
    assert plan[1][0] is not None
    assert plan[1][2] is None  # no corners
    assert plan[6] == {}  # frame beyond the sidecar data
    assert medians[0] is not None  # 5 stable frames >= min_frames
    assert medians[1] is None  # 1 stable frame < min_frames -> fallback
    assert medians[2] is None


def _make_gradient_video(path: Path, n_frames: int, width: int, height: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height))
    assert writer.isOpened()
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(height)[:, None] * 3 % 256).astype(np.uint8)
        writer.write(frame)
    writer.release()


def test_two_pass_render_stabilizes_and_falls_back(tmp_path):
    """Pass 2 renders engaged tracks through the median quad (constant across
    frames) and short tracks through their per-frame corners (the fallback)."""
    import cv2

    n_frames = 10
    vid = tmp_path / "grad.mp4"
    _make_gradient_video(vid, n_frames, 320, 240)

    base_b = np.array([[240, 60], [310, 60], [310, 130], [240, 130]], dtype=np.float32)
    rng = np.random.default_rng(7)
    jittered_a = [(_BASE + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for _ in range(n_frames)]
    jittered_b = [(base_b + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for _ in range(2)]

    frame_data: dict = {}
    for i in range(n_frames):
        dets = [_det(0, jittered_a[i])]
        if i < 2:  # track 1 is short -> per-frame fallback
            dets.append(_det(1, jittered_b[i], bbox=[240, 60, 310, 130]))
        frame_data[str(i)] = dets

    plan, medians = face_geometry.plan_stabilized_track_crops(frame_data, 0, CFG, n_frames)
    track_frames = face_geometry.render_stabilized_track_frames(vid, plan, medians, n_frames)

    assert medians[0] is not None
    assert medians[1] is None
    assert len(track_frames[0]) == n_frames
    assert all(oc is not None for _, oc in track_frames[0])

    # Engaged track: every crop is the same (jitter killed) and equals the
    # median-quad warp of the (identical) source frames.
    from itertools import pairwise

    for a, b in pairwise(track_frames[0]):
        assert np.array_equal(a[1], b[1])

    ref = cv2.VideoCapture(str(vid))
    ok, src_frame = ref.read()
    ref.release()
    assert ok
    expected = face_geometry._corners_to_warp(src_frame, medians[0], 616)
    assert np.array_equal(track_frames[0][0][1], expected)

    # Fallback track: crops follow the per-frame corners (they differ).
    assert len(track_frames[1]) == 2
    assert not np.array_equal(track_frames[1][0][1], track_frames[1][1][1])


def test_two_pass_render_bounded_memory(tmp_path):
    """300 frames @ 1280×720: retaining source frames would need >= 0.83 GB;
    the 2-pass design must stay well under that (it holds one frame + crops)."""
    import resource

    n_frames = 300
    vid = tmp_path / "big.mp4"
    _make_gradient_video(vid, n_frames, 1280, 720)

    frame_data = {str(i): [_det(0, _BASE)] for i in range(n_frames)}

    # ru_maxrss unit: KB on Linux, bytes on macOS/Windows
    scale = 1024**2 if sys.platform in ("darwin", "win32") else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    plan, medians = face_geometry.plan_stabilized_track_crops(frame_data, 0, CFG, n_frames)
    track_frames = face_geometry.render_stabilized_track_frames(vid, plan, medians, n_frames)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert len(track_frames[0]) == n_frames
    delta_mb = (after - before) / scale
    msg = f"2-pass render used {delta_mb:.0f} MB peak delta (retention regression?)"
    assert delta_mb < 700, msg
