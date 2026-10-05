"""CPU-only tests for corner-only crop stabilization (issue #9).

Synthetic frames: a textured rectangle warped from jittered corners vs its
static median-corner render. Stabilization ON (default since 2026-09-30) makes
the crop background constant across frames; explicit OFF reproduces the
per-frame rendering. Design doc: docs/DESIGN_crop_stabilization.md.
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
face_stabilization = _load_module("fs_mod", "dardcollect/face_stabilization.py")


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
    median = face_stabilization.compute_track_mean_corners(corners, min_frames=5)
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
    median = face_stabilization.compute_track_mean_corners(corners, min_frames=5)
    assert np.abs(median - base).max() < 1e-6


def test_fewer_than_min_frames_returns_none():
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = [base.copy() for _ in range(4)]
    assert face_stabilization.compute_track_mean_corners(corners, min_frames=5) is None


def test_all_none_returns_none():
    assert face_stabilization.compute_track_mean_corners([None, None, None], min_frames=5) is None


def test_smoothed_trajectory_kills_jitter_but_stays_centred():
    """The smoothed per-frame quads remove frame-to-frame jitter while keeping
    their mean on the true (static) centre — the face does not freeze off-axis."""
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = _jittered_corners(base, jitter=3.0, n=60)
    smoothed = face_stabilization.smooth_track_corners(corners, fps=25.0, window_seconds=0.4)
    assert smoothed[0] is not None

    # Frame-to-frame wobble (what makes the crop tremble) is far below raw.
    raw = np.stack([c for c in corners if c is not None])[:, 0, 0]
    sm = np.stack([c for c in smoothed if c is not None])[:, 0, 0]
    assert np.diff(sm).std() < np.diff(raw).std() * 0.35
    # Centred: the smoothed trajectory tracks the base position.
    assert abs(sm.mean() - base[0, 0]) < 1.0


def test_smoothed_trajectory_follows_real_motion():
    """A genuine slow translation must be followed (eyes stay aligned), unlike
    the old global median which froze the whole track on one quad."""
    n = 40
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    drift = np.linspace(0.0, 40.0, n).astype(np.float32)
    corners = [(base + np.array([d, 0.0], dtype=np.float32)).astype(np.float32) for d in drift]
    smoothed = face_stabilization.smooth_track_corners(corners, fps=25.0, window_seconds=0.4)
    last = smoothed[-1]
    # Endpoint error small vs the 40 px total travel, and clearly moving
    # (a median would sit at +20 px and never reach the end).
    assert abs(float(last[0, 0]) - (100.0 + 40.0)) < 3.0
    assert float(last[0, 0]) - float(smoothed[0][0, 0]) > 30.0


def test_smoothed_trajectory_interpolates_gaps_only_in_range():
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners: list = [base.copy() for _ in range(30)]
    corners[10] = None  # interior gap
    smoothed = face_stabilization.smooth_track_corners(corners, fps=25.0, window_seconds=0.4)
    assert smoothed[10] is None  # interior gap stays None (frame has no crop)
    assert smoothed[0] is not None and smoothed[-1] is not None


def test_sidecar_corners_stay_raw():
    """Design invariant: stabilization is render-time only — the function
    never mutates its inputs."""
    base = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    corners = [base.copy() for _ in range(8)]
    before = [c.copy() for c in corners]
    face_stabilization.compute_track_mean_corners(corners, min_frames=5)
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
        "  stabilization_min_frames: 7\n"
        "  stabilization_window_seconds: 0.4\n",
        encoding="utf-8",
    )
    cfg = FaceCropConfig.from_yaml(str(yaml_path))
    assert cfg.stabilize_face_crops is True
    assert cfg.stabilization_min_frames == 7
    assert cfg.stabilization_window_seconds == 0.4


def test_config_stabilization_defaults_on(tmp_path):
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n  input_dir: in\n  output_dir: out\n",
        encoding="utf-8",
    )
    cfg = FaceCropConfig.from_yaml(str(yaml_path))
    assert cfg.stabilize_face_crops is True
    assert cfg.stabilization_min_frames == 5
    assert cfg.stabilization_window_seconds == 0.4


# ── 2-pass stabilization (2026-09-16 fix: O(1) source-frame memory) ─────────
# The single-pass design retained every full-resolution source frame in memory
# (~11 GB for a 60 s 1080p clip). The fix plans corners from the sidecar JSON
# (no pixels), then re-decodes once and renders through the track-median quad.


CFG = SimpleNamespace(
    max_overlap_iou=0.3,
    stabilization_min_frames=5,
    stabilization_window_seconds=0.4,
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
    plan, stabs = face_stabilization.plan_stabilized_track_crops(frame_data, 0, CFG, 7, fps=25.0)
    assert plan[0][0] is None  # overlap (symmetric)
    assert plan[0][1] is None  # overlap (symmetric)
    assert plan[1][0] is not None
    assert plan[1][2] is None  # no corners
    assert plan[6] == {}  # frame beyond the sidecar data
    assert stabs[0].median is not None  # 6 stable frames >= min_frames
    assert stabs[0].per_frame[0] is not None  # smoothed series engaged
    assert stabs[1].median is None  # 1 stable frame < min_frames -> fallback
    assert stabs[1].per_frame == [None] * 7
    assert stabs[2].median is None


def _make_gradient_video(path: Path, n_frames: int, width: int, height: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (width, height))
    assert writer.isOpened()
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(height)[:, None] * 3 % 256).astype(np.uint8)
        writer.write(frame)
    writer.release()


def test_two_pass_render_stabilizes_and_falls_back(tmp_path):
    """Pass 2 renders engaged tracks through their smoothed per-frame quad and
    short tracks through their per-frame corners (the fallback)."""
    import cv2

    n_frames = 12
    vid = tmp_path / "grad.mp4"
    _make_gradient_video(vid, n_frames, 320, 240)

    base_b = np.array([[240, 60], [310, 60], [310, 130], [240, 130]], dtype=np.float32)
    rng = np.random.default_rng(7)
    # Track 0 drifts slowly (real motion) on top of jitter, so the smoothed
    # per-frame quad is clearly not the constant median.
    jittered_a = [
        (_BASE + np.array([i * 2.0, 0.0], np.float32) + rng.uniform(-2, 2, (4, 2))).astype(
            np.float32
        )
        for i in range(n_frames)
    ]
    jittered_b = [(base_b + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for _ in range(2)]

    frame_data: dict = {}
    for i in range(n_frames):
        dets = [_det(0, jittered_a[i])]
        if i < 2:  # track 1 is short -> per-frame fallback
            dets.append(_det(1, jittered_b[i], bbox=[240, 60, 310, 130]))
        frame_data[str(i)] = dets

    plan, stabs = face_stabilization.plan_stabilized_track_crops(
        frame_data, 0, CFG, n_frames, fps=25.0
    )
    track_frames = face_stabilization.render_stabilized_track_frames(vid, plan, stabs, n_frames)

    assert stabs[0].median is not None
    assert stabs[1].median is None
    assert len(track_frames[0]) == n_frames
    assert all(oc is not None for _, oc in track_frames[0])

    # Engaged track: renders follow the smoothed per-frame quads, NOT the
    # constant median (so genuine motion is followed, jitter is gone).
    ref = cv2.VideoCapture(str(vid))
    ok, src_frame = ref.read()
    ref.release()
    assert ok
    expected0 = face_geometry._corners_to_warp(src_frame, stabs[0].per_frame[0], 616)
    expected_med = face_geometry._corners_to_warp(src_frame, stabs[0].median, 616)
    assert np.array_equal(track_frames[0][0][1], expected0)
    assert not np.array_equal(track_frames[0][0][1], expected_med)

    # Fallback track: crops follow the per-frame corners (they differ).
    assert len(track_frames[1]) == 2
    assert not np.array_equal(track_frames[1][0][1], track_frames[1][1][1])


def test_sidecar_annotations_use_render_warp(tmp_path):
    """Regression (filtered-crop misalignment): crop sidecars must store
    keypoints/bbox warped with the quad the pixels were rendered through
    (the smoothed per-frame quad when stabilization engaged) — not a per-frame
    re-estimated alignment, which drifts several pixels from the rendered crop."""
    import dardcollect.face_crop_writers as writers
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n  input_dir: in\n  output_dir: out\n",
        encoding="utf-8",
    )
    face_config = FaceCropConfig.from_yaml(str(yaml_path))

    rng = np.random.default_rng(11)
    base = np.array([[100, 60], [220, 60], [220, 180], [100, 180]], dtype=np.float32)
    src_nose = [160.0, 120.0]
    frame_data: dict = {}
    for i in range(12):
        quad = (base + np.array([i, 0], np.float32) + rng.uniform(-3, 3, (4, 2))).astype(np.float32)
        frame_data[str(i)] = [
            {
                "track_id": 0,
                "bbox": [100, 60, 220, 180],
                "score": 0.9,
                "keypoints": [src_nose] + [[0.0, 0.0]] * 132,
                "keypoint_scores": [0.9] * 133,
                "face_crop_corners_ofiq": [[float(x), float(y)] for x, y in quad],
            }
        ]
    _, stabs = face_stabilization.plan_stabilized_track_crops(frame_data, 0, CFG, 12, fps=25.0)
    assert stabs[0].median is not None
    smoothed0 = stabs[0].per_frame[0]
    assert smoothed0 is not None

    det = frame_data["0"][0]
    entry_pf = writers._build_track_frame_entry(det, 0, face_config, [], None)
    entry_sm = writers._build_track_frame_entry(det, 0, face_config, [], smoothed0)
    assert entry_pf is not None and entry_sm is not None

    expected = face_geometry.warp_points_to_output([src_nose], smoothed0)[0]
    assert np.allclose(entry_sm["keypoints"][0], expected)
    # The smoothed warp differs from the raw per-frame alignment (the old bug
    # stored the latter while pixels used the former).
    assert not np.allclose(entry_sm["keypoints"][0], entry_pf["keypoints"][0])
    assert "bbox" in entry_sm


def test_two_pass_render_bounded_memory(tmp_path):
    """300 frames @ 1280×720: retaining source frames would need >= 0.83 GB;
    the 2-pass design must stay well under that (it holds one frame + crops)."""
    import resource
    import sys

    n_frames = 300
    vid = tmp_path / "big.mp4"
    _make_gradient_video(vid, n_frames, 1280, 720)

    frame_data = {str(i): [_det(0, _BASE)] for i in range(n_frames)}

    # ru_maxrss unit: KB on Linux, bytes on macOS/Windows
    scale = 1024**2 if sys.platform in ("darwin", "win32") else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    plan, stabs = face_stabilization.plan_stabilized_track_crops(
        frame_data, 0, CFG, n_frames, fps=25.0
    )
    track_frames = face_stabilization.render_stabilized_track_frames(vid, plan, stabs, n_frames)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert len(track_frames[0]) == n_frames
    delta_mb = (after - before) / scale
    msg = f"2-pass render used {delta_mb:.0f} MB peak delta (retention regression?)"
    assert delta_mb < 700, msg
