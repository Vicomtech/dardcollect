"""CPU tests for eye-anchored OFIQ video-crop stabilization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


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

CFG = SimpleNamespace(
    max_overlap_iou=0.3,
    stabilization_min_frames=5,
    stabilization_window_seconds=0.8,
    stabilization_anchor_tolerance_px=2.5,
    pose_keypoint_threshold=0.5,
    min_eye_distance_px=10.0,
)

_BASE = np.array([[100, 60], [220, 60], [220, 180], [100, 180]], dtype=np.float32)
_CANONICAL_EYE_MIDPOINT = np.array([307.5, 272.0], dtype=np.float32)


def _eye_midpoint_for_quad(quad: np.ndarray) -> np.ndarray:
    inverse = cv2.invertAffineTransform(face_geometry._output_warp_matrix(quad))
    point = cv2.transform(_CANONICAL_EYE_MIDPOINT.reshape(1, 1, 2), inverse)
    return point.reshape(2)


def _eye_centers_for_quad(quad: np.ndarray) -> np.ndarray:
    midpoint = _eye_midpoint_for_quad(quad)
    edge = quad[1] - quad[0]
    unit = edge / np.linalg.norm(edge)
    scale = np.linalg.norm(edge) / 616.0
    offset = unit * (scale * 113.0 / 2.0)
    return np.stack([midpoint - offset, midpoint + offset])


def _det(tid: int, corners: np.ndarray, bbox: list | None = None) -> dict:
    eye_centers = _eye_centers_for_quad(corners)
    eyes_mid = eye_centers.mean(axis=0)
    keypoints = np.zeros((133, 2), dtype=np.float32)
    contour = np.array([[-3, 0], [-2, -2], [0, -3], [3, 0], [2, 2], [0, 3]])
    keypoints[59:65] = eye_centers[0] + contour
    keypoints[65:71] = eye_centers[1] + contour
    keypoints[1] = eye_centers[1]
    keypoints[2] = eye_centers[0]
    keypoints[0] = eyes_mid + np.array([0.0, 30.0])
    keypoints[71] = eyes_mid + np.array([-15.0, 50.0])
    keypoints[77] = eyes_mid + np.array([15.0, 50.0])
    scores = np.full(133, 0.9, dtype=np.float32)
    return {
        "track_id": tid,
        "bbox": bbox or [100, 60, 220, 180],
        "score": 0.9,
        "keypoints": keypoints.tolist(),
        "keypoint_scores": scores.tolist(),
        "face_crop_corners_ofiq": corners.astype(float).tolist(),
    }


def _pose_quad(eye: np.ndarray, angle: float, scale: float) -> np.ndarray:
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    tl = eye - scale * (_CANONICAL_EYE_MIDPOINT[0] * u + _CANONICAL_EYE_MIDPOINT[1] * v)
    side = scale * 616.0
    tr, br, bl = tl + side * u, tl + side * (u + v), tl + side * v
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _make_gradient_video(path: Path, n_frames: int, width: int, height: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (width, height))
    assert writer.isOpened()
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(height)[:, None] * 3 % 256).astype(np.uint8)
        writer.write(frame)
    writer.release()


def test_eye_anchored_smoothing_reduces_pose_jitter_and_tracks_eyes():
    rng = np.random.default_rng(24)
    n = 80
    quads, eye_centers = [], []
    raw_angles, raw_scales = [], []
    for i in range(n):
        eye = np.array([500.0 + i * 0.45, 330.0 + i * 0.15])
        eye += rng.normal(0.0, 1.5, size=2)
        angle = 0.025 * np.sin(i / 9.0) + rng.normal(0.0, 0.035)
        scale = 1.4 + 0.004 * np.sin(i / 13.0) + rng.normal(0.0, 0.04)
        vector = scale * 113.0 * np.array([np.cos(angle), np.sin(angle)])
        eye_centers.append(np.stack([eye - vector / 2.0, eye + vector / 2.0]).astype(np.float32))
        quads.append(_pose_quad(eye, angle, scale))
        raw_angles.append(angle)
        raw_scales.append(scale)

    smoothing = face_stabilization.EyeAnchorSmoothing(29.97, 0.65, 2.5, 5)
    stabilized = face_stabilization.smooth_quad_pose_anchored_to_eyes(quads, eye_centers, smoothing)
    assert all(quad is not None for quad in stabilized)
    eye_targets = np.stack(
        [
            face_geometry.warp_points_to_output([eye.mean(axis=0).tolist()], quad)[0]
            for eye, quad in zip(eye_centers, stabilized, strict=True)
        ]
    )
    eye_errors = np.linalg.norm(eye_targets - _CANONICAL_EYE_MIDPOINT, axis=1)
    assert np.percentile(eye_errors, 95) < 2.5
    assert eye_errors.max() <= 2.5 + 1e-3
    raw_matrices = np.stack([face_geometry._output_warp_matrix(quad) for quad in quads])
    stable_matrices = np.stack([face_geometry._output_warp_matrix(quad) for quad in stabilized])
    raw_accel = np.diff(raw_matrices[:, :2, 2], n=2, axis=0)
    stable_accel = np.diff(stable_matrices[:, :2, 2], n=2, axis=0)
    raw_wobble = np.sqrt(np.mean(np.sum(raw_accel**2, axis=1)))
    stable_wobble = np.sqrt(np.mean(np.sum(stable_accel**2, axis=1)))
    assert stable_wobble < raw_wobble * 0.1

    new_angles, new_scales = [], []
    for quad in stabilized:
        edge = quad[1] - quad[0]
        new_angles.append(float(np.arctan2(edge[1], edge[0])))
        new_scales.append(float(np.linalg.norm(edge) / 616.0))
    raw_pose_accel = np.std(np.diff(np.unwrap(raw_angles), n=2)) + np.std(
        np.diff(np.log(raw_scales), n=2)
    )
    stable_pose_accel = np.std(np.diff(np.unwrap(new_angles), n=2)) + np.std(
        np.diff(np.log(new_scales), n=2)
    )
    assert stable_pose_accel < raw_pose_accel * 0.4
    # Translation follows the real source-face motion rather than a track median.
    assert np.linalg.norm(eye_centers[-1].mean(axis=0) - eye_centers[0].mean(axis=0)) > 30.0


def test_eye_anchored_smoothing_preserves_gaps_and_short_tracks():
    quad = _BASE.copy()
    eyes: list[np.ndarray | None] = [_eye_centers_for_quad(quad) for _ in range(12)]
    corners: list[np.ndarray | None] = [quad.copy() for _ in range(12)]
    corners[5] = eyes[5] = None
    smoothing = face_stabilization.EyeAnchorSmoothing(25.0, 0.4, 2.5, 5)
    result = face_stabilization.smooth_quad_pose_anchored_to_eyes(corners, eyes, smoothing)
    assert result[5] is None
    assert result[0] is not None and result[-1] is not None
    short = face_stabilization.smooth_quad_pose_anchored_to_eyes(corners[:4], eyes[:4], smoothing)
    assert short == [None] * 4


def test_config_stabilization_defaults_on(tmp_path):
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n  input_dir: in\n  output_dir: out\n", encoding="utf-8"
    )
    cfg = FaceCropConfig.from_yaml(str(yaml_path))
    assert cfg.stabilize_face_crops is True
    assert cfg.stabilization_min_frames == 5
    assert cfg.stabilization_window_seconds == 0.8
    assert cfg.stabilization_anchor_tolerance_px == 2.5


def test_stabilization_fails_loudly_without_eye_contour_support():
    det = _det(0, _BASE)
    det["keypoint_scores"][59:63] = [0.1] * 4
    with pytest.raises(ValueError, match="at least three confident contour landmarks"):
        face_stabilization.plan_stabilized_track_crops({"0": [det]}, 0, CFG, 1, 25.0)


def test_plan_marks_overlap_and_missing_corner_frames():
    det_a = _det(0, _BASE)
    det_b = _det(1, _BASE + 5.0)
    no_corners = {"track_id": 2, "bbox": [300, 60, 320, 180]}
    frame_data = {
        "0": [det_a, det_b],
        "1": [det_a, no_corners],
        "2": [det_a],
        "3": [det_a],
        "4": [det_a],
        "5": [det_a],
    }
    plan, stabs = face_stabilization.plan_stabilized_track_crops(frame_data, 0, CFG, 7, 25.0)
    assert plan[0][0] is None and plan[0][1] is None
    assert plan[1][0] is not None and plan[1][2] is None
    assert plan[6] == {}
    assert stabs[0].per_frame[0] is not None
    assert stabs[1].per_frame == [None] * 7
    assert stabs[2].per_frame == [None] * 7


def test_two_pass_render_uses_eye_anchored_quad_and_raw_short_track(tmp_path):
    import cv2

    n_frames = 12
    video = tmp_path / "gradient.mp4"
    _make_gradient_video(video, n_frames, 320, 240)
    base_b = np.array([[240, 60], [310, 60], [310, 130], [240, 130]], dtype=np.float32)
    rng = np.random.default_rng(7)
    moved = [
        (_BASE + np.array([i * 2.0, 0.0]) + rng.uniform(-2, 2, (4, 2))).astype(np.float32)
        for i in range(n_frames)
    ]
    short = [(base_b + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for _ in range(2)]
    frame_data = {
        str(i): [_det(0, moved[i]), *([_det(1, short[i], [240, 60, 310, 130])] if i < 2 else [])]
        for i in range(n_frames)
    }
    plan, stabs = face_stabilization.plan_stabilized_track_crops(frame_data, 0, CFG, n_frames, 25.0)
    rendered = face_stabilization.render_stabilized_track_frames(video, plan, stabs, n_frames)
    assert all(quad is not None for quad in stabs[0].per_frame)
    assert stabs[1].per_frame == [None] * n_frames
    assert len(rendered[0]) == n_frames and all(frame is not None for _, frame in rendered[0])

    cap = cv2.VideoCapture(str(video))
    ok, source = cap.read()
    cap.release()
    assert ok
    expected = face_geometry._corners_to_warp(source, stabs[0].per_frame[0], 616)
    assert np.array_equal(rendered[0][0][1], expected)
    assert len(rendered[1]) == 2
    assert not np.array_equal(rendered[1][0][1], rendered[1][1][1])


def test_sidecar_annotations_and_quad_match_pixel_render(tmp_path):
    import dardcollect.face_crop_writers as writers
    from dardcollect.config import FaceCropConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "face_crop_extraction:\n  input_dir: in\n  output_dir: out\n", encoding="utf-8"
    )
    face_config = FaceCropConfig.from_yaml(str(yaml_path))
    rng = np.random.default_rng(11)
    frame_data = {}
    for i in range(12):
        quad = (_BASE + np.array([i, 0]) + rng.uniform(-3, 3, (4, 2))).astype(np.float32)
        det = _det(0, quad)
        det["keypoints"][0] = [160.0, 120.0]
        frame_data[str(i)] = [det]
    _, stabs = face_stabilization.plan_stabilized_track_crops(frame_data, 0, CFG, 12, 25.0)
    render_quad = stabs[0].per_frame[0]
    assert render_quad is not None
    entry = writers._build_track_frame_entry(frame_data["0"][0], 0, face_config, [], render_quad)
    assert entry is not None
    assert np.array_equal(np.asarray(entry["render_quad_source"]), render_quad.astype(np.float32))
    expected = face_geometry.warp_points_to_output([[160.0, 120.0]], render_quad)[0]
    assert np.allclose(entry["keypoints"][0], expected)
    raw_entry = writers._build_track_frame_entry(frame_data["0"][0], 0, face_config, [], None)
    assert raw_entry is not None
    assert not np.allclose(entry["render_quad_source"], raw_entry["render_quad_source"])


def test_keep_all_gap_repeats_image_quad_and_annotations(tmp_path):
    import dardcollect.face_crop_writers as writers
    from dardcollect.config import FaceCropConfig
    from dardcollect.face_crop_writers import _CropWriteContext

    q0 = _BASE.copy()
    q2 = (_BASE + np.array([12.0, 0.0])).astype(np.float32)
    detections = {"10": [_det(0, q0)], "11": [_det(0, _BASE + 6.0)], "12": [_det(0, q2)]}
    config = FaceCropConfig(
        input_dir="in",
        output_dir="out",
        detection_threshold=0.3,
        pose_keypoint_threshold=0.3,
        min_eye_distance_px=10.0,
        min_track_face_frames=1,
        skip_no_face_frames=False,
    )
    crop0 = np.full((616, 616, 3), 10, dtype=np.uint8)
    crop2 = np.full((616, 616, 3), 30, dtype=np.uint8)
    ctx = _CropWriteContext(
        video_path=Path("clip.mp4"),
        clip_data={},
        frame_data_orig=detections,
        start_frame=10,
        face_config=config,
        output_dir=tmp_path,
        fps=25.0,
        encoding=None,
        face_crops_logger=None,
        black_ofiq=np.zeros_like(crop0),
        arcface_corners_json=[],
        stabilizations={0: face_stabilization.TrackStabilization([q0, q2, q2], 2.5)},
    )

    images, frame_data = writers._collect_track_frames_keep_all(
        ctx, 0, [(0, crop0), (1, None), (2, crop2)]
    )
    assert images[0] is images[1] is crop0
    assert np.array_equal(images[2], crop2)
    first, repeated, last = (frame_data[str(i)][0] for i in range(3))
    assert repeated["render_quad_source"] == first["render_quad_source"]
    assert repeated["keypoints"] == first["keypoints"]
    assert repeated["source_frame_index"] == first["source_frame_index"] == 10
    assert last["source_frame_index"] == 12
    assert last["render_quad_source"] != first["render_quad_source"]


def test_video_sidecar_schema_requires_the_exact_render_quad():
    import json

    from jsonschema import Draft7Validator

    schema_path = Path(__file__).resolve().parent.parent / "schemas/face_crop_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    crop = {
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "schema_version": "1.0",
        "parent_clip": {"uuid": "123e4567-e89b-12d3-a456-426614174001", "file": "clip.json"},
        "source_video": "clip.mp4",
        "track_id": 1,
        "duration_seconds": 1.0,
        "stabilized": True,
        "stabilization_window_seconds": 0.8,
        "stabilization_anchor_tolerance_px": 2.5,
        "frame_data": {"0": [{"source_frame_index": 0, "render_quad_source": _BASE.tolist()}]},
    }
    assert validator.is_valid(crop)
    del crop["frame_data"]["0"][0]["render_quad_source"]
    assert not validator.is_valid(crop)
    crop["frame_data"]["0"][0]["render_quad_source"] = _BASE.tolist()
    del crop["frame_data"]["0"][0]["source_frame_index"]
    assert not validator.is_valid(crop)


def test_two_pass_render_bounded_memory(tmp_path):
    import resource
    import sys

    n_frames = 300
    video = tmp_path / "big.mp4"
    _make_gradient_video(video, n_frames, 1280, 720)
    frame_data = {str(i): [_det(0, _BASE)] for i in range(n_frames)}
    scale = 1024**2 if sys.platform in ("darwin", "win32") else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    plan, stabs = face_stabilization.plan_stabilized_track_crops(
        frame_data, 0, CFG, n_frames, fps=25.0
    )
    rendered = face_stabilization.render_stabilized_track_frames(video, plan, stabs, n_frames)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert len(rendered[0]) == n_frames
    assert (after - before) / scale < 700


def test_corners_to_warp_black_pads_outside_source():
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    outside = np.array([[-10, -10], [-5, -10], [-5, -5], [-10, -5]], dtype=np.float32)
    out = face_geometry._corners_to_warp(frame, outside, 32)
    assert out.shape == (32, 32, 3)
    assert int(out.max()) == 0


def test_corners_to_warp_pads_only_out_of_frame_part():
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    straddling = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float32)
    out = face_geometry._corners_to_warp(frame, straddling, 64)
    assert int(out[0, 0].max()) == 0
    assert int(out[-1, -1].min()) == 255


def test_quad_overshoot_px_inside_and_outside():
    inside = np.array([[1, 1], [9, 1], [9, 9], [1, 9]], dtype=np.float32)
    assert face_geometry.quad_overshoot_px(inside, 10, 10) == 0.0
    outside = np.array([[-3, 1], [15, 1], [15, 9], [-3, 9]], dtype=np.float32)
    assert face_geometry.quad_overshoot_px(outside, 10, 10) == 5.0
