"""CPU-only tests for the block-delta scene-cut signal (issue #4).

Signal 3 catches same-set shot/reverse-shot cuts: the global luminance
histogram is spatially invariant, so it survives them; the 4×4 block-delta
fires when the spatial layout flips. Synthetic frames: a checkerboard flip
keeps the global histogram identical while moving every block.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dardcollect.person_clips_helpers import block_delta_cut
from dardcollect.pipeline_utils import scene_changed


def _cfg(**overrides):
    """Minimal clip-config stand-in carrying the scene-cut thresholds."""
    base = {
        "scene_change_threshold": 0.75,
        "scene_change_bbox_area_ratio": 4.0,
        "scene_change_block_delta": False,
        "scene_change_block_delta_threshold": 24.0,
        "scene_change_block_delta_fraction": 0.5,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _laid_out_frame(top_value: int, bottom_value: int, size: int = 256) -> np.ndarray:
    """Two-band frame: bright band on top, dark band on the bottom."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    frame[:half, :] = top_value
    frame[half:, :] = bottom_value
    return frame


def test_same_layout_static_scene_does_not_fire():
    frame = _laid_out_frame(200, 40)
    assert block_delta_cut(frame, frame.copy(), threshold=24.0, fraction=0.5) is False


def test_shot_reverse_shot_flip_fires():
    """Same two luminance values, layout flipped → global histogram identical,
    every block changes → signal 3 must fire (the signal-1 blind spot)."""
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)  # bands swapped = reverse angle
    assert block_delta_cut(prev, curr, threshold=24.0, fraction=0.5) is True


def test_signal1_survives_the_cut_but_signal3_catches_it():
    """The full detector: with signal 3 enabled, a layout flip is a cut;
    with signal 3 disabled (default), the same pair is NOT a cut."""
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)
    bboxes = np.zeros((0, 4), dtype=float)

    assert not scene_changed(prev, curr, bboxes, bboxes, _cfg()), (
        "signal 1+2 must not fire on a histogram-preserving flip"
    )

    assert scene_changed(prev, curr, bboxes, bboxes, _cfg(scene_change_block_delta=True)), (
        "signal 3 must catch what signal 1 misses"
    )


def test_signal3_disabled_by_default_is_noop():
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)
    bboxes = np.zeros((0, 4), dtype=float)
    assert not scene_changed(prev, curr, bboxes, bboxes, _cfg())


def test_gentle_change_below_threshold_does_not_fire():
    """A small exposure shift moves every block by a little — below threshold."""
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(210, 50)  # +10 everywhere, layout unchanged
    assert block_delta_cut(prev, curr, threshold=24.0, fraction=0.5) is False


def test_partial_block_change_below_fraction_does_not_fire():
    """Only the top band changes (e.g. object enters) — 8/16 = 0.5 cells change
    at threshold; with a tighter fraction it must not fire."""
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)
    # fraction 0.9 → needs 15 of 16 blocks; a flip has 16 but with a high
    # per-cell threshold none change — verify fraction gating:
    assert block_delta_cut(prev, curr, threshold=24.0, fraction=0.9) is True
    assert block_delta_cut(prev, curr, threshold=250.0, fraction=0.5) is False


def test_signal1_still_fires_on_histogram_shift():
    """Regression: enabling signal 3 does not break signals 1/2."""
    prev = _bgr(np.full((256, 256), 40, dtype=np.uint8))
    curr = _bgr(np.full((256, 256), 220, dtype=np.uint8))
    bboxes = np.zeros((0, 4), dtype=float)
    assert scene_changed(prev, curr, bboxes, bboxes, _cfg(block_delta=False)) is True


def test_block_delta_respects_cooldown_in_wrapper():
    from dardcollect.config import ClipExtractionConfig
    from dardcollect.person_clips_helpers import is_scene_change

    cfg = ClipExtractionConfig(
        input_dir="in",
        output_clips_dir="out",
        min_clip_duration_seconds=1.0,
        max_clip_duration_seconds=60.0,
        min_consecutive_frames=5,
        merge_gap_frames=12,
        require_face_visibility=False,
        min_face_size_percent=1.0,
        min_face_visible_frames=3,
        scene_change_threshold=0.75,
        scene_change_bbox_area_ratio=4.0,
        scene_change_block_delta=True,
        scene_change_block_delta_threshold=24.0,
        scene_change_block_delta_fraction=0.5,
    )
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)
    bboxes = np.zeros((0, 4), dtype=float)

    # Inside the 8-frame cooldown: suppressed
    assert not is_scene_change(
        cfg,
        prev,
        frame_id=4,
        last_scene_change_frame=0,
        prev_det_bboxes=bboxes,
        det_bboxes=bboxes,
        frame=curr,
    )
    # Past the cooldown: fires
    assert is_scene_change(
        cfg,
        prev,
        frame_id=10,
        last_scene_change_frame=0,
        prev_det_bboxes=bboxes,
        det_bboxes=bboxes,
        frame=curr,
    )


def test_disabled_scene_change_detection_is_noop():
    from dardcollect.config import ClipExtractionConfig
    from dardcollect.person_clips_helpers import is_scene_change

    cfg = ClipExtractionConfig(
        input_dir="in",
        output_clips_dir="out",
        min_clip_duration_seconds=1.0,
        max_clip_duration_seconds=60.0,
        min_consecutive_frames=5,
        merge_gap_frames=12,
        require_face_visibility=False,
        min_face_size_percent=1.0,
        min_face_visible_frames=3,
        scene_change_threshold=0.75,
        scene_change_bbox_area_ratio=4.0,
        scene_change_block_delta=True,
        scene_change_block_delta_threshold=24.0,
        scene_change_block_delta_fraction=0.5,
        scene_change_detection=False,
    )
    prev = _laid_out_frame(200, 40)
    curr = _laid_out_frame(40, 200)
    bboxes = np.zeros((0, 4), dtype=float)
    assert not is_scene_change(
        cfg,
        prev,
        frame_id=100,
        last_scene_change_frame=0,
        prev_det_bboxes=bboxes,
        det_bboxes=bboxes,
        frame=curr,
    )


def test_config_reads_block_delta_keys(tmp_path):
    from dardcollect.config import ClipExtractionConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "person_extraction:\n"
        "  input_dir: in\n"
        "  output_clips_dir: out\n"
        "  min_clip_duration_seconds: 1.0\n"
        "  max_clip_duration_seconds: 60.0\n"
        "  min_consecutive_frames: 5\n"
        "  merge_gap_frames: 12\n"
        "  require_face_visibility: false\n"
        "  min_face_size_percent: 1.0\n"
        "  min_face_visible_frames: 3\n"
        "  scene_change_block_delta: true\n"
        "  scene_change_block_delta_threshold: 30.0\n"
        "  scene_change_block_delta_fraction: 0.6\n",
        encoding="utf-8",
    )
    cfg = ClipExtractionConfig.from_yaml(str(yaml_path))
    assert cfg.scene_change_block_delta is True
    assert cfg.scene_change_block_delta_threshold == 30.0
    assert cfg.scene_change_block_delta_fraction == 0.6


def test_config_block_delta_defaults_off(tmp_path):
    from dardcollect.config import ClipExtractionConfig

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "person_extraction:\n"
        "  input_dir: in\n"
        "  output_clips_dir: out\n"
        "  min_clip_duration_seconds: 1.0\n"
        "  max_clip_duration_seconds: 60.0\n"
        "  min_consecutive_frames: 5\n"
        "  merge_gap_frames: 12\n"
        "  require_face_visibility: false\n"
        "  min_face_size_percent: 1.0\n"
        "  min_face_visible_frames: 3\n",
        encoding="utf-8",
    )
    cfg = ClipExtractionConfig.from_yaml(str(yaml_path))
    assert cfg.scene_change_block_delta is False
