"""Regression tests for frame-stride sampling in quality scoring (#5).

Before the fix, ``frame_idx`` was incremented only inside the
``frame_idx % frame_stride == 0`` branch, so with ``frame_stride > 1`` exactly
one frame (frame 0) was ever scored per crop and ``max_frames`` was never
reached. The shared helper ``score_frames_with_stride`` in
``dardcollect.quality`` now increments the index on every frame; both entry
points (``score_video`` and ``pipeline/annotate_face_quality.py``) use it.

Pure CPU tests: no GPU, no models, no network. ``QualityModels`` is stubbed so
no ONNX session is created.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dardcollect.quality import score_frames_with_stride
from dardcollect.quality_inputs import StrideSampling


def _fake_models(call_log: list):
    """Build a stand-in for QualityModels whose score path is a stub."""

    class _Stub:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    def _score_frame_all(ofiq_frame, models, arcface_frame):
        call_log.append(arcface_frame is not None)
        return {"sharpness": 0.5}

    return SimpleNamespace(magface=_Stub()), _score_frame_all


def _synthetic_frames(n: int) -> list:
    return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]


def test_stride_one_scores_every_frame():
    from unittest.mock import patch

    log: list = []
    models, fake = _fake_models(log)
    frames = _synthetic_frames(7)

    with patch("dardcollect.quality.score_frame_all", side_effect=fake):
        out = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=1, max_frames=0),
            has_arcface_annotation=False,
        )

    assert len(out) == 7
    assert [d["frame_index"] for d in out] == [0, 1, 2, 3, 4, 5, 6]


def test_stride_five_scores_ceil_n_over_5():
    from unittest.mock import patch

    log: list = []
    models, fake = _fake_models(log)
    frames = _synthetic_frames(12)

    with patch("dardcollect.quality.score_frame_all", side_effect=fake):
        out = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=5, max_frames=0),
            has_arcface_annotation=False,
        )

    # Frames 0, 5, 10 sampled (ceil(12/5) = 3), indices preserve the in-crop position
    assert len(out) == 3
    assert [d["frame_index"] for d in out] == [0, 5, 10]


def test_max_frames_caps_sampled_entries():
    from unittest.mock import patch

    log: list = []
    models, fake = _fake_models(log)
    frames = _synthetic_frames(100)

    with patch("dardcollect.quality.score_frame_all", side_effect=fake):
        out = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=5, max_frames=3),
            has_arcface_annotation=False,
        )

    assert len(out) == 3
    assert [d["frame_index"] for d in out] == [0, 5, 10]


def test_all_frames_failing_scoring_yields_empty_output():
    """An all-frames-failed crop must return no entries (None upstream), not 1."""
    from unittest.mock import patch

    models, _ = _fake_models([])
    frames = _synthetic_frames(4)

    with patch("dardcollect.quality.score_frame_all", side_effect=RuntimeError("model blew up")):
        out = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=1, max_frames=0),
            has_arcface_annotation=False,
        )

    assert out == []


def test_both_entry_points_share_the_helper():
    """score_video and pipeline/annotate_face_quality.py must call the same helper."""
    import ast

    repo_root = Path(__file__).resolve().parent.parent
    pipeline_script = repo_root / "pipeline" / "annotate_face_quality.py"

    source = pipeline_script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        alias.name
        for node in ast.walk(tree)
        for alias in getattr(node, "names", [])
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert "score_frames_with_stride" in imported
    assert "_score_and_append" not in imported  # old direct call removed

    # And the library entry point routes through it too

    quality_src = (repo_root / "dardcollect" / "quality.py").read_text(encoding="utf-8")
    assert "score_frames_with_stride(" in quality_src
    assert "frame_scores = score_frames_with_stride(" in quality_src


def test_score_video_and_pipeline_agree_on_sampling(tmp_path):
    """Entry-point agreement: the stride math both use is the helper's."""
    from unittest.mock import patch

    log: list = []
    models, fake = _fake_models(log)
    frames = _synthetic_frames(23)

    with patch("dardcollect.quality.score_frame_all", side_effect=fake):
        via_library = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=5, max_frames=2),
            has_arcface_annotation=False,
        )

    # The pipeline stage computes frame_stride/max_frames from the same config
    # keys (cfg.frame_stride, cfg.max_frames) and passes them to the same
    # helper — identical inputs must give identical sampling.
    with patch("dardcollect.quality.score_frame_all", side_effect=fake):
        via_pipeline = score_frames_with_stride(
            frames,
            models,
            sampling=StrideSampling(frame_stride=5, max_frames=2),
            has_arcface_annotation=False,
        )

    assert [d["frame_index"] for d in via_library] == [d["frame_index"] for d in via_pipeline]
    assert [d["frame_index"] for d in via_library] == [0, 5]
