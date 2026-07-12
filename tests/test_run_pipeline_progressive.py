"""Regression tests for progressive orchestrator edge cases.

These tests cover downstream stages that require generated input directories.
When dependencies finish without producing those directories, the downstream
stage must be marked as skipped (finished=True, failed=False) rather than fail.
"""

from pathlib import Path
from threading import Event, Lock

import pytest

from scripts import run_pipeline


def test_stage_worker_skips_when_deps_finished_and_no_inputs(monkeypatch, tmp_path):
    """Filter-like stage should skip cleanly if deps ended without input dirs."""
    dep_state = run_pipeline.StageState(
        alias="quality",
        script="annotate_face_quality",
        deps=[],
        started=True,
        finished=True,
    )
    stage_state = run_pipeline.StageState(
        alias="filter",
        script="filter_face_crops_by_quality",
        deps=["quality"],
    )

    states = {"quality": dep_state, "filter": stage_state}
    stop_event = Event()
    lock = Lock()
    missing_input = tmp_path / "does_not_exist" / "video_face_crops"

    called = {"runs": 0}

    def _unexpected_run(*args, **kwargs):
        called["runs"] += 1
        raise AssertionError("stage execution should not be attempted")

    monkeypatch.setattr(run_pipeline, "_run_stage_once", _unexpected_run)

    run_pipeline._stage_worker(
        state=stage_state,
        states=states,
        py="python",
        child_env=None,
        rerun_interval_s=1,
        input_waits={"filter": [missing_input]},
        lock=lock,
        stop_event=stop_event,
    )

    assert called["runs"] == 0
    assert stage_state.finished is True
    assert stage_state.failed is False


def test_stage_worker_skips_when_input_dir_exists_but_is_empty(monkeypatch, tmp_path):
    """Clips-like stage should skip when dep finished but media files are absent."""
    dep_state = run_pipeline.StageState(
        alias="download",
        script="download_media_from_archive",
        deps=[],
        started=True,
        finished=True,
    )
    stage_state = run_pipeline.StageState(
        alias="clips",
        script="extract_person_clips_from_videos",
        deps=["download"],
    )

    states = {"download": dep_state, "clips": stage_state}
    stop_event = Event()
    lock = Lock()
    empty_input_dir = tmp_path / "archive_org_public_domain" / "videos"
    empty_input_dir.mkdir(parents=True, exist_ok=True)

    called = {"runs": 0}

    def _unexpected_run(*args, **kwargs):
        called["runs"] += 1
        raise AssertionError("stage execution should not be attempted for empty input dir")

    monkeypatch.setattr(run_pipeline, "_run_stage_once", _unexpected_run)

    run_pipeline._stage_worker(
        state=stage_state,
        states=states,
        py="python",
        child_env=None,
        rerun_interval_s=1,
        input_waits={"clips": [empty_input_dir]},
        lock=lock,
        stop_event=stop_event,
    )

    assert called["runs"] == 0
    assert stage_state.finished is True
    assert stage_state.failed is False


def test_has_required_stage_inputs_requires_media_for_clips(tmp_path):
    """Clips readiness requires media files, not only directory existence."""
    input_dir = tmp_path / "videos"
    input_dir.mkdir(parents=True, exist_ok=True)
    assert not run_pipeline._has_required_stage_inputs("clips", [input_dir])

    (input_dir / "sample.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    assert run_pipeline._has_required_stage_inputs("clips", [input_dir])


def test_build_progressive_input_waits_reads_filter_inputs(tmp_path):
    """Input waits are resolved for all known downstream stages."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
person_extraction:
    input_dir: DARD/archive_org_public_domain/videos
    output_clips_dir: DARD/extracted_person_clips
image_extraction:
    input_dir: DARD/archive_org_public_domain/images
face_crop_extraction:
    input_dir: DARD/extracted_person_clips
image_face_crop_extraction:
    input_dir: DARD/archive_org_public_domain/images
transcription:
    person_clips_dir: DARD/extracted_person_clips
face_quality_filtering:
  input_dir: DARD/video_face_crops
image_face_quality_filtering:
  input_dir: DARD/image_face_crops
""".strip()
        + "\n",
        encoding="utf-8",
    )

    waits = run_pipeline._build_progressive_input_waits(cfg)

    assert "clips" in waits
    assert "images" in waits
    assert "audio_clips" in waits
    assert "face_crops_video" in waits
    assert "face_crops_image" in waits
    assert "transcribe_video" in waits
    assert "filter" in waits

    assert waits["clips"][0].as_posix().endswith("DARD/archive_org_public_domain/videos")
    assert waits["images"][0].as_posix().endswith("DARD/archive_org_public_domain/images")
    assert waits["audio_clips"][0].as_posix().endswith("DARD/extracted_person_clips")
    assert waits["face_crops_video"][0].as_posix().endswith("DARD/extracted_person_clips")
    assert waits["face_crops_image"][0].as_posix().endswith("DARD/archive_org_public_domain/images")
    assert waits["transcribe_video"][0].as_posix().endswith("DARD/extracted_person_clips")
    assert len(waits["filter"]) == 2
    assert all(isinstance(p, Path) for p in waits["filter"])
    assert waits["filter"][0].as_posix().endswith("DARD/video_face_crops")
    assert waits["filter"][1].as_posix().endswith("DARD/image_face_crops")


def test_stage_enabled_for_media_disables_audio_in_video_only():
    """Video-only workflows must not run standalone audio transcription stage."""
    assert run_pipeline._stage_enabled_for_media("transcribe_audio", {"video"}) is False
    assert run_pipeline._stage_enabled_for_media("transcribe_audio", {"audio"}) is True
    assert run_pipeline._stage_enabled_for_media("docs", {"video"}) is False
    assert run_pipeline._stage_enabled_for_media("quality", {"video"}) is True


@pytest.mark.parametrize(
    ("media_types", "expected"),
    [
        (
            {"video"},
            [
                "clips",
                "audio_clips",
                "face_crops_video",
                "transcribe_video",
                "quality",
                "filter",
                "masks",
            ],
        ),
        ({"image"}, ["images", "face_crops_image", "quality", "filter", "masks"]),
        ({"audio"}, ["transcribe_audio"]),
        ({"text"}, ["docs"]),
        (
            {"video", "image"},
            [
                "clips",
                "audio_clips",
                "images",
                "face_crops_video",
                "face_crops_image",
                "transcribe_video",
                "quality",
                "filter",
                "masks",
            ],
        ),
        ({"audio", "text"}, ["transcribe_audio", "docs"]),
    ],
)
def test_stage_matrix_for_media_types(media_types, expected):
    """Orchestrator stage activation must match enabled modality combinations."""
    actual = [
        a for a, _ in run_pipeline.STAGES if run_pipeline._stage_enabled_for_media(a, media_types)
    ]
    assert actual == expected


def test_load_media_types_is_case_insensitive_and_filters_unknowns(tmp_path):
    """media_types parser accepts case variants and ignores unknown values."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("media_types: ['VIDEO', 'Audio', 'bogus']\n", encoding="utf-8")
    assert run_pipeline._load_media_types(cfg) == {"video", "audio"}


def test_load_media_types_defaults_to_video_when_empty_or_invalid(tmp_path):
    """Invalid media_types config should fall back to video-only behavior."""
    cfg_empty = tmp_path / "config-empty.yaml"
    cfg_empty.write_text("media_types: []\n", encoding="utf-8")
    assert run_pipeline._load_media_types(cfg_empty) == {"video"}

    cfg_invalid = tmp_path / "config-invalid.yaml"
    cfg_invalid.write_text("media_types: 'video'\n", encoding="utf-8")
    assert run_pipeline._load_media_types(cfg_invalid) == {"video"}
