"""CPU-only tests for the content-based colour filter (issue #11).

Synthetic videos: a saturated-colour clip and a pure grayscale clip, written
with OpenCV's VideoWriter (mp4v), classify correctly. CSV resumability: a
rerun skips already-classified videos.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "color_filter",
        Path(__file__).resolve().parent.parent / "pipeline" / "filter_videos_by_color.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load pipeline/filter_videos_by_color.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["color_filter"] = module
    spec.loader.exec_module(module)
    return module


color_filter = _load_module()


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    """Write frames to an .mp4 with OpenCV (mp4v codec)."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def _saturated_frames(n: int = 6, size: int = 64) -> list[np.ndarray]:
    """Highly saturated frames (pure red/blue bands)."""
    frames = []
    for i in range(n):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, : size // 2] = (0, 0, 255)  # red in BGR
        frame[:, size // 2 :] = (255, 0, 0)  # blue in BGR
        frames.append(frame)
    return frames


def _gray_frames(n: int = 6, size: int = 64) -> list[np.ndarray]:
    """Grayscale-valued frames (R==G==B → zero HSV saturation)."""
    frames = []
    for i in range(n):
        val = 60 + i * 20
        gray = np.full((size, size, 3), val, dtype=np.uint8)
        frames.append(gray)
    return frames


@pytest.fixture()
def _no_ffmpeg_fast_path(monkeypatch):
    """Force the OpenCV sampling path (deterministic, no real ffmpeg output)."""
    monkeypatch.setattr(color_filter, "_sample_frames_ffmpeg_fast", lambda *a, **k: [])


def test_saturated_video_classifies_color(tmp_path, _no_ffmpeg_fast_path):
    video = tmp_path / "color_movie.mp4"
    _write_video(video, _saturated_frames())
    classification, mean_sat = color_filter.video_color_score(video, n_frames=4)
    assert classification == "color"
    assert mean_sat > 50


def test_gray_video_classifies_black_and_white(tmp_path, _no_ffmpeg_fast_path):
    video = tmp_path / "bw_movie.mp4"
    _write_video(video, _gray_frames())
    classification, mean_sat = color_filter.video_color_score(video, n_frames=4)
    assert classification == "black_and_white"
    # mp4v lossy compression injects small chroma noise into R==G==B frames;
    # the BW threshold (12.0) must sit above that floor but below real colour.
    assert 0.0 <= mean_sat < color_filter.BW_SATURATION_THRESHOLD


def test_unreadable_video_reports_unreadable(tmp_path, _no_ffmpeg_fast_path, caplog):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video")
    classification, mean_sat = color_filter.video_color_score(video, n_frames=4)
    assert classification == "unreadable"
    assert mean_sat == 0.0


def test_classify_video_writes_csv_row_and_is_resumable(tmp_path, _no_ffmpeg_fast_path):
    video = tmp_path / "bw_movie.mp4"
    _write_video(video, _gray_frames())
    csv_path = tmp_path / color_filter.CSV_NAME

    row = color_filter.classify_video(video, csv_path, n_frames=4, do_move=False)
    assert row["classification"] == "black_and_white"
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1

    # Rerun: resumable skip — no new row, same classification returned
    row2 = color_filter.classify_video(video, csv_path, n_frames=4, do_move=False)
    assert row2 is rows[0] or row2["uuid"] == rows[0]["uuid"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_move_relocates_bw_videos_and_records_row(tmp_path, _no_ffmpeg_fast_path):
    bw = tmp_path / "bw_movie.mp4"
    color = tmp_path / "color_movie.mp4"
    _write_video(bw, _gray_frames())
    _write_video(color, _saturated_frames())
    csv_path = tmp_path / color_filter.CSV_NAME

    color_filter.classify_video(bw, csv_path, n_frames=4, do_move=True)
    color_filter.classify_video(color, csv_path, n_frames=4, do_move=True)

    assert not bw.exists()
    assert (tmp_path / "black_and_white" / "bw_movie.mp4").exists()
    assert color.exists()  # colour videos stay put

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = {r["video_name"]: r for r in csv.DictReader(f)}
    assert rows["bw_movie"]["moved"] == "true"
    assert rows["bw_movie"]["moved_to"]
    assert rows["color_movie"]["moved"] == ""


def test_move_is_reversible_from_recorded_rows(tmp_path, _no_ffmpeg_fast_path):
    """The CSV records source/destination — an undo replays the recorded move."""
    video = tmp_path / "bw_movie.mp4"
    _write_video(video, _gray_frames())
    csv_path = tmp_path / color_filter.CSV_NAME

    color_filter.classify_video(video, csv_path, n_frames=4, do_move=True)
    dest = tmp_path / "black_and_white" / "bw_movie.mp4"
    assert dest.exists() and not video.exists()

    # Undo from recorded rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["moved"] == "true":
                __import__("shutil").move(r["moved_to"], str(tmp_path / r["video_path"]))
    assert video.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # no extra rows written by the undo


def test_move_collision_never_overwrites(tmp_path, _no_ffmpeg_fast_path, caplog):
    import logging

    video = tmp_path / "bw_movie.mp4"
    _write_video(video, _gray_frames())
    dest_dir = tmp_path / "black_and_white"
    dest_dir.mkdir()
    (dest_dir / "bw_movie.mp4").write_bytes(b"existing")
    csv_path = tmp_path / color_filter.CSV_NAME

    with caplog.at_level(logging.WARNING, logger="color_filter"):
        row = color_filter.classify_video(video, csv_path, n_frames=4, do_move=True)
    assert row["moved"] == ""
    assert video.exists()  # left in place
    assert (dest_dir / "bw_movie.mp4").read_bytes() == b"existing"


def test_rerun_skips_already_classified(tmp_path, _no_ffmpeg_fast_path):
    """main() resumability: a video with an existing CSV row is not re-scored."""
    video = tmp_path / "bw_movie.mp4"
    _write_video(video, _gray_frames())
    csv_path = tmp_path / color_filter.CSV_NAME
    color_filter.classify_video(video, csv_path, n_frames=4, do_move=False)

    monkey_spy = pytest.MonkeyPatch()
    monkey_spy.setattr(
        color_filter,
        "video_color_score",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-score")),
    )
    try:
        row = color_filter.classify_video(video, csv_path, n_frames=4, do_move=False)
        assert row["classification"] == "black_and_white"
    finally:
        monkey_spy.undo()
