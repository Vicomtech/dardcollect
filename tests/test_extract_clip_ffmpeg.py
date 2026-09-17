"""Tests for ffmpeg-direct clip extraction (dardcollect.pipeline_utils.extract_clip).

The clip↔source alignment contract is the point of these tests: downstream stages
map a clip's frames back to source detections by position
(``abs_frame = start_frame + frame_id``), so extract_clip MUST start exactly at
``start_frame`` and emit exactly ``end_frame - start_frame + 1`` frames. A single
off-by-one would misalign the whole ``frame_data`` mapping. Each source frame here
is filled with a gray level encoding its index, so the extracted clip's first/last
frames can be checked against the expected source indices deterministically.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dardcollect.pipeline_utils import extract_clip

_FPS = 25.0
_N_SRC = 80
_W, _H = 160, 120


def _gray_for(idx: int) -> int:
    """Distinct, well-separated gray level per frame index (survives H.264)."""
    return (idx * 3) % 256


def _make_source(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_W, _H))
    assert writer.isOpened(), "could not open VideoWriter"
    for i in range(_N_SRC):
        frame = np.full((_H, _W, 3), _gray_for(i), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    src = tmp_path / "source.mp4"
    _make_source(src)
    assert src.exists() and src.stat().st_size > 0
    return src


def test_exact_frame_count(source_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    ok = extract_clip(source_video, out, start_frame=10, end_frame=29, fps=_FPS)
    assert ok
    frames = _read_frames(out)
    # end - start + 1 = 20 frames, inclusive of both endpoints.
    assert len(frames) == 20


def test_alignment_first_and_last_frame(source_video: Path, tmp_path: Path) -> None:
    start, end = 30, 44
    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=start, end_frame=end, fps=_FPS)

    frames = _read_frames(out)
    assert len(frames) == end - start + 1

    first_mean = float(frames[0].mean())
    last_mean = float(frames[-1].mean())
    # The clip's frame 0 must be the SOURCE frame `start`, not start±1.
    assert abs(first_mean - _gray_for(start)) <= 6, (
        f"first frame gray {first_mean:.1f} != source frame {start} ({_gray_for(start)})"
    )
    assert abs(last_mean - _gray_for(end)) <= 6, (
        f"last frame gray {last_mean:.1f} != source frame {end} ({_gray_for(end)})"
    )


def test_single_frame_clip(source_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "one.mp4"
    assert extract_clip(source_video, out, start_frame=5, end_frame=5, fps=_FPS)
    assert len(_read_frames(out)) == 1


def test_atomic_no_partial_left_on_success(source_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=0, end_frame=9, fps=_FPS)
    assert out.exists()
    # The .partial temp must have been renamed away, never left behind.
    assert not out.with_name(out.name + ".partial").exists()


def test_missing_source_returns_false_no_output(tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    ok = extract_clip(tmp_path / "does_not_exist.mp4", out, start_frame=0, end_frame=9, fps=_FPS)
    assert ok is False
    assert not out.exists()
    assert not out.with_name(out.name + ".partial").exists()


def test_invalid_fps_returns_false(source_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=0, end_frame=9, fps=0.0) is False
    assert not out.exists()


def test_empty_range_returns_false(source_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=20, end_frame=10, fps=_FPS) is False
    assert not out.exists()


# ── FFmpeg binary resolution (2026-09-16 fix) ───────────────────────────────
# extract_clip must encode with the SAME binary the stage validated at startup
# (dardcollect.archive._ffmpeg_exe: FFMPEG_BINARY → IMAGEIO_FFMPEG_EXE →
# imageio bundle). The old code validated with the env-aware resolver but
# encoded with the imageio bundle only, so a NVENC config passed validation
# and then failed per-clip with "Unknown encoder 'h264_nvenc'".


def _fake_ffmpeg_run(calls: list[list[str]]):
    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp4")

    return fake_run


def test_extract_clip_uses_validated_ffmpeg_binary(
    source_video: Path, tmp_path: Path, monkeypatch
) -> None:
    import dardcollect.video_writers as video_writers

    fake_bin = tmp_path / "validated-ffmpeg"
    fake_bin.write_text("x", encoding="utf-8")
    monkeypatch.setenv("FFMPEG_BINARY", str(fake_bin))

    calls: list[list[str]] = []
    monkeypatch.setattr(video_writers.subprocess, "run", _fake_ffmpeg_run(calls))

    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=0, end_frame=9, fps=_FPS)
    assert calls[0][0] == str(fake_bin)


def test_extract_clip_default_binary_is_imageio_bundle(
    source_video: Path, tmp_path: Path, monkeypatch
) -> None:
    import imageio_ffmpeg

    import dardcollect.video_writers as video_writers

    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)

    calls: list[list[str]] = []
    monkeypatch.setattr(video_writers.subprocess, "run", _fake_ffmpeg_run(calls))

    out = tmp_path / "clip.mp4"
    assert extract_clip(source_video, out, start_frame=0, end_frame=9, fps=_FPS)
    assert calls[0][0] == imageio_ffmpeg.get_ffmpeg_exe()


def test_moviepy_write_points_at_validated_ffmpeg_binary(tmp_path: Path, monkeypatch) -> None:
    """_write_video_with_moviepy must route moviepy's internal imageio-ffmpeg
    call at the validated binary too (moviepy honors IMAGEIO_FFMPEG_EXE)."""
    import os

    import moviepy.video.io.ImageSequenceClip as isc

    import dardcollect.video_writers as video_writers

    fake_bin = tmp_path / "validated-ffmpeg"
    fake_bin.write_text("x", encoding="utf-8")
    monkeypatch.setenv("FFMPEG_BINARY", str(fake_bin))

    seen: dict[str, str | None] = {}

    class FakeClip:
        def __init__(self, frames, durations=None):
            pass

        def write_videofile(self, path, **kwargs):
            seen["exe"] = os.environ.get("IMAGEIO_FFMPEG_EXE")
            Path(path).write_bytes(b"fake-mp4")

    monkeypatch.setattr(isc, "ImageSequenceClip", FakeClip)

    frames = [np.full((8, 8, 3), 128, dtype=np.uint8)]
    assert video_writers._write_video_with_moviepy(frames, tmp_path / "out.mp4", _FPS)
    assert seen["exe"] == str(fake_bin)
