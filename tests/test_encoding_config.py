"""CPU-only tests for the encoding config + codec validation (issue #8).

``EncodingConfig.from_yaml_dict`` reads the ``encoding:`` section with
historical defaults; ``validate_video_codec`` fails loud when a configured
non-default codec is absent from the resolved ffmpeg (validated against a fake
``ffmpeg -encoders`` listing).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


encoding_config = _load_module("encoding_config_mod", "dardcollect/encoding_config.py")

from dardcollect.encoding_config import EncodingConfig


def test_defaults_reproduce_hardcoded_behavior():
    enc = encoding_config.EncodingConfig.from_yaml_dict({})
    assert enc.video_codec == "libx264"
    assert enc.audio_codec == "aac"
    assert enc.encoder_threads == 8


def test_reads_encoding_section():
    enc = encoding_config.EncodingConfig.from_yaml_dict(
        {"encoding": {"video_codec": "h264_nvenc", "audio_codec": "aac", "encoder_threads": 4}}
    )
    assert enc.video_codec == "h264_nvenc"
    assert enc.audio_codec == "aac"
    assert enc.encoder_threads == 4


def test_default_codec_skips_probe():
    """libx264 needs no probe — validate never touches ffmpeg."""
    encoding_config.validate_video_codec("libx264", None)  # must not raise


def _make_segment():
    from dardcollect.tracker import Segment

    return Segment(
        start_frame=0,
        end_frame=5,
        track_ids=[0],
        max_persons=1,
        face_visible_frames=6,
        max_consecutive_face_frames=6,
        mouth_open_frames=0,
        frame_data={},
    )


def _fake_ffmpeg_bat(tmp_path: Path, encoders_line: str) -> Path:
    """Fake ``ffmpeg -encoders`` printing a single encoders line.

    Portable across platforms: on Windows a `.bat` running python inline; on
    POSIX a `#!/bin/sh` wrapper around python3 with the exec bit set.
    """
    import sys

    inline = "import sys; sys.stdout.write(" + repr(encoders_line) + ")\n"
    if sys.platform == "win32":
        fake = tmp_path / "ffmpeg.bat"
        fake.write_text(
            '@echo off\r\npython -c "' + inline.replace('"', '\\"').replace("\n", " ") + '"\r\n',
            encoding="utf-8",
        )
    else:
        fake = tmp_path / "ffmpeg.sh"
        fake.write_text(
            "#!/bin/sh\nexec python3 -c " + repr(inline.strip()) + "\n", encoding="utf-8"
        )
        fake.chmod(0o755)
    return fake


def test_missing_codec_raises_with_remediation(tmp_path):
    fake_ffmpeg = _fake_ffmpeg_bat(tmp_path, " V....D libx264  |  V....D h264_mf ")
    with pytest.raises(RuntimeError) as excinfo:
        encoding_config.validate_video_codec("h264_nvenc", str(fake_ffmpeg))
    msg = str(excinfo.value)
    assert "h264_nvenc" in msg
    assert "FFMPEG_BINARY" in msg or "IMAGEIO_FFMPEG_EXE" in msg


def test_present_codec_validates(tmp_path):
    fake_ffmpeg = _fake_ffmpeg_bat(tmp_path, " V....D libx264  V....D h264_nvenc ")
    encoding_config.validate_video_codec("h264_nvenc", str(fake_ffmpeg))  # must not raise


def test_missing_ffmpeg_binary_raises():
    with pytest.raises(RuntimeError) as excinfo:
        encoding_config.validate_video_codec("h264_nvenc", None)
    assert "no ffmpeg binary" in str(excinfo.value)


def test_extract_clip_and_moviepy_accept_encoding():
    """Both encoder call sites accept an optional EncodingConfig (plumbing)."""
    import inspect

    import dardcollect.video_writers as vw

    assert "encoding" in list(inspect.signature(vw.extract_clip).parameters)
    assert "encoding" in list(inspect.signature(vw._write_video_with_moviepy).parameters)


def test_extraction_pipeline_calls_extract_clip_with_encoding(monkeypatch, tmp_path):
    """_extract_one_clip forwards the encoding config to extract_clip."""
    import dardcollect.clip_extraction as ce

    seg = _make_segment()
    seen = {}

    def fake_extract_clip(read_path, clip_path, s, e, fps, encoding=None):
        seen["encoding"] = encoding
        return True

    monkeypatch.setattr(ce, "extract_clip", fake_extract_clip)

    marker = EncodingConfig(video_codec="h264_nvenc")  # any instance satisfies the type
    ce._extract_one_clip(
        seg,
        ce.ClipBatchContext(
            read_path=tmp_path / "src.mp4",
            output_dir=tmp_path,
            fps=24.0,
            video_path=tmp_path / "src.mp4",
            video_info={},
            archive_org_id=None,
            archive_org_url=None,
            encoding=marker,
        ),
    )
    assert seen.get("encoding") is marker


def test_config_plumbs_encoding_to_extract_clips_serial(tmp_path, monkeypatch):
    """extract_clips serial path forwards encoding through to extract_clip."""
    import dardcollect.clip_extraction as ce
    from dardcollect.config import ClipExtractionConfig

    seen = {}

    def fake_extract_clip(read_path, clip_path, s, e, fps, encoding=None):
        seen["encoding"] = encoding
        return True

    monkeypatch.setattr(ce, "extract_clip", fake_extract_clip)

    clip_config = ClipExtractionConfig(
        input_dir="in",
        output_clips_dir=str(tmp_path),
        min_clip_duration_seconds=1.0,
        max_clip_duration_seconds=60.0,
        min_consecutive_frames=5,
        merge_gap_frames=12,
        require_face_visibility=False,
        min_face_size_percent=1.0,
        min_face_visible_frames=3,
    )
    marker = EncodingConfig(video_codec="h264_nvenc")  # any instance satisfies the type
    ce.extract_clips(
        [_make_segment()],
        ce.ClipBatchContext(
            read_path=tmp_path / "src.mp4",
            output_dir=tmp_path,
            fps=24.0,
            video_path=tmp_path / "src.mp4",
            video_info={},
            archive_org_id=None,
            archive_org_url=None,
            encoding=marker,
        ),
        clip_config,
    )
    assert seen.get("encoding") is marker
