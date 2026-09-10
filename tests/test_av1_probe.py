"""CPU-only tests for AV1 codec probing + policy routing (issue #10).

``probe_video_codec`` / ``_apply_av1_policy`` in ``dardcollect.archive`` route
downloaded videos by codec. Tests stub the ffmpeg/ffprobe binaries (fake
scripts) so no real ffmpeg is needed and the fallback path (no ffprobe →
``ffmpeg -i`` stderr parse) is exercised deterministically.
"""

from __future__ import annotations

import json
import logging

import pytest

from dardcollect.archive import _apply_av1_policy, probe_video_codec


@pytest.fixture()
def fake_ffmpeg_env(tmp_path, monkeypatch):
    """Create fake ffmpeg/ffprobe executables that emit controllable output.

    The fake tools read a JSON instruction file pointed at by _FAKE_CTRL to
    decide their stdout/stderr, so each test selects codec output without
    patching subprocess internals.
    """
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    instruction = tmp_path / "instruction.json"

    # No real binary is needed: on Windows a `.bat` is directly executable via
    # CreateProcess, on POSIX a `#!/bin/sh` script with the exec bit works. The
    # fakes emit the fields of the JSON instruction file via python (present on
    # both platforms and in both venvs).
    suffix = ".bat" if sys.platform == "win32" else ".sh"
    ffprobe = bin_dir / f"ffprobe{suffix}"
    ffmpeg = bin_dir / f"ffmpeg{suffix}"

    body = (
        "import json, os, sys\n"
        'i = json.load(open(os.environ["_FAKE_INSTRUCTION"]))\n'
        "out = i.get('ffprobe_stdout', '')\n"
        "err = i.get('ffprobe_stderr', '')\n"
        "sys.stdout.write(out)\n"
        "sys.stderr.write(err)\n"
    )
    ffprobe.write_text(_fake_tool_text(suffix, body), encoding="utf-8")

    ffmpeg_body = (
        "import json, os, sys\n"
        'i = json.load(open(os.environ["_FAKE_INSTRUCTION"]))\n'
        "sys.stderr.write(i.get('ffmpeg_stderr', ''))\n"
    )
    ffmpeg.write_text(_fake_tool_text(suffix, ffmpeg_body), encoding="utf-8")
    if sys.platform != "win32":
        ffprobe.chmod(0o755)
        ffmpeg.chmod(0o755)

    import dardcollect.archive as archive_mod

    instruction.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(archive_mod, "_ffmpeg_exe", lambda: str(ffmpeg))
    monkeypatch.setenv("_FAKE_INSTRUCTION", str(instruction))

    def set_behavior(
        *, ffprobe_stdout: str = "", ffmpeg_stderr: str = "", drop_ffprobe: bool = False
    ):
        if drop_ffprobe:
            ffprobe.unlink()
        instruction.write_text(
            json.dumps({"ffprobe_stdout": ffprobe_stdout, "ffmpeg_stderr": ffmpeg_stderr}),
            encoding="utf-8",
        )

    return set_behavior


def _fake_tool_text(suffix: str, body: str) -> str:
    """Shebang wrapper for POSIX; `@echo off`-free bat that runs python inline."""
    if suffix == ".bat":
        return (
            "@echo off\r\n"
            'python -c "'
            + body.replace('"', '\\"').replace("\r\n", "; ").replace("\n", "; ")
            + '"\r\n'
        )
    return "#!/bin/sh\nexec python3 - <<'PYEOF'\n" + body + "\nPYEOF\n"


def test_probe_av1_via_ffprobe(tmp_path, fake_ffmpeg_env):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="av1\n")
    assert probe_video_codec(video) == "av1"


def test_probe_h264_via_ffprobe(fake_ffmpeg_env, tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="h264\n")
    assert probe_video_codec(video) == "h264"


def test_probe_falls_back_to_ffmpeg_parse_when_no_ffprobe(tmp_path, fake_ffmpeg_env, caplog):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(
        drop_ffprobe=True, ffmpeg_stderr="Input #0 ...\n  Stream #0:0: Video: av1 (Main), yuv420p\n"
    )
    with caplog.at_level(logging.INFO):
        assert probe_video_codec(video) == "av1"
    assert "ffmpeg-parse fallback" in caplog.text  # fallback path is logged (policy rule)


def test_probe_no_ffmpeg_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    import dardcollect.archive as archive_mod

    monkeypatch.setattr(archive_mod, "_ffmpeg_exe", lambda: None)
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    assert probe_video_codec(video) is None


def test_warn_policy_keeps_file_and_logs(tmp_path, fake_ffmpeg_env, caplog):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="av1\n")
    with caplog.at_level(logging.WARNING):
        keep = _apply_av1_policy(video, "warn")
    assert keep is True
    assert video.exists()
    assert "AV1 codec detected" in caplog.text


def test_skip_policy_deletes_file(tmp_path, fake_ffmpeg_env, caplog):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="av1\n")
    with caplog.at_level(logging.WARNING):
        keep = _apply_av1_policy(video, "skip")
    assert keep is False
    assert not video.exists()
    assert "av1_policy=skip" in caplog.text


def test_non_av1_codec_is_untouched(tmp_path, fake_ffmpeg_env):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="h264\n")
    assert _apply_av1_policy(video, "skip") is True
    assert video.exists()


def test_failed_probe_keeps_file_with_warning(tmp_path, caplog):
    """Probe failure must not delete data — keep + loud warning (fallback policy)."""
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    with caplog.at_level(logging.WARNING):
        keep = _apply_av1_policy(video, "skip")
    assert keep is True
    assert video.exists()
    assert "Codec probe failed" in caplog.text


def test_unknown_policy_falls_back_to_warn(tmp_path, fake_ffmpeg_env, caplog):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    fake_ffmpeg_env(ffprobe_stdout="av1\n")
    with caplog.at_level(logging.WARNING):
        keep = _apply_av1_policy(video, "explode")
    assert keep is True
    assert video.exists()
    assert "Unknown av1_policy" in caplog.text
