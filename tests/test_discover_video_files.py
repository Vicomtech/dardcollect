"""Regression test for case-insensitive source-video discovery.

Pins the fix for the bug where ``extract_person_clips_from_videos.py`` globbed
source films with per-extension lowercase patterns (``rglob("*.mp4")``), which
is case-sensitive on Linux and silently skipped uppercase ``.MP4`` downloads
from Archive.org. Reverting ``discover_video_files`` to a lowercase-only glob
makes ``test_uppercase_extensions_are_found`` fail.
"""

from pathlib import Path

from dardcollect.pipeline_utils import discover_video_files


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")
    return path


def test_uppercase_extensions_are_found(tmp_path: Path) -> None:
    lower = _touch(tmp_path / "lang_en" / "film_a.mp4")
    upper = _touch(tmp_path / "lang_es" / "film_b.MP4")
    mixed = _touch(tmp_path / "film_c.Mp4")

    found = discover_video_files(tmp_path)

    assert set(found) == {lower, upper, mixed}


def test_non_video_files_are_ignored(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.json")
    _touch(tmp_path / "poster.jpg")
    video = _touch(tmp_path / "clip.mkv")

    assert discover_video_files(tmp_path) == [video]


def test_all_supported_extensions_any_case(tmp_path: Path) -> None:
    names = ["a.mp4", "b.AVI", "c.mkv", "d.MOV", "e.webm", "f.M4V"]
    expected = {_touch(tmp_path / n) for n in names}

    assert set(discover_video_files(tmp_path)) == expected


def test_results_are_sorted(tmp_path: Path) -> None:
    _touch(tmp_path / "z.mp4")
    _touch(tmp_path / "a.MP4")
    _touch(tmp_path / "m.mkv")

    found = discover_video_files(tmp_path)

    assert found == sorted(found)


def test_single_file_input(tmp_path: Path) -> None:
    video = _touch(tmp_path / "solo.MP4")
    assert discover_video_files(video) == [video]

    other = _touch(tmp_path / "doc.pdf")
    assert discover_video_files(other) == []
