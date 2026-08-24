"""Tests for the reclaimed-source re-download helper (no network).

Covers find_missing_sources: it reconstructs the download stage's on-disk layout
(<media>/<language>/<filename>) from downloads.csv and returns only rows whose file
is genuinely absent, so a re-download restores exactly the reclaimed originals and
nothing that is still present.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.redownload_sources import _expected_path, find_missing_sources

_FIELDS = ["archive_org_identifier", "filename_downloaded", "media_type", "language"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_expected_path_language_aware(tmp_path: Path) -> None:
    p = _expected_path(tmp_path, "Film (1959).mp4", "eng", "video")
    assert p == tmp_path / "eng" / "Film (1959).mp4"


def test_expected_path_blank_language_falls_back_to_und(tmp_path: Path) -> None:
    p = _expected_path(tmp_path, "x.mp4", "", "video")
    assert p == tmp_path / "und" / "x.mp4"


def test_expected_path_image_not_language_aware(tmp_path: Path) -> None:
    p = _expected_path(tmp_path, "pic.jpg", "eng", "image")
    assert p == tmp_path / "pic.jpg"


def test_missing_when_file_absent(tmp_path: Path) -> None:
    media = tmp_path / "videos"
    csvp = tmp_path / "downloads.csv"
    _write_csv(
        csvp,
        [
            {
                "archive_org_identifier": "id-a",
                "filename_downloaded": "a.mp4",
                "media_type": "video",
                "language": "eng",
            },
        ],
    )
    missing = find_missing_sources(media, csvp, "video")
    assert len(missing) == 1
    assert missing[0][0] == "id-a"
    assert missing[0][2] == media / "eng" / "a.mp4"


def test_present_file_is_not_missing(tmp_path: Path) -> None:
    media = tmp_path / "videos"
    (media / "eng").mkdir(parents=True)
    (media / "eng" / "a.mp4").write_bytes(b"\x00")
    csvp = tmp_path / "downloads.csv"
    _write_csv(
        csvp,
        [
            {
                "archive_org_identifier": "id-a",
                "filename_downloaded": "a.mp4",
                "media_type": "video",
                "language": "eng",
            }
        ],
    )
    assert find_missing_sources(media, csvp, "video") == []


def test_present_under_other_language_folder_is_not_missing(tmp_path: Path) -> None:
    # File exists but under a different language folder than recorded — still present.
    media = tmp_path / "videos"
    (media / "spa").mkdir(parents=True)
    (media / "spa" / "a.mp4").write_bytes(b"\x00")
    csvp = tmp_path / "downloads.csv"
    _write_csv(
        csvp,
        [
            {
                "archive_org_identifier": "id-a",
                "filename_downloaded": "a.mp4",
                "media_type": "video",
                "language": "eng",
            }
        ],
    )
    assert find_missing_sources(media, csvp, "video") == []


def test_only_requested_media_type(tmp_path: Path) -> None:
    media = tmp_path / "videos"
    csvp = tmp_path / "downloads.csv"
    _write_csv(
        csvp,
        [
            {
                "archive_org_identifier": "vid",
                "filename_downloaded": "v.mp4",
                "media_type": "video",
                "language": "eng",
            },
            {
                "archive_org_identifier": "aud",
                "filename_downloaded": "a.mp3",
                "media_type": "audio",
                "language": "eng",
            },
        ],
    )
    missing = find_missing_sources(media, csvp, "video")
    assert [m[0] for m in missing] == ["vid"]


def test_deduplicates_repeated_rows(tmp_path: Path) -> None:
    media = tmp_path / "videos"
    csvp = tmp_path / "downloads.csv"
    _write_csv(
        csvp,
        [
            {
                "archive_org_identifier": "id-a",
                "filename_downloaded": "a.mp4",
                "media_type": "video",
                "language": "eng",
            },
            {
                "archive_org_identifier": "id-a",
                "filename_downloaded": "a.mp4",
                "media_type": "video",
                "language": "eng",
            },
        ],
    )
    assert len(find_missing_sources(media, csvp, "video")) == 1
