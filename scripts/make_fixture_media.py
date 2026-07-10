#!/usr/bin/env python3
"""Build the fast-fixture media under ``tests/fixtures/media/`` from the local
downloaded dataset (``DARD/archive_org_public_domain/``).

The fixture is a small representative subset — a 30 s video trim + a few
smallest images + the smallest audio file + the two smallest PDFs — so the
objective gate runs in ~1-2 min instead of hours over the full dataset. The
selection is deterministic given the dataset (smallest by file size), so the
fixture is stable as long as the downloaded dataset is unchanged.

The media is gitignored (it is copied/derived from the gitignored DARD/ dataset,
so it is per-machine). Run once per machine before the gate. Idempotent:
overwrites.

Usage::

    python scripts/make_fixture_media.py

Requires the dataset present under ``DARD/archive_org_public_domain/`` (the
download stage) and imageio-ffmpeg (a project dependency) for the video trim.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "DARD" / "archive_org_public_domain"
OUT = REPO_ROOT / "tests" / "fixtures" / "media"
VIDEO_SECONDS = 30


def _smallest(glob_dir: Path, pattern: str = "*", n: int = 1) -> list[Path]:
    files = [p for p in glob_dir.rglob(pattern) if p.is_file()]
    files.sort(key=lambda p: p.stat().st_size)
    return files[:n]


def main(argv: list[str] | None = None) -> int:
    if not SRC.exists():
        print(
            f"error: dataset not found at {SRC} — run the download stage first",
            file=sys.stderr,
        )
        return 2

    (OUT / "videos" / "eng").mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "audio" / "eng").mkdir(parents=True, exist_ok=True)
    (OUT / "texts" / "eng").mkdir(parents=True, exist_ok=True)

    # 1) 30s trim of the smallest video.
    videos = _smallest(SRC / "videos", "*.mp4")
    if not videos:
        print("error: no .mp4 videos found in dataset", file=sys.stderr)
        return 2
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("error: imageio-ffmpeg not installed", file=sys.stderr)
        return 2
    out_video = OUT / "videos" / "eng" / "_test_short.mp4"
    rc = subprocess.call(
        [
            ff,
            "-y",
            "-i",
            str(videos[0]),
            "-t",
            str(VIDEO_SECONDS),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            str(out_video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if rc != 0 or not out_video.exists():
        print(f"error: ffmpeg trim failed (rc={rc}) for {videos[0].name}", file=sys.stderr)
        return 2
    print(f"[fixture] video: 30s trim of {videos[0].name} -> {out_video.name}")

    # 2) 3 smallest images.
    for img in _smallest(SRC / "images", "*", 3):
        shutil.copy2(img, OUT / "images" / img.name)
    print("[fixture] images: 3 copied")

    # 3) smallest audio.
    audio = _smallest(SRC / "audio", "*.mp3") or _smallest(SRC / "audio", "*")
    if audio:
        shutil.copy2(audio[0], OUT / "audio" / "eng" / audio[0].name)
        print(f"[fixture] audio: {audio[0].name}")

    # 4) 2 smallest PDFs.
    pdfs = _smallest(SRC / "texts", "*.pdf", 2)
    for pdf in pdfs:
        shutil.copy2(pdf, OUT / "texts" / "eng" / pdf.name)
    print(f"[fixture] texts: {len(pdfs)} PDFs copied")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"[fixture] total: {total / 1e6:.1f} MB under {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
