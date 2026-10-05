#!/usr/bin/env python3
"""
Content-based colour vs black-and-white video filter (issue #11).

archive.org's ``color`` metadata tag is unreliable — in the reporter's sample of
~150 public-domain films, 8 titles tagged ``color`` were actually black-and-white.
This stage classifies each video **by its pixels**: it samples a few frames,
measures mean chroma saturation, and writes ``color_classification.csv`` next to
the input tree (one row per video, resumable).

Optional ``--move`` relocates classified B&W videos to a sibling directory
(``<input_dir>_bw/``). The move is reversible: the CSV records the classification
and both source and destination paths, so moving back is an undo on the recorded
rows.

CPU-only, resumable (per-video CSV row + skip on rerun), no new sidecars, no new
dependency. The ffmpeg keyframe fast path (``-skip_frame nokey`` at low
resolution) is ~10-50x faster than full-resolution OpenCV random seeks on
feature-length films; the OpenCV fallback is logged per the runtime-fallback
policy.

This stage is NOT yet wired into run_pipeline.py's DAG — it runs standalone
(like the disk tooling) while thresholds are calibrated on real data.

Usage::

    python pipeline/filter_videos_by_color.py [config_path] [--move] [--sample-frames 12]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

import dardcollect.archive as archive_mod

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)

CSV_NAME = "color_classification.csv"
CSV_FIELDS = [
    "uuid",
    "timestamp",
    "video_path",
    "video_name",
    "classification",
    "mean_saturation",
    "frames_sampled",
    "moved",
    "moved_to",
]

# Mean HSV saturation above this → colour; below → black-and-white. Calibrated
# on real footage before enabling --move en masse (see issue #11).
BW_SATURATION_THRESHOLD = 12.0


# ── Frame sampling ────────────────────────────────────────────────────────────


def _sample_frames_ffmpeg_fast(video_path: Path, max_side: int = 320):
    """Sample frames via the ffmpeg keyframe fast path (logged fallback to OpenCV).

    ``-skip_frame nokey`` decodes only keyframes — 10-50x faster than full
    decode on feature films. Returns a list of BGR frames (possibly empty).
    """
    ffmpeg = archive_mod._ffmpeg_exe()
    if not ffmpeg:
        logger.info("No ffmpeg available — using OpenCV sampling fallback")
        return []

    with tempfile.TemporaryDirectory() as td:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-skip_frame",
                    "nokey",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"scale='min({max_side},iw)':-2",
                    "-vsync",
                    "0",
                    os.path.join(td, "frame_%06d.png"),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except Exception as exc:
            logger.info(
                "ffmpeg sampling failed for %s (%s) — OpenCV fallback", video_path.name, exc
            )
            return []

        frames = []
        for png in sorted(Path(td).glob("frame_*.png")):
            img = cv2.imread(str(png))
            if img is not None:
                frames.append(img)
        if frames:
            logger.debug("Sampled %d keyframes via ffmpeg fast path", len(frames))
        return frames


def _sample_frames_opencv(video_path: Path, n_frames: int, max_side: int = 320):
    """Uniform OpenCV sampling fallback (slow but dependency-free)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video %s", video_path.name)
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    try:
        if total <= 0:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        else:
            step = max(1, total // max(1, n_frames))
            for idx in range(0, total, step):
                if len(frames) >= n_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
    finally:
        cap.release()
    return [cv2.resize(f, (max_side, int(f.shape[0] * max_side / f.shape[1]))) for f in frames]


def sample_frames(video_path: Path, n_frames: int = 12, max_side: int = 320) -> list:
    """Sample up to *n_frames* frames from a video (ffmpeg fast path first).

    The OpenCV fallback is logged per the runtime-fallback policy.
    """
    frames = _sample_frames_ffmpeg_fast(video_path, max_side)
    if frames:
        return frames[:n_frames]
    logger.info("Using OpenCV sampling fallback for %s", video_path.name)
    return _sample_frames_opencv(video_path, n_frames, max_side)


# ── Classification ────────────────────────────────────────────────────────────


def video_color_score(video_path: Path, n_frames: int = 12) -> tuple[str, float]:
    """Classify a video by mean chroma saturation over sampled frames.

    Returns:
        (classification, mean_saturation) where classification is
        ``"color"`` | ``"black_and_white"`` | ``"unreadable"``.
    """
    frames = sample_frames(video_path, n_frames)
    if not frames:
        return "unreadable", 0.0

    saturations = []
    for frame in frames:
        small = cv2.resize(frame, (320, max(1, int(frame.shape[0] * 320 / frame.shape[1]))))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturations.append(float(hsv[:, :, 1].mean()))
    mean_sat = float(np.mean(saturations))
    classification = "color" if mean_sat >= BW_SATURATION_THRESHOLD else "black_and_white"
    return classification, round(mean_sat, 2)


# ── CSV plumbing ──────────────────────────────────────────────────────────────


def _read_csv_rows(csv_path: Path) -> dict[str, dict]:
    """Read existing classification rows keyed by video path."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["video_path"]: row for row in csv.DictReader(f)}


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────


def classify_video(
    video_path: Path,
    csv_path: Path,
    n_frames: int,
    do_move: bool,
) -> dict:
    """Classify one video; resumable via the CSV. Optionally move B&W videos.

    Returns the row written (or the existing row on a rerun skip).
    """
    existing = _read_csv_rows(csv_path)
    if video_path.name in existing:
        logger.debug("Already classified, skipping: %s", video_path.name)
        return existing[video_path.name]

    classification, mean_sat = video_color_score(video_path, n_frames)
    row = {
        "uuid": str(uuid_mod.uuid4()),
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "video_path": video_path.name,
        "video_name": video_path.stem,
        "classification": classification,
        "mean_saturation": mean_sat,
        "frames_sampled": n_frames if classification != "unreadable" else 0,
        "moved": "",
        "moved_to": "",
    }

    if do_move and classification == "black_and_white":
        bw_dir = video_path.parent / "black_and_white"
        bw_dir.mkdir(parents=True, exist_ok=True)
        dest = bw_dir / video_path.name
        if not dest.exists():
            shutil.move(str(video_path), str(dest))
            row["moved"] = "true"
            row["moved_to"] = str(dest)
            logger.info("MOVED %s → %s (black_and_white)", video_path.name, bw_dir.name)
        else:
            logger.warning(
                "Move collision for %s — destination exists, left in place", video_path.name
            )

    _append_row(csv_path, row)
    logger.info("%s: %s (mean_sat=%.1f)", video_path.name, classification, mean_sat)
    return row


def main() -> None:
    description = (__doc__ or "filter_videos_by_color").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Directory containing videos (default: config download video output dir)",
    )
    parser.add_argument("--move", action="store_true", help="Move B&W videos to a sibling folder")
    parser.add_argument("--sample-frames", type=int, default=12, help="Frames to sample per video")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        base = Path(cfg.get("base_output_dir", "./archive_org_public_domain"))
        video_cfg = cfg.get("media_download", {}).get("video", {})
        input_dir = base / video_cfg.get("output_subdir", "videos")

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    csv_path = input_dir / CSV_NAME
    videos = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v")
        and "black_and_white" not in str(p)
    )
    # Skip videos inside the B&W move destination (already relocated).
    videos = [v for v in videos if v.parent.name != "black_and_white"]

    if not videos:
        logger.info("No videos found in %s", input_dir)
        return

    logger.info(
        "Classifying %d videos in %s (sample=%d frames, move=%s)",
        len(videos),
        input_dir,
        args.sample_frames,
        args.move,
    )
    counts = {"color": 0, "black_and_white": 0, "unreadable": 0}
    for video in videos:
        row = classify_video(video, csv_path, args.sample_frames, args.move)
        cls = row.get("classification", "unreadable")
        counts[cls] = counts.get(cls, 0) + 1

    logger.info(
        "Done. color=%d  black_and_white=%d  unreadable=%d  →  %s",
        counts["color"],
        counts["black_and_white"],
        counts["unreadable"],
        csv_path,
    )


if __name__ == "__main__":
    main()
