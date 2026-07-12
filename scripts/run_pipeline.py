#!/usr/bin/env python3
"""Run the DARDcollect pipeline stages in order.

A thin cross-platform orchestrator so the objective-verification loop is
reproducible: it runs each stage via ``uv run python pipeline/<stage>.py``,
logs start/exit, continues on failure (so you see every stage's result), and
exits non-zero if any stage failed. Stage scripts read ``config.yaml``
themselves.

Usage::

    python scripts/run_pipeline.py                 # all stages
    python scripts/run_pipeline.py --stages clips,face_crops_video
    python scripts/run_pipeline.py --python .venv/Scripts/python.exe

Stages (in run order, auto-detected):

    download           download_media_from_archive (only if not using fixture config)
    clips              extract_person_clips_from_videos
    images             extract_persons_from_images
    face_crops_video   extract_face_crops_from_videos
    face_crops_image   extract_face_crops_from_images
    transcribe_video   transcribe_video_clips
    transcribe_audio   transcribe_audio_files
    docs               extract_text_from_doc
    quality            annotate_face_quality
    filter             filter_face_crops_by_quality

The download stage is auto-skipped if using ``config.test.yaml`` (fixture workflow).
For full data runs, invoke without ``--config`` (uses ``config.yaml``) to download +
process. Frame extraction (``extract_frames_from_videos``) is optional and skipped by
default.

This script only orchestrates subprocesses — no GPU, no models. Pair it with
``scripts/golden_snapshot.py compare ... --validate`` to verify the objective.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

STAGES: list[tuple[str, str]] = [
    ("clips", "extract_person_clips_from_videos"),
    ("audio_clips", "extract_audio_from_clips"),
    ("images", "extract_persons_from_images"),
    ("face_crops_video", "extract_face_crops_from_videos"),
    ("face_crops_image", "extract_face_crops_from_images"),
    ("transcribe_video", "transcribe_video_clips"),
    ("transcribe_audio", "transcribe_audio_files"),
    ("docs", "extract_text_from_doc"),
    ("quality", "annotate_face_quality"),
    ("filter", "filter_face_crops_by_quality"),
    ("masks", "generate_face_masks"),
]

# Download is prepended for full runs (only if not using fixture config)
DOWNLOAD_STAGE: tuple[str, str] = ("download", "download_media_from_archive")
HEARTBEAT_INTERVAL_SECONDS = 30

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"


def _format_elapsed(total_seconds: float) -> str:
    """Format elapsed seconds in a compact human-readable string."""
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _find_python(preferred: str | None) -> str:
    if preferred:
        return preferred
    for candidate in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (REPO_ROOT / candidate).exists():
            return str(REPO_ROOT / candidate)
    if shutil.which("python"):
        return "python"
    return sys.executable


def _load_run_pipeline_settings(config_path: Path) -> dict:
    """Load optional run_pipeline settings from config YAML."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    settings = data.get("run_pipeline", {})
    return settings if isinstance(settings, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
        if __doc__
        else "Run the DARDcollect pipeline stages in order."
    )
    parser.add_argument(
        "--stages",
        default="",
        help="comma-separated stage aliases to run (default: all, in order)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter to use (default: .venv python, then python)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "config path override passed to every stage via the "
            "DARDCOLLECT_CONFIG env var (default: each stage's config.yaml)"
        ),
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=HEARTBEAT_INTERVAL_SECONDS,
        help=(
            "seconds between periodic elapsed-time updates while a stage runs "
            f"(default: {HEARTBEAT_INTERVAL_SECONDS})"
        ),
    )
    args = parser.parse_args(argv)

    selected = {s.strip() for s in args.stages.split(",") if s.strip()} if args.stages else None
    py = _find_python(args.python)
    print(f"[run_pipeline] interpreter: {py}")

    # Auto-detect fixture workflow and config-driven overrides
    config_path = Path(args.config).resolve() if args.config else Path("config.yaml").resolve()
    is_fixture = "test" in config_path.name.lower()
    run_pipeline_settings = _load_run_pipeline_settings(config_path)
    skip_download = bool(run_pipeline_settings.get("skip_download", False))

    # Build stages list: prepend download for full runs
    stages = list(STAGES)
    if not is_fixture and not skip_download:
        stages.insert(0, DOWNLOAD_STAGE)

    child_env = None
    if args.config:
        child_env = {**os.environ, "DARDCOLLECT_CONFIG": str(config_path)}
        print(f"[run_pipeline] config: {args.config}")
    else:
        print("[run_pipeline] config: config.yaml (default)")

    if is_fixture:
        print("[run_pipeline] fixture workflow (skipping download)")
    elif skip_download:
        print("[run_pipeline] config workflow (run_pipeline.skip_download=true)")
    else:
        print("[run_pipeline] full workflow (including download)")

    # Global timer
    start_time = time.time()

    heartbeat_interval_override = run_pipeline_settings.get("heartbeat_interval_seconds")
    if isinstance(heartbeat_interval_override, int):
        heartbeat_interval = max(1, heartbeat_interval_override)
    else:
        heartbeat_interval = max(1, args.heartbeat_interval)

    failures: list[str] = []
    for alias, script in stages:
        if selected is not None and alias not in selected:
            continue
        script_path = PIPELINE_DIR / f"{script}.py"
        if not script_path.exists():
            print(f"[{alias}] MISSING script: {script_path}")
            failures.append(alias)
            continue
        start = time.time()
        print(f"\n=== [{alias}] {script} START ===", flush=True)
        proc = subprocess.Popen([py, str(script_path)], cwd=str(REPO_ROOT), env=child_env)
        next_heartbeat = start + heartbeat_interval
        while True:
            rc = proc.poll()
            if rc is not None:
                break

            now = time.time()
            if now >= next_heartbeat:
                stage_elapsed = _format_elapsed(now - start)
                total_elapsed = _format_elapsed(now - start_time)
                print(
                    f"[run_pipeline] elapsed total: {total_elapsed} | "
                    f"stage [{alias}]: {stage_elapsed}",
                    flush=True,
                )
                next_heartbeat += heartbeat_interval

            time.sleep(0.2)

        elapsed = time.time() - start
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"=== [{alias}] {script} {status} ({elapsed:.1f}s) ===", flush=True)
        if rc != 0:
            failures.append(alias)

    print("\n[run_pipeline] summary:")
    for alias, script in stages:
        if selected is not None and alias not in selected:
            continue
        mark = "FAIL" if alias in failures else "ok"
        print(f"  [{mark}] {alias:<18} {script}")

    # Global elapsed time
    total_elapsed = time.time() - start_time
    time_str = _format_elapsed(total_elapsed)
    print(f"\n[run_pipeline] total time: {time_str}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
