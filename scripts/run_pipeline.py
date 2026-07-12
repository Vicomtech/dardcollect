#!/usr/bin/env python3
"""Run the DARDcollect pipeline in progressive mode.

This orchestrator runs stage scripts as incremental workers with dependency
ordering. Downstream stages re-run periodically while upstream stages keep
producing artifacts, so results appear progressively across the full pipeline.
Stage scripts read ``config.yaml`` themselves via ``DARDCOLLECT_CONFIG``.

Usage::

    python scripts/run_pipeline.py                 # all stages (progressive)
    python scripts/run_pipeline.py --stages clips,face_crops_video
    python scripts/run_pipeline.py --python .venv/Scripts/python.exe

Stages (in run order, auto-detected):

    download           download_media_from_archive (only if not using fixture config)
    clips              extract_person_clips_from_videos
    audio_clips        extract_audio_from_clips
    images             extract_persons_from_images
    face_crops_video   extract_face_crops_from_videos
    face_crops_image   extract_face_crops_from_images
    transcribe_video   transcribe_video_clips
    transcribe_audio   transcribe_audio_files
    docs               extract_text_from_doc
    quality            annotate_face_quality
    filter             filter_face_crops_by_quality
    masks              generate_face_masks

The download stage is auto-skipped if using ``config.test.yaml`` (fixture workflow)
or when ``run_pipeline.skip_download=true`` is set in config.

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
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

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
RERUN_INTERVAL_SECONDS = 30

STAGE_DEPENDENCIES: dict[str, list[str]] = {
    "download": [],
    "clips": ["download"],
    "audio_clips": ["clips"],
    "images": ["download"],
    "face_crops_video": ["clips"],
    "face_crops_image": ["images"],
    "transcribe_video": ["clips"],
    "transcribe_audio": ["download"],
    "docs": ["download"],
    "quality": ["face_crops_video", "face_crops_image"],
    "filter": ["quality"],
    "masks": ["filter"],
}

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


@dataclass
class StageState:
    alias: str
    script: str
    deps: list[str]
    runs: int = 0
    failed: bool = False
    finished: bool = False
    last_end_ts: float = 0.0
    last_elapsed_s: float = 0.0


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


def _effective_dependencies(alias: str, active_aliases: set[str]) -> list[str]:
    """Return dependencies for a stage, filtered to active stages only."""
    return [dep for dep in STAGE_DEPENDENCIES.get(alias, []) if dep in active_aliases]


def _run_stage_once(
    state: StageState,
    py: str,
    child_env: dict[str, str] | None,
    lock: Lock,
    stop_event: Event,
) -> int:
    """Execute one stage run and update shared state."""
    run_id = state.runs + 1
    start = time.time()
    print(f"\n=== [{state.alias}] {state.script} START (run {run_id}) ===", flush=True)

    script_path = PIPELINE_DIR / f"{state.script}.py"
    rc = subprocess.call([py, str(script_path)], cwd=str(REPO_ROOT), env=child_env)
    elapsed = time.time() - start
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"=== [{state.alias}] {state.script} {status} ({elapsed:.1f}s) ===", flush=True)

    with lock:
        state.runs += 1
        state.last_end_ts = time.time()
        state.last_elapsed_s = elapsed
        if rc != 0:
            state.failed = True
            stop_event.set()

    return rc


def _stage_worker(
    state: StageState,
    states: dict[str, StageState],
    py: str,
    child_env: dict[str, str] | None,
    rerun_interval_s: int,
    lock: Lock,
    stop_event: Event,
) -> None:
    """Run a stage progressively until its dependencies and outputs converge."""
    while not stop_event.is_set():
        rc = _run_stage_once(state, py, child_env, lock, stop_event)
        if rc != 0:
            return

        with lock:
            dep_states = [states[d] for d in state.deps]
            if not dep_states:
                # Root stage converges after one successful run.
                state.finished = True
                return

            deps_finished = all(dep.finished for dep in dep_states)
            latest_dep_end = max(dep.last_end_ts for dep in dep_states)
            converged = deps_finished and state.last_end_ts >= latest_dep_end
            if converged:
                state.finished = True
                return

        # Dependencies are still producing; run again later to pick up new outputs.
        time.sleep(rerun_interval_s)


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
            "seconds between periodic orchestrator status updates "
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
    print("[run_pipeline] execution mode: progressive")

    # Global timer
    start_time = time.time()

    heartbeat_interval_override = run_pipeline_settings.get("heartbeat_interval_seconds")
    if isinstance(heartbeat_interval_override, int):
        heartbeat_interval = max(1, heartbeat_interval_override)
    else:
        heartbeat_interval = max(1, args.heartbeat_interval)

    rerun_interval_override = run_pipeline_settings.get("rerun_interval_seconds")
    if isinstance(rerun_interval_override, int):
        rerun_interval = max(1, rerun_interval_override)
    else:
        rerun_interval = RERUN_INTERVAL_SECONDS

    print(f"[run_pipeline] intervals: heartbeat={heartbeat_interval}s, rerun={rerun_interval}s")

    active = [(a, s) for (a, s) in stages if selected is None or a in selected]
    active_aliases = {a for a, _ in active}

    states: dict[str, StageState] = {}
    for alias, script in active:
        script_path = PIPELINE_DIR / f"{script}.py"
        if not script_path.exists():
            print(f"[{alias}] MISSING script: {script_path}")
            return 1
        states[alias] = StageState(
            alias=alias,
            script=script,
            deps=_effective_dependencies(alias, active_aliases),
        )

    stop_event = Event()
    lock = Lock()
    threads: list[Thread] = []
    for alias, _ in active:
        t = Thread(
            target=_stage_worker,
            args=(
                states[alias],
                states,
                py,
                child_env,
                rerun_interval,
                lock,
                stop_event,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(heartbeat_interval)
        with lock:
            finished = sum(1 for s in states.values() if s.finished)
            failed = sum(1 for s in states.values() if s.failed)
            total_elapsed = _format_elapsed(time.time() - start_time)
        print(
            f"[run_pipeline] elapsed total: {total_elapsed} | "
            f"finished stages: {finished}/{len(states)} | failed: {failed}",
            flush=True,
        )

    for t in threads:
        t.join()

    print("\n[run_pipeline] summary:")
    failures: list[str] = []
    for alias, _script in active:
        state = states[alias]
        mark = "FAIL" if state.failed else "ok"
        print(
            f"  [{mark}] {state.alias:<18} {state.script} "
            f"(runs={state.runs}, last={state.last_elapsed_s:.1f}s)"
        )
        if state.failed:
            failures.append(alias)

    # Global elapsed time
    total_elapsed = time.time() - start_time
    time_str = _format_elapsed(total_elapsed)
    print(f"\n[run_pipeline] total time: {time_str}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
