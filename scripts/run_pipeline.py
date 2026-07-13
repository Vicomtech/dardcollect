#!/usr/bin/env python3
"""Run the DARDcollect pipeline in progressive mode.

This orchestrator runs stage scripts as incremental workers with dependency
ordering. Downstream stages re-run periodically while upstream stages keep
producing artifacts, so results appear progressively across the full pipeline.
Stage scripts read ``config.yaml`` themselves via ``DARDCOLLECT_CONFIG``.

Usage::

    python scripts/run_pipeline.py                 # all stages (progressive)
    python scripts/run_pipeline.py --config config.test.yaml

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
    frames             extract_frames_from_videos
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
    ("frames", "extract_frames_from_videos"),
    ("masks", "generate_face_masks"),
]

# Download is prepended for full runs (only if not using fixture config)
DOWNLOAD_STAGE: tuple[str, str] = ("download", "download_media_from_archive")
HEARTBEAT_INTERVAL_SECONDS = 10
RERUN_INTERVAL_SECONDS = 5
WAIT_POLL_INTERVAL_SECONDS = 1

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
    "frames": ["filter"],
    "masks": ["filter", "frames"],
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
    started: bool = False
    runs: int = 0
    failed: bool = False
    finished: bool = False
    skipped: bool = False
    in_progress: bool = False
    waiting_reason: str = ""
    last_end_ts: float = 0.0
    last_elapsed_s: float = 0.0
    last_rc: int = 0


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


def _load_media_types(config_path: Path) -> set[str]:
    """Load enabled media modalities from config (video/image/audio/text)."""
    if not config_path.exists():
        return {"video"}
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("media_types", ["video"])
    if not isinstance(raw, list):
        return {"video"}
    allowed = {"video", "image", "audio", "text"}
    parsed = {str(item).strip().lower() for item in raw}
    selected = parsed & allowed
    return selected or {"video"}


def _stage_enabled_for_media(alias: str, media_types: set[str]) -> bool:
    """Return whether a stage should run for the configured modality set."""
    stage_modalities: dict[str, set[str]] = {
        "clips": {"video"},
        "audio_clips": {"video"},
        "images": {"image"},
        "face_crops_video": {"video"},
        "face_crops_image": {"image"},
        "transcribe_video": {"video"},
        "transcribe_audio": {"audio"},
        "docs": {"text"},
        "quality": {"video", "image"},
        "filter": {"video", "image"},
        "frames": {"video"},
        "masks": {"video", "image"},
    }
    required = stage_modalities.get(alias)
    if required is None:
        return True
    return bool(required & media_types)


def _resolve_config_path(raw_path: str, config_path: Path) -> Path:
    """Resolve a config path relative to the config file directory."""
    p = Path(raw_path)
    if not p.is_absolute():
        p = config_path.parent / p
    return p.resolve()


def _build_progressive_input_waits(config_path: Path) -> dict[str, list[Path]]:
    """Build stage input-path checks used to avoid transient progressive failures.

    Only stages that are known to raise hard errors on missing inputs are listed.
    """
    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Expand {root}/{output_root} path templates so wait_paths in the
    # readiness check resolve to real directories on disk.
    try:
        from dardcollect.config import _resolve_path_templates

        data = _resolve_path_templates(data)
    except (ImportError, AttributeError):
        pass  # fall through with literal paths if helper unavailable

    stage_sections: dict[str, list[tuple[str, str]]] = {
        "clips": [("person_extraction", "input_dir")],
        "images": [("image_extraction", "input_dir")],
        "audio_clips": [("person_extraction", "output_clips_dir")],
        "face_crops_video": [("face_crop_extraction", "input_dir")],
        "face_crops_image": [("image_face_crop_extraction", "input_dir")],
        "transcribe_video": [("transcription", "person_clips_dir")],
        "filter": [
            ("face_quality_filtering", "input_dir"),
            ("image_face_quality_filtering", "input_dir"),
        ],
        "frames": [("frame_extraction", "input_dir")],
    }

    waits: dict[str, list[Path]] = {}
    for alias, section_keys in stage_sections.items():
        paths: list[Path] = []
        for section, key in section_keys:
            section_data = data.get(section, {})
            if not isinstance(section_data, dict):
                continue
            raw = section_data.get(key)
            if isinstance(raw, str) and raw.strip():
                paths.append(_resolve_config_path(raw, config_path))
        if paths:
            waits[alias] = paths

    return waits


def _has_required_stage_inputs(alias: str, paths: list[Path]) -> bool:
    """Return True when a stage has usable inputs in at least one configured path.

    Some stages fail hard when a directory exists but is still empty. For those,
    readiness means matching files are present, not just the directory.
    """
    if not paths:
        return True

    required_globs: dict[str, tuple[str, ...]] = {
        "clips": ("*.mp4", "*.avi", "*.mkv", "*.mov"),
        "face_crops_video": ("*.mp4",),
        "frames": ("*.mp4",),
        "face_crops_image": (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.tiff",
            "*.bmp",
            "*.webp",
        ),
    }

    globs = required_globs.get(alias)
    if globs is None:
        return any(p.exists() for p in paths)

    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            if any(p.match(pattern) for pattern in globs):
                return True
            continue
        for pattern in globs:
            if any(p.rglob(pattern)):
                return True

    return False


def _effective_dependencies(alias: str, active_aliases: set[str]) -> list[str]:
    """Return dependencies for a stage, filtered to active stages only."""
    return [dep for dep in STAGE_DEPENDENCIES.get(alias, []) if dep in active_aliases]


def _run_stage_once(
    state: StageState,
    py: str,
    child_env: dict[str, str] | None,
    lock: Lock,
) -> int:
    """Execute one stage run and update shared state."""
    with lock:
        state.started = True
        state.in_progress = True
        state.waiting_reason = ""

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
        state.last_rc = rc
        state.in_progress = False

    return rc


def _wait_for_dependency_progress(
    state: StageState,
    states: dict[str, StageState],
    lock: Lock,
    stop_event: Event,
    since_ts: float,
    max_wait_s: int,
) -> str:
    """Wait until dependencies update, finish, or timeout.

    This avoids long fixed sleeps when upstream stages produce outputs sooner.
    """
    start = time.time()
    while not stop_event.is_set():
        with lock:
            dep_states = [states[d] for d in state.deps]
            if any(dep.failed for dep in dep_states):
                return "deps_failed"
            deps_finished = all(dep.finished for dep in dep_states)
            latest_dep_end = max((dep.last_end_ts for dep in dep_states), default=0.0)

        if latest_dep_end > since_ts:
            return "deps_updated"
        if deps_finished:
            return "deps_finished"
        if (time.time() - start) >= max_wait_s:
            return "timeout"

        time.sleep(WAIT_POLL_INTERVAL_SECONDS)

    return "stopped"


def _stage_worker(
    state: StageState,
    states: dict[str, StageState],
    py: str,
    child_env: dict[str, str] | None,
    rerun_interval_s: int,
    input_waits: dict[str, list[Path]],
    lock: Lock,
    stop_event: Event,
) -> None:
    """Run a stage progressively until its dependencies and outputs converge."""
    while not stop_event.is_set():
        # Wait until dependencies have started at least once so first-run ordering
        # is progressive but still dependency-aware.
        with lock:
            dep_states = [states[d] for d in state.deps]
            deps_ready = all(dep.started or dep.finished for dep in dep_states)
            deps_failed = any(dep.failed for dep in dep_states)
            deps_finished = all(dep.finished for dep in dep_states)

        if deps_failed:
            return
        if state.deps and not deps_ready:
            with lock:
                pending = [dep.alias for dep in dep_states if not dep.started and not dep.finished]
                state.waiting_reason = f"waiting deps to start: {','.join(pending)}"
            time.sleep(1)
            continue

        # Progressive guard: wait for known hard-required inputs before launching
        # stages that would otherwise fail noisily while upstream is still running.
        wait_paths = input_waits.get(state.alias, [])
        if state.deps and wait_paths and not _has_required_stage_inputs(state.alias, wait_paths):
            if deps_finished:
                print(
                    f"[run_pipeline] {state.alias}: no usable inputs produced by dependencies; "
                    "marking stage as skipped",
                    flush=True,
                )
                with lock:
                    state.skipped = True
                    state.finished = True
                    state.waiting_reason = ""
                return
            with lock:
                state.waiting_reason = "waiting for upstream inputs"
            time.sleep(1)
            continue

        rc = _run_stage_once(state, py, child_env, lock)
        if rc != 0:
            with lock:
                dep_states = [states[d] for d in state.deps]
                deps_incomplete = any(not dep.finished for dep in dep_states)

            # Transient failure while dependencies are still producing outputs.
            # Common case: downstream starts before first input artifact exists.
            if dep_states and deps_incomplete:
                print(
                    "[run_pipeline] transient failure in "
                    f"{state.alias}; retrying in {rerun_interval_s}s",
                    flush=True,
                )
                wait_result = _wait_for_dependency_progress(
                    state=state,
                    states=states,
                    lock=lock,
                    stop_event=stop_event,
                    since_ts=state.last_end_ts,
                    max_wait_s=rerun_interval_s,
                )
                if wait_result in {"deps_failed", "stopped"}:
                    return
                continue

            with lock:
                state.failed = True
                state.waiting_reason = ""
            stop_event.set()
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
                state.waiting_reason = ""
                return

            state.waiting_reason = "waiting for deps to converge"

        # Dependencies are still producing; run again later to pick up new outputs.
        wait_result = _wait_for_dependency_progress(
            state=state,
            states=states,
            lock=lock,
            stop_event=stop_event,
            since_ts=state.last_end_ts,
            max_wait_s=rerun_interval_s,
        )
        if wait_result in {"deps_failed", "stopped"}:
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
        if __doc__
        else "Run the DARDcollect pipeline stages in order."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "config path override passed to every stage via the "
            "DARDCOLLECT_CONFIG env var (default: each stage's config.yaml)"
        ),
    )
    args = parser.parse_args(argv)

    py = _find_python(None)
    print(f"[run_pipeline] interpreter: {py}")

    # Auto-detect fixture workflow and config-driven overrides
    config_path = Path(args.config).resolve() if args.config else Path("config.yaml").resolve()
    is_fixture = "test" in config_path.name.lower()
    media_types = _load_media_types(config_path)
    run_pipeline_settings = _load_run_pipeline_settings(config_path)
    skip_download = bool(run_pipeline_settings.get("skip_download", False))

    # Build stages list from configured media modalities.
    stages = [(a, s) for (a, s) in STAGES if _stage_enabled_for_media(a, media_types)]

    # Prepend download for full runs when there is at least one active stage.
    if not is_fixture and not skip_download:
        if stages:
            stages.insert(0, DOWNLOAD_STAGE)

    child_env = None
    if args.config:
        child_env = {**os.environ, "DARDCOLLECT_CONFIG": str(config_path)}
        print(f"[run_pipeline] config: {args.config}")
    else:
        print("[run_pipeline] config: config.yaml (default)")
    # Disable tqdm bars in stage subprocesses — without this, each stage's
    # progress bar interleaves with the orchestrator's "=== stage START/OK ==="
    # prints and produces unreadable output. Stage output stays a plain log.
    if child_env is not None:
        child_env["TQDM_DISABLE"] = "1"
    print(f"[run_pipeline] media_types: {sorted(media_types)}")

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
        heartbeat_interval = HEARTBEAT_INTERVAL_SECONDS

    rerun_interval_override = run_pipeline_settings.get("rerun_interval_seconds")
    if isinstance(rerun_interval_override, int):
        rerun_interval = max(1, rerun_interval_override)
    else:
        rerun_interval = RERUN_INTERVAL_SECONDS

    print(f"[run_pipeline] intervals: heartbeat={heartbeat_interval}s, rerun={rerun_interval}s")

    active = list(stages)
    active_aliases = {a for a, _ in active}
    progressive_input_waits = _build_progressive_input_waits(config_path)

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
                progressive_input_waits,
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
            running = [s.alias for s in states.values() if s.in_progress and not s.finished]
            waiting = [
                f"{s.alias}({s.waiting_reason})"
                for s in states.values()
                if (not s.in_progress and not s.finished and s.waiting_reason)
            ]
            skipped = [s.alias for s in states.values() if s.skipped]
            total_elapsed = _format_elapsed(time.time() - start_time)
        running_txt = ",".join(running[:4]) if running else "-"
        waiting_txt = ",".join(waiting[:2]) if waiting else "-"
        skipped_txt = ",".join(skipped[:4]) if skipped else "-"
        print(
            f"[run_pipeline] elapsed total: {total_elapsed} | "
            f"finished stages: {finished}/{len(states)} | failed: {failed} | "
            f"running: {running_txt} | waiting: {waiting_txt} | skipped: {skipped_txt}",
            flush=True,
        )

    for t in threads:
        t.join()

    print("\n[run_pipeline] summary:")
    failures: list[str] = []
    for alias, _script in active:
        state = states[alias]
        mark = "FAIL" if state.failed else ("skip" if state.skipped else "ok")
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
