#!/usr/bin/env python3
"""Run the DARDcollect pipeline in progressive mode.

This orchestrator runs stage scripts as incremental workers with dependency
ordering. Downstream stages re-run periodically while upstream stages keep
producing artifacts, so results appear progressively across the full pipeline.
Stage scripts read ``configs/config.archive_all.yaml`` themselves via ``DARDCOLLECT_CONFIG``.

Usage::

    python scripts/run_pipeline.py                 # all stages (progressive, default config)
    python scripts/run_pipeline.py --config configs/config.test.yaml

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

The download stage is auto-skipped if using ``configs/config.test.yaml`` (fixture workflow)
or when ``run_pipeline.skip_download=true`` is set in config.

Individual downstream stages can be skipped via ``run_pipeline.skip_stages``
(a list of stage aliases); each named stage is skipped along with every stage
that transitively depends on it, so downstream stages don't run with missing
inputs. Valid aliases: clips, audio_clips, images, face_crops_video,
face_crops_image, transcribe_video, transcribe_audio, docs, quality, filter,
frames, masks, download.

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

from dardcollect.orchestrator_plan import (
    DEFER_UNTIL_DEPS_DONE,
    HEARTBEAT_INTERVAL_SECONDS,
    PIPELINE_DIR,
    REPO_ROOT,
    RERUN_INTERVAL_SECONDS,
    _build_progressive_input_waits,
    _build_stage_plan,
    _effective_dependencies,
    _format_elapsed,
    _has_required_stage_inputs,
)


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


def _run_stage_once(state: StageState, py: str, child_env, lock: Lock) -> int:
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

        time.sleep(1)

    return "stopped"


def _handle_stage_result(
    state: StageState,
    states: dict[str, StageState],
    rc: int,
    rerun_interval_s: int,
    lock: Lock,
    stop_event: Event,
) -> bool:
    """Act on a stage run's rc. Returns True to continue the progressive loop,
    False to stop (stage finished/failed, or dependencies failed/stopped).

    - rc != 0 + deps still producing → transient, retry after waiting (continue).
    - rc != 0 + deps done → permanent failure: mark failed, signal others to stop.
    - rc == 0 + no deps → root stage converged (stop).
    - rc == 0 + deps converged → finished (stop).
    - rc == 0 + deps still producing → wait for more, then run again (continue).
    """
    if rc != 0:
        with lock:
            dep_states = [states[d] for d in state.deps]
            deps_incomplete = any(not dep.finished for dep in dep_states)
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
            return wait_result not in {"deps_failed", "stopped"}
        with lock:
            state.failed = True
            state.waiting_reason = ""
        stop_event.set()
        return False

    with lock:
        dep_states = [states[d] for d in state.deps]
        if not dep_states:
            # Root stage converges after one successful run.
            state.finished = True
            return False
        deps_finished = all(dep.finished for dep in dep_states)
        latest_dep_end = max(dep.last_end_ts for dep in dep_states)
        converged = deps_finished and state.last_end_ts >= latest_dep_end
        if converged:
            state.finished = True
            state.waiting_reason = ""
            return False
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
    return wait_result not in {"deps_failed", "stopped"}


def _stage_worker(
    state: StageState,
    states: dict[str, StageState],
    py: str,
    child_env,
    rerun_interval_s: int,
    input_waits: dict[str, list[Path]],
    lock: Lock,
    stop_event: Event,
) -> None:
    """Run a stage progressively until its dependencies and outputs converge.

    For stages in DEFER_UNTIL_DEPS_DONE (heavy-model stages), the first launch is
    deferred until all deps FINISH — so the stage loads its models once (after
    deps freed GPU) and runs a single pass, instead of re-launching every
    rerun_interval while deps still produce (which would reload models each time).
    """
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

        # Defer heavy-model stages until deps FINISH: avoids re-launching (and
        # reloading models) every rerun_interval while deps still produce. The
        # stage then launches once (deps done) and converges after a single run.
        if state.alias in DEFER_UNTIL_DEPS_DONE and state.deps and not deps_finished:
            with lock:
                state.waiting_reason = "waiting for deps to finish (defer-launch)"
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
        if not _handle_stage_result(state, states, rc, rerun_interval_s, lock, stop_event):
            return


def _launch_stage_workers(
    active: list[tuple[str, str]],
    states: dict[str, StageState],
    py: str,
    child_env,
    rerun_interval: int,
    progressive_input_waits: dict[str, list[Path]],
) -> tuple[list[Thread], Lock]:
    """Start one daemon thread per active stage. Returns (threads, lock).

    The stop_event is created here and owned by the workers; the caller does not
    need it (a failing worker sets it to halt the others).
    """
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
    return threads, lock


def _run_heartbeat(
    threads: list[Thread],
    states: dict[str, StageState],
    start_time: float,
    heartbeat_interval: int,
    lock: Lock,
) -> None:
    """Print periodic progress until all stage threads finish, then join them."""
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


def _print_summary(
    active: list[tuple[str, str]],
    states: dict[str, StageState],
    start_time: float,
) -> list[str]:
    """Print the per-stage summary and total time. Returns the list of failed aliases."""
    print("\n[run_pipeline] summary:")
    failures: list[str] = []
    for alias, _ in active:
        state = states[alias]
        mark = "FAIL" if state.failed else ("skip" if state.skipped else "ok")
        print(
            f"  [{mark}] {state.alias:<18} {state.script} "
            f"(runs={state.runs}, last={state.last_elapsed_s:.1f}s)"
        )
        if state.failed:
            failures.append(alias)
    print(f"\n[run_pipeline] total time: {_format_elapsed(time.time() - start_time)}")
    return failures


def _resolve_intervals(settings: dict) -> tuple[int, int]:
    """Resolve (heartbeat, rerun) intervals from run_pipeline settings, clamped to >=1."""
    hb = settings.get("heartbeat_interval_seconds")
    heartbeat = max(1, hb) if isinstance(hb, int) else HEARTBEAT_INTERVAL_SECONDS
    rr = settings.get("rerun_interval_seconds")
    rerun = max(1, rr) if isinstance(rr, int) else RERUN_INTERVAL_SECONDS
    return heartbeat, rerun


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
            "DARDCOLLECT_CONFIG env var (default: configs/config.archive_all.yaml)"
        ),
    )
    args = parser.parse_args(argv)

    py = _find_python(None)
    print(f"[run_pipeline] interpreter: {py}")

    config_path = (
        Path(args.config).resolve()
        if args.config
        else Path("configs/config.archive_all.yaml").resolve()
    )
    is_fixture = "test" in config_path.name.lower()
    plan = _build_stage_plan(config_path, is_fixture)

    child_env = None
    if args.config:
        child_env = {**os.environ, "DARDCOLLECT_CONFIG": str(config_path)}
        print(f"[run_pipeline] config: {args.config}")
    else:
        print("[run_pipeline] config: configs/config.archive_all.yaml (default)")
    # Disable tqdm bars in stage subprocesses so they don't interleave with the
    # orchestrator's === stage START/OK === prints. Stage output stays a plain log.
    if child_env is not None:
        child_env["TQDM_DISABLE"] = "1"
    print(f"[run_pipeline] media_types: {sorted(plan.media_types)}")
    if is_fixture:
        print("[run_pipeline] fixture workflow (skipping download)")
    elif plan.skip_download_effective:
        print("[run_pipeline] config workflow (download skipped)")
    else:
        print("[run_pipeline] full workflow (including download)")
    print("[run_pipeline] execution mode: progressive")

    start_time = time.time()
    heartbeat, rerun = _resolve_intervals(plan.run_pipeline_settings)
    print(f"[run_pipeline] intervals: heartbeat={heartbeat}s, rerun={rerun}s")

    active = list(plan.stages)
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

    threads, lock = _launch_stage_workers(
        active, states, py, child_env, rerun, progressive_input_waits
    )
    _run_heartbeat(threads, states, start_time, heartbeat, lock)
    failures = _print_summary(active, states, start_time)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
