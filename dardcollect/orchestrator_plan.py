"""Pure config/plan helpers for the pipeline orchestrator (``scripts/run_pipeline.py``).

Split out of the orchestrator so the plan logic (which stages to run, what to
skip, where inputs live) is testable and the orchestrator file stays under the
god-file size limit. This module holds no execution/loop/threading code — only
pure functions over the config + the static stage graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Static stage graph ────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_elapsed(total_seconds: float) -> str:
    """Format elapsed seconds in a compact human-readable string."""
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


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
    """Resolve a config path the way the stage scripts do: relative to the repo
    root (the cwd the stages run in), NOT relative to the config file's directory.

    Stage scripts do ``Path(cfg.input_dir)`` which is relative to their cwd (the
    repo root, since the orchestrator launches them there). The wait-paths must
    resolve identically or downstream stages are wrongly marked as having no
    inputs. ``config_path`` is kept for callers but no longer used for resolution.
    """
    p = Path(raw_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
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


def _expand_skip_to_downstream(skip_aliases: set[str]) -> set[str]:
    """Expand skipped stage aliases to include every stage that transitively
    depends on any of them (reverse closure over STAGE_DEPENDENCIES).

    Skipping a stage without this cascade would leave its downstream stages
    running with missing inputs (they'd error instead of skipping cleanly).
    Example: skip ``filter`` → also skip ``frames`` (dep: filter) and ``masks``
    (deps: filter, frames).
    """
    dependents: dict[str, list[str]] = {}
    for alias, deps in STAGE_DEPENDENCIES.items():
        for dep in deps:
            dependents.setdefault(dep, []).append(alias)
    closure = set(skip_aliases)
    changed = True
    while changed:
        changed = False
        for alias in list(closure):
            for downstream in dependents.get(alias, []):
                if downstream not in closure:
                    closure.add(downstream)
                    changed = True
    return closure


# ── Stage plan ────────────────────────────────────────────────────────────────


@dataclass
class StagePlan:
    """The resolved plan for one orchestrator run."""

    config_path: Path
    is_fixture: bool
    media_types: set[str]
    run_pipeline_settings: dict
    stages: list[tuple[str, str]]
    skip_set: set[str]
    skip_download_effective: bool


def _build_stage_plan(config_path: Path, is_fixture: bool) -> StagePlan:
    """Load config, resolve skip_stages (with downstream cascade), and build the
    ordered stage list (download prepended for full runs). Prints skip warnings.
    """
    media_types = _load_media_types(config_path)
    settings = _load_run_pipeline_settings(config_path)
    skip_download = bool(settings.get("skip_download", False))

    # Optional per-stage skip (run_pipeline.skip_stages: [aliases]). Each named
    # stage is skipped along with every stage that transitively depends on it
    # (reverse closure), so downstream stages don't run with missing inputs.
    valid_aliases = {a for a, _ in STAGES} | {DOWNLOAD_STAGE[0]}
    skip_raw = settings.get("skip_stages", []) or []
    skip_requested = {str(s).strip() for s in skip_raw if str(s).strip()}
    unknown_skip = skip_requested - valid_aliases
    if unknown_skip:
        print(
            f"[run_pipeline] WARNING: unknown skip_stages (ignored): "
            f"{sorted(unknown_skip)}; valid: {sorted(valid_aliases)}",
            flush=True,
        )
        skip_requested &= valid_aliases
    skip_set = _expand_skip_to_downstream(skip_requested)
    if skip_set:
        print(
            f"[run_pipeline] skip_stages: {sorted(skip_requested)} -> "
            f"skipping (with downstream): {sorted(skip_set)}",
            flush=True,
        )
    # skip_stages including 'download' is treated like skip_download.
    skip_download_effective = skip_download or "download" in skip_set

    stages = [
        (a, s)
        for (a, s) in STAGES
        if _stage_enabled_for_media(a, media_types) and a not in skip_set
    ]
    # Prepend download for full runs when there is at least one active stage.
    if not is_fixture and not skip_download_effective and stages:
        stages.insert(0, DOWNLOAD_STAGE)

    return StagePlan(
        config_path=config_path,
        is_fixture=is_fixture,
        media_types=media_types,
        run_pipeline_settings=settings,
        stages=stages,
        skip_set=skip_set,
        skip_download_effective=skip_download_effective,
    )
