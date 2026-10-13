"""Reproducibility metadata for sidecars.

`add_fair_metadata` injects a `producer` block into every sidecar recording the
pipeline commit, the config file hash and the model files loaded by the stage, so
a dataset can be traced to the exact code + configuration + weights that
generated it (EU AI Act Annex IV).

The config path is read from the ``DARDCOLLECT_CONFIG`` environment variable
(set by the pipeline orchestrator for each stage subprocess). The commit comes
from ``DARD_PIPELINE_COMMIT`` when set, else from the checkout's git HEAD.
Model files are registered by the loader that opens them
(`onnx_utils.create_ort_session`) and hashed once per process. Each value is
omitted when it cannot be determined (the block is optional).
"""

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from threading import Lock

_REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def pipeline_commit() -> str | None:
    """Short git HEAD of the running checkout, or None when unavailable.

    ``DARD_PIPELINE_COMMIT`` (an explicit override for deployed runs) wins;
    otherwise the checkout's short HEAD is read once per process.
    """
    override = os.environ.get("DARD_PIPELINE_COMMIT")
    if override:
        return override
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


@lru_cache(maxsize=8)
def _config_hash(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def config_sha256() -> str | None:
    """sha256 of the config file named by ``DARDCOLLECT_CONFIG``, or None."""
    path = os.environ.get("DARDCOLLECT_CONFIG")
    if not path:
        return None
    return _config_hash(path)


def producer_block() -> dict:
    """The reproducibility block for a sidecar, or {} when nothing is known."""
    block: dict = {}
    commit = pipeline_commit()
    if commit:
        block["pipeline_commit"] = commit
    cfg = config_sha256()
    if cfg:
        block["config_sha256"] = cfg
    models = _models_block()
    if models:
        block["models"] = models
    return block


# ── model files ───────────────────────────────────────────────────────────────

_MODEL_LOCK = Lock()
_registered_models: list[str] = []


def register_model(path: str | Path) -> None:
    """Record a model file loaded by this process (dedup, order-stable).

    Called by the loader that opens the weights (``onnx_utils.create_ort_session``)
    so ``producer.models`` lists only the models the stage actually used.
    """
    resolved = str(Path(path))
    with _MODEL_LOCK:
        if resolved not in _registered_models:
            _registered_models.append(resolved)


@lru_cache(maxsize=64)
def _file_sha256(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _models_block() -> list[dict]:
    with _MODEL_LOCK:
        registered = list(_registered_models)
    out: list[dict] = []
    for path in registered:
        digest = _file_sha256(path)
        if digest:
            out.append({"file": Path(path).name, "sha256": digest})
    return sorted(out, key=lambda m: m["file"])
