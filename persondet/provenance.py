"""Provenance record utilities for the data collection pipeline."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

FORMAT_VERSION = "1.0"
PROVENANCE_FILENAME = "dataset_provenance.json"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def model_info(path: Path) -> dict:
    """Return name and SHA-256 checksum for a model file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except OSError:
        sha = "unavailable"
    return {"name": path.name, "sha256": sha}


def record_stage(provenance_path: Path, run: dict) -> None:
    """Append a pipeline-stage run entry to the provenance record.

    Creates the file on first call; subsequent calls append to pipeline_runs.
    """
    if provenance_path.exists():
        with open(provenance_path, encoding="utf-8") as f:
            record = json.load(f)
    else:
        record = {
            "format_version": FORMAT_VERSION,
            "dataset_created_at": now_iso(),
            "pipeline_runs": [],
        }

    record["dataset_updated_at"] = now_iso()
    pipeline_runs: list = cast(list, record.setdefault("pipeline_runs", []))
    pipeline_runs.append(run)

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def config_as_dict(cfg) -> dict:
    """Convert a dataclass config to a plain dict for serialisation."""
    return asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
