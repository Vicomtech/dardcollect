"""Provenance utilities for the data collection pipeline."""

from dataclasses import asdict
from datetime import UTC, datetime


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def config_as_dict(cfg) -> dict:
    """Convert a dataclass config to a plain dict for serialisation."""
    return asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
