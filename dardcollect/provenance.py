"""Provenance and timestamp utilities for the data collection pipeline.

Provides helpers for generating ISO 8601 timestamps and serializing
configuration objects for inclusion in metadata sidecars.
"""

from dataclasses import asdict
from datetime import UTC, datetime


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 formatted string.

    Returns:
        str: Timestamp in the format "YYYY-MM-DDTHH:MM:SS.ssssss+00:00".
    """
    return datetime.now(UTC).isoformat()


def config_as_dict(cfg) -> dict:
    """Convert a dataclass or dict-like config to a plain dictionary.

    Useful for serializing detector/tracker configuration into JSON sidecars.

    Args:
        cfg: Configuration object (dataclass instance or dict).

    Returns:
        dict: Plain dictionary representation of the config.
    """
    return asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
