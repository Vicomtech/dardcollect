"""Provenance and timestamp utilities for the data collection pipeline.

Provides helpers for generating ISO 8601 timestamps for inclusion in
metadata sidecars.
"""

from datetime import UTC, datetime


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 formatted string.

    Returns:
        str: Timestamp in the format "YYYY-MM-DDTHH:MM:SS.ssssss+00:00".
    """
    return datetime.now(UTC).isoformat()
