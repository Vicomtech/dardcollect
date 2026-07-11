"""Simple elapsed time tracker for pipeline stages."""

import functools
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ElapsedTimer:
    """Track elapsed time with human-readable formatting."""

    def __init__(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Elapsed seconds since creation."""
        return time.time() - self.start_time

    def format_elapsed(self) -> str:
        """Return human-readable elapsed time (e.g., '1h 2m 3s', '45s')."""
        elapsed = int(self.elapsed())
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def __str__(self) -> str:
        """String representation: human-readable elapsed time."""
        return self.format_elapsed()


def add_timer(func):
    """Decorator: wraps a function with automatic timing + logging."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = ElapsedTimer()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("[total time] %s", timer)

    return wrapper


@contextmanager
def timer(name: str = ""):
    """Context manager for timing a block with optional logging."""
    t = ElapsedTimer()
    try:
        yield t
    finally:
        if name:
            print(f"[timer] {name}: {t}")
