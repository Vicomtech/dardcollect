"""Config path-template substitution (``{root}`` / ``{key}`` interpolation).

Split out of ``dardcollect/config.py`` (2026-10-06) when that module crossed the
600-line god-file cap. Used by the stage-dataclass loaders in ``config.py`` and
by the viewer / standalone scripts that need the same substitution.
"""

from __future__ import annotations

from typing import Any


def _resolve_path_templates(config_data: dict) -> dict:
    """Recursively replace ``{root}`` (and any ``{key}`` from the top-level
    config) in every string value of *config_data*.

    Returns a new dict; does not mutate the input. Substitutable keys are
    taken from the top-level config (any value that is a string/int/float/bool).
    The top-level source-of-truth entries are not themselves templated
    (avoids ``{root}`` being applied to ``root: '...{root}...'``).
    """
    if not isinstance(config_data, dict):
        return config_data
    substitutions = {k: v for k, v in config_data.items() if isinstance(v, (str, int, float, bool))}
    return _apply_substitutions(config_data, substitutions, source_keys=set(substitutions))


def _apply_substitutions(obj: Any, subs: dict, source_keys: set) -> Any:
    """Walk *obj* and return a copy with ``{key}`` interpolated in strings."""
    if isinstance(obj, dict):
        return {
            k: v if k in source_keys else _apply_substitutions(v, subs, source_keys)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_apply_substitutions(item, subs, source_keys) for item in obj]
    if isinstance(obj, str):
        # Resolve every ``{key}`` that has a known substitution. Leave
        # unknown placeholders literal so configs can mix templated and
        # untemplated strings without one aborting the other.
        class _SafeDict(dict):
            def __missing__(self, key):  # type: ignore[override]
                return "{" + key + "}"

        try:
            return obj.format_map(_SafeDict(**subs))
        except (KeyError, IndexError, ValueError):
            return obj
    return obj
