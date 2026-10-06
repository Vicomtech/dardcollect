#!/usr/bin/env python3
"""Documented-surfaces inventory for `scripts/validate_harness.py`.

Adapted from the ai-harness-eng `component_inventory.py`. The whole
component-docs check lives here rather than inline: the god-file + code-quality
ratchets fail on growth of `validate_harness.py`, so the check is extracted and
the validator keeps only a thin wrapper (the same reason `privacy_scan.py` and
`quality_gates.py` exist). Moving it here also lets `validate_harness.py`
*shrink* instead of growing as detection is added.

Two kinds of inventory are checked:

**Contents of a known class.** Every public `pipeline/*.py` stage script must be
named in `.vscode/launch.json` or in README/docs. Private helpers (`_*.py`) are
implementation detail extracted for the god-file cap, not stages (no
entrypoint), so they are exempt.

**Existence of the class itself** (the discovery gate). The contents check above
compares only the classes someone enumerated, so a brand-new *class* of
component -- a new top-level directory -- can enter the repository with no
document naming it, invisible until a human notices. `documented_surfaces.json`
inverts the direction: it declares, for each depth-1 component directory under
its declared roots, the document that names it (token-checked) or an explicit
exemption with a reason. A class is discovered only when it holds at least one
tracked or not-ignored file, so local scaffolding does not fabricate classes.
An absent registry, an empty registry, an undeclared class, an unreasoned
exemption, or a declaration whose document does not name the class -- every one
is an error, never a silent pass.

Stdlib only (`json`, `subprocess`), so it runs on a bare interpreter like the
rest of the pipeline. Every function returns a list of error strings (empty =
pass), matching the other `_check_*` helpers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REGISTRY_FILE = "scripts/documented_surfaces.json"


def _tracked_rels(base_dir: Path) -> list[str]:
    """Repo-relative POSIX paths of every file that could reach a commit.

    Tracked plus untracked-not-ignored (the same authority the privacy scan
    uses), so a gitignored local directory does not fabricate a class. The
    fallback walk (no git) excludes build scaffolding for the same reason.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        rels = [r for r in out.split() if r and (base_dir / r).exists()]
        if rels:
            return rels
    except (OSError, subprocess.CalledProcessError):
        pass
    # Mirror the repo's gitignore for local-only state, so the no-git fallback
    # does not discover agent worktrees, caches or provider config as component
    # classes (git ls-files already excludes them, and the validator always runs
    # in a git repo — this keeps the fallback honest rather than noisy).
    skip_parts = {"__pycache__", "node_modules", ".git", ".mypy_cache", ".pytest_cache"}
    skip_prefixes = (".kilo/worktrees/", ".kilo/config/")
    return [
        p.relative_to(base_dir).as_posix()
        for p in sorted(base_dir.rglob("*"))
        if p.is_file()
        and not (skip_parts & set(p.parts))
        and not p.relative_to(base_dir).as_posix().startswith(skip_prefixes)
    ]


def discover_classes(base_dir: Path, roots: list[str]) -> list[str]:
    """Depth-1 component directories under each root that hold a file.

    A class is a directory, so a file directly under a root is never one
    (`README.md` is not a class). For the empty-string root the class is the
    first path segment; for a nested root it is `root/<segment>`.
    """
    classes: set[str] = set()
    for rel in _tracked_rels(base_dir):
        parts = rel.split("/")
        for root in roots:
            if root == "":
                if len(parts) >= 2:
                    classes.add(parts[0])
            else:
                rp = root.split("/")
                if parts[: len(rp)] == rp and len(parts) > len(rp) + 1:
                    classes.add("/".join(parts[: len(rp) + 1]))
    return sorted(classes)


def load_surfaces(path: Path | None) -> dict | None:
    """Parse the registry; None on any read or JSON failure (reported as error)."""
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _exempt_errors(exempt: dict) -> list[str]:
    return [
        f"{REGISTRY_FILE}: '{cls}' is exempted from the documented-surfaces "
        f"gate with no real reason -> state why '{cls}' has no documentation "
        f"surface (an empty or stub reason hides the class it should explain)"
        for cls, reason in sorted(exempt.items())
        if not (isinstance(reason, str) and len(reason.strip()) > 20)
    ]


def _declaration_errors(base_dir: Path, surfaces: dict) -> list[str]:
    """A declared surface must be backed by its doc actually naming the class."""
    errors: list[str] = []
    for cls, spec in sorted(surfaces.items()):
        doc = spec.get("doc") if isinstance(spec, dict) else None
        if not doc:
            continue
        path = base_dir / doc
        if not path.is_file():
            continue
        token = (spec.get("token") if isinstance(spec, dict) else None) or cls
        if token not in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(
                f"{REGISTRY_FILE}: '{cls}' is declared documented in {doc}, but "
                f"'{token}' does not appear there -> document '{cls}' in {doc}, "
                f"or correct the registry entry (a declaration whose doc does "
                f"not name the class is a lie)"
            )
    return errors


def _pipeline_stage_errors(base_dir: Path) -> list[str]:
    """Contents check: each public pipeline/*.py stage must be named in the docs.

    Private modules (`_*.py`) are shared helpers, not components — they carry no
    CLI entrypoint and are not run directly, so requiring a docs mention would
    flag implementation detail (the check's scope is the stage surface).
    """
    errors: list[str] = []
    pipeline_dir = base_dir / "pipeline"
    if not pipeline_dir.is_dir():
        return [
            "empty domain: pipeline/ directory missing -> restore it from git "
            "history (a content-driven check over an absent domain must fail, "
            "never report a vacuous OK)"
        ]
    stages = sorted(p for p in pipeline_dir.glob("*.py") if not p.name.startswith("_"))
    if not stages:
        return [
            "empty domain: pipeline/*.py is empty -> restore the stage scripts from git history"
        ]
    docs_text = ""
    readme = base_dir / "README.md"
    if readme.exists():
        docs_text += readme.read_text(encoding="utf-8", errors="replace")
    for md in sorted((base_dir / "docs").glob("*.md")):
        docs_text += "\n" + md.read_text(encoding="utf-8", errors="replace")
    launch_text = ""
    launch = base_dir / ".vscode" / "launch.json"
    if launch.exists():
        launch_text = launch.read_text(encoding="utf-8-sig", errors="replace")
    for py in stages:
        stem = py.stem
        if stem not in docs_text and stem not in launch_text:
            errors.append(
                f"undocumented pipeline component: pipeline/{py.name} is named "
                f"neither in .vscode/launch.json nor in README/docs -> add a "
                f"launch config or a docs mention (undocumented components "
                f"mask their own future evolution)"
            )
    return errors


def surface_errors(base_dir: Path, registry: dict | None) -> list[str]:
    """Every discovered component class must be declared in the registry.

    An absent or empty registry reports rather than passing over nothing, so a
    deleted registry cannot turn the gate green.
    """
    if registry is None:
        return [
            f"{REGISTRY_FILE}: absent or unparsable -> a new component class can "
            f"appear with nothing comparing it to the docs; restore or recreate "
            f"scripts/documented_surfaces.json"
        ]
    roots = registry.get("roots") or []
    surfaces = registry.get("surfaces") or {}
    exempt = registry.get("exempt") or {}
    if not roots or not surfaces:
        return [
            f"{REGISTRY_FILE}: declares no roots or no surfaces -> the "
            f"discovery gate guards nothing; declare the roots to scan and one "
            f"surface per class"
        ]
    errors: list[str] = []
    for cls in discover_classes(base_dir, roots):
        if cls in surfaces or cls in exempt:
            continue
        errors.append(
            f"undeclared component class: '{cls}' exists but is not declared in "
            f"{REGISTRY_FILE} -> add '{cls}' to `surfaces` with the doc that "
            f"names it, or to `exempt` with a reason (a new class of component "
            f"must not enter the repository with no document naming it)"
        )
    errors.extend(_exempt_errors(exempt))
    errors.extend(_declaration_errors(base_dir, surfaces))
    return errors


def component_docs_errors(base_dir: Path) -> list[str]:
    """The full component-docs check: stage docs + the discovery gate.

    Entry point for `validate_harness._check_component_docs` (kept a thin
    wrapper so this module owns the logic and the validator does not grow).
    """
    registry = load_surfaces(base_dir / REGISTRY_FILE)
    return _pipeline_stage_errors(base_dir) + surface_errors(base_dir, registry)
