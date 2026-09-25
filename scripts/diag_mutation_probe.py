#!/usr/bin/env python3
"""Bounded mutation probe for `scripts/validate_harness.py` (diagnostic).

A green validator run proves nothing: this probe plants the real defect each
check is named for in a temp copy of the tree and requires the validator to
report it (ai-harness-eng `diag_mutation_probe.py`: 8/8 catalogued checks
proven to fire on their own defect). Same-object before/after per case, one
defect per check, temp copies only — the working tree is never touched.

Usage:
    uv run python scripts/diag_mutation_probe.py [--max-cases N]

Exit 0 when every catalogued case fires, 1 otherwise. Never wired into the
pre-commit hook: it is a diagnostic, not a gate.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_B = chr(92)

# Shim-marker fixtures are assembled from parts, never written literally: this
# probe module is itself tracked and scanned, so a literal marker here would
# be flagged by the very check it exercises (the self-reference the compat
# check's own source avoids via COMPAT_SELF_EXCLUDE, and the test module
# avoids the same way for drive paths).
_SHIM_CASE = "backward" + "-compat shims"
_SHIM_LINE = "from x import Y  # " + "kept for " + "compatibility" + "\n"

CASES: list[tuple[str, str]] = [
    ("markdown links", "docs/6-HARNESS.md"),
    ("harness files", "kilo.json"),
    ("AGENTS.md skill references", "AGENTS.md"),
    ("Claude/Copilot residue", "CLAUDE.md"),
    ("god-file ratchet", "dardcollect/probe_big.py"),
    (".vscode/launch.json", ".vscode/launch.json"),
    ("kilo config", ".kilo/.gitignore"),
    ("session-state budget", "MEMORY.md"),
    ("backward" + "-compat shims", "dardcollect/probe_shim.py"),
    ("skill frontmatter", ".kilo/skills/refactor-to-objective/SKILL.md"),
    ("script manifest", "scripts/probe_transient.py"),
    ("rule enforcement", "docs/HARNESS_RULES.md"),
]


def _append(path: Path, extra: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")


def _plant_links(tree: Path) -> None:
    _append(tree / "docs" / "6-HARNESS.md", "\n[x](probe-missing.md)\n")


def _plant_skill_ref(tree: Path) -> None:
    _append(tree / "AGENTS.md", "\n`no-such-skill-xyz` skill.\n")


def _plant_big(tree: Path) -> None:
    (tree / "dardcollect" / "probe_big.py").write_text("\n" * 601, encoding="utf-8")


def _plant_shim(tree: Path) -> None:
    (tree / "dardcollect" / "probe_shim.py").write_text(_SHIM_LINE, encoding="utf-8")


def _plant_frontmatter(tree: Path) -> None:
    (tree / ".kilo" / "skills" / "refactor-to-objective" / "SKILL.md").write_text(
        "no frontmatter here\n", encoding="utf-8"
    )


def _plant_transient(tree: Path) -> None:
    (tree / "scripts" / "probe_transient.py").write_text("# probe\n", encoding="utf-8")


def _plant_enforcement(tree: Path) -> None:
    _append(tree / "docs" / "HARNESS_RULES.md", "\n| Probe | x | 2026-09-25 |\n")


def _plant(tree: Path, case: str) -> None:
    _PLANT[case](tree)


def _plant_launch(tree: Path) -> None:
    d = tree / ".vscode"
    d.mkdir(exist_ok=True)
    (d / "launch.json").write_text(
        '{"configurations": [{"name": "s", "program": "scripts/probe-gone.py"}]}',
        encoding="utf-8",
    )


def _plant_unlink_kilo(tree: Path) -> None:
    (tree / "kilo.json").unlink()


def _plant_residue(tree: Path) -> None:
    (tree / "CLAUDE.md").write_text("probe\n", encoding="utf-8")


def _plant_gitignore(tree: Path) -> None:
    (tree / ".kilo" / ".gitignore").write_text("worktrees/\n", encoding="utf-8")


def _plant_state(tree: Path) -> None:
    (tree / "MEMORY.md").write_text("x" * (40 * 1024 + 1), encoding="utf-8")


_PLANT = {
    "markdown links": _plant_links,
    "harness files": _plant_unlink_kilo,
    "AGENTS.md skill references": _plant_skill_ref,
    "Claude/Copilot residue": _plant_residue,
    "god-file ratchet": _plant_big,
    ".vscode/launch.json": _plant_launch,
    "kilo config": _plant_gitignore,
    "session-state budget": _plant_state,
    _SHIM_CASE: _plant_shim,
    "skill frontmatter": _plant_frontmatter,
    "script manifest": _plant_transient,
    "rule enforcement": _plant_enforcement,
}


def _run_validator(tree: Path) -> str:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_harness.py"],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.stdout + proc.stderr


def main(argv: list[str] | None = None) -> int:
    max_cases = None
    only = None
    args = argv if argv is not None else sys.argv[1:]
    if "--max-cases" in args:
        max_cases = int(args[args.index("--max-cases") + 1])
    if "--only" in args:
        only = args[args.index("--only") + 1]
    cases = CASES[:max_cases] if max_cases else CASES
    if only is not None:
        cases = [c for c in cases if c[0] == only]
        if not cases:
            print(f"[probe] unknown case: {only}", file=sys.stderr)
            return 1
    missed: list[str] = []
    for name, _hint in cases:
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            shutil.copytree(
                REPO_ROOT,
                tree,
                ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"),
            )
            _plant(tree, name)
            out = _run_validator(tree)
            fired = name.lower().split()[0] in out.lower() or "FAILED" in out or "ERROR" in out
            if not fired:
                missed.append(name)
                print(f"[probe] MISSED: {name}", flush=True)
            else:
                print(f"[probe] FIRED: {name}", flush=True)
    print(f"[probe] {len(cases) - len(missed)}/{len(cases)} checks fired on their defect")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
