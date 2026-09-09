#!/usr/bin/env python3
"""Harness validation — deterministic checks for the AI-agent harness files.

Complements the objective gate (pipeline + golden snapshot) with cheap,
CPU-only structural checks that turn judgment-only rules from AGENTS.md into
mechanical gates. Each failure message states WHAT failed and HOW to fix it
(remediation-injecting errors; every gate must be runnable, not remembered).

Checks:
1. Markdown links in README.md + docs/*.md resolve to existing files/anchors.
2. Harness files exist: AGENTS.md, .kilo/skills/<referenced>, .kilo/command/,
   kilo.json, docs/6-HARNESS.md.
3. Skills referenced by AGENTS.md exist in .kilo/skills/.
4. No Claude/Copilot harness residue (CLAUDE.md, .claude/, copilot files).
5. God-file ratchet: tracked .py files must not grow past 600 lines; any file
   listed in GOD_FILE_BASELINES must not grow from its recorded size.
6. .vscode/launch.json program paths point at existing files.
7. kilo.json + .kilo local-state exclusions present.

Usage:
    uv run python scripts/validate_harness.py

Exit codes:
    0 = no errors (warnings do not fail validation)
    1 = errors detected
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# God-file ratchet (lines). Files listed here are tracked debt: they must not
# GROW past their recorded line count; shrinking updates the baseline.
# 600 is the hard cap for any tracked .py file (AGENTS.md § Objective
# verification). Measure with wc -l, record the number, fix the message.
GOD_FILE_BASELINES: dict[str, int] = {
    "dardcollect/quality.py": 507,
    # debt registered 2026-09-08 when the validator first caught it at 642
    # lines (wide utility module, 22 importers — split is its own chunk).
    "dardcollect/pipeline_utils.py": 642,
}
GOD_FILE_HARD_CAP = 600

# Residue patterns from the removed Claude/Copilot harness.
RESIDUE_FILES = [
    "CLAUDE.md",
    ".claude",
    ".github/copilot-instructions.md",
    ".github/copilot-tab-rules.md",
    ".github/instructions",
    ".github/prompts",
]

HARNESS_REQUIRED = [
    "AGENTS.md",
    "kilo.json",
    "docs/6-HARNESS.md",
    ".kilo/.gitignore",
    ".kilo/command/refactor-loop.md",
    ".kilo/FEATURE_WORKFLOW.md",
    ".kilo/skills/socraticode-index-first/SKILL.md",
    ".kilo/skills/refactor-to-objective/SKILL.md",
    ".kilo/skills/keep-docs-navigable/SKILL.md",
]


def _check_markdown_links() -> list[str]:
    """Every relative link in README.md + docs/*.md must resolve on disk."""
    errors: list[str] = []
    files = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]
    link_re = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in link_re.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (f.parent / target).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken link in {f.relative_to(REPO_ROOT)}: ({target}) "
                    f"-> fix the path or restore the target file"
                )
    return errors


def _check_harness_files() -> list[str]:
    """Required harness files must exist."""
    errors: list[str] = []
    for rel in HARNESS_REQUIRED:
        if not (REPO_ROOT / rel).exists():
            errors.append(
                f"missing harness file: {rel} -> restore it from git history "
                f"(git log --diff-filter=D -- {rel}) or recreate it"
            )
    return errors


def _check_agent_skills_reference() -> list[str]:
    """Skills named in AGENTS.md must exist in .kilo/skills/."""
    errors: list[str] = []
    agents = REPO_ROOT / "AGENTS.md"
    if not agents.exists():
        return ["missing harness file: AGENTS.md -> restore from git history"]
    text = agents.read_text(encoding="utf-8", errors="replace")
    for name in re.findall(r"`([a-z0-9-]+)` skill", text):
        if not (REPO_ROOT / ".kilo" / "skills" / name / "SKILL.md").exists():
            errors.append(
                f"AGENTS.md references skill '{name}' but "
                f".kilo/skills/{name}/SKILL.md is missing -> create it or fix "
                f"the reference"
            )
    return errors


def _check_residue() -> list[str]:
    """The old Claude/Copilot harness must stay removed."""
    errors: list[str] = []
    for rel in RESIDUE_FILES:
        if (REPO_ROOT / rel).exists():
            errors.append(
                f"obsolete harness residue present: {rel} -> delete it "
                f"(the harness is Kilo Code: AGENTS.md + .kilo/ + kilo.json)"
            )
    md_files = list(REPO_ROOT.glob("*.md")) + list((REPO_ROOT / "docs").glob("*.md"))
    for f in md_files:
        if f.name in ("AGENTS.md", "README.md", "6-HARNESS.md"):
            # Harness self-documentation may legitimately mention the retired
            # harnesses; existence checks above still catch real leftovers.
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bCLAUDE\.md\b|\.claude/|copilot", text, re.IGNORECASE):
            errors.append(
                f"obsolete Claude/Copilot reference in {f.name} -> update it "
                f"to AGENTS.md / .kilo/ / Kilo Code"
            )
    return errors


def _check_god_files() -> list[str]:
    """Tracked .py files must respect the size ratchet."""
    errors: list[str] = []
    for rel, baseline in sorted(GOD_FILE_BASELINES.items()):
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        if lines > baseline:
            errors.append(
                f"god-file grew: {rel} is {lines} lines (baseline {baseline}) "
                f"-> shrink it (extract coherent units) and lower the "
                f"GOD_FILE_BASELINES entry in scripts/validate_harness.py"
            )
        elif lines < baseline:
            print(
                f"[validate_harness] note: {rel} shrank to {lines} lines "
                f"(baseline {baseline}); lower its GOD_FILE_BASELINES entry",
                flush=True,
            )
    for py in sorted(REPO_ROOT.glob("dardcollect/*.py")) + sorted(
        (REPO_ROOT / "scripts").glob("*.py")
    ):
        rel = py.relative_to(REPO_ROOT).as_posix()
        if rel in GOD_FILE_BASELINES:
            continue
        lines = len(py.read_text(encoding="utf-8", errors="replace").splitlines())
        if lines > GOD_FILE_HARD_CAP:
            errors.append(
                f"god-file: {rel} is {lines} lines (cap {GOD_FILE_HARD_CAP}) "
                f"-> split it (extract coherent units) or, if intrinsically a "
                f"dispatcher, add it to GOD_FILE_BASELINES with user approval"
            )
    return errors


def _check_launch_json() -> list[str]:
    """launch.json program paths must point at existing files."""
    errors: list[str] = []
    launch = REPO_ROOT / ".vscode" / "launch.json"
    if not launch.exists():
        return []  # optional file
    text = launch.read_text(encoding="utf-8-sig", errors="replace")
    # Strip // comments (VS Code JSONC) before parsing.
    text = re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in .vscode/launch.json: {exc} -> fix the syntax"]
    for cfg in data.get("configurations", []):
        program = cfg.get("program", "")
        if not program or program == "${file}":
            continue
        if "$" in program:
            continue
        if not (REPO_ROOT / program).exists():
            errors.append(
                f".vscode/launch.json program not found: {program} "
                f"(config '{cfg.get('name', '?')}') -> update the path to "
                f"match pipeline/ + scripts/ reality"
            )
    return errors


def _check_kilo_config() -> list[str]:
    """kilo.json exists, is valid JSON, and .kilo local state is excluded."""
    errors: list[str] = []
    kilo = REPO_ROOT / "kilo.json"
    if not kilo.exists():
        return ["missing harness file: kilo.json -> recreate it (permissions)"]
    try:
        json.loads(kilo.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in kilo.json: {exc} -> fix the syntax"]
    gi = REPO_ROOT / ".kilo" / ".gitignore"
    gi_text = gi.read_text(encoding="utf-8", errors="replace") if gi.exists() else ""
    for needed in ("agent-manager.json", "worktrees/", "__pycache__/"):
        if needed not in gi_text:
            errors.append(
                f".kilo/.gitignore must exclude local state: missing '{needed}' "
                f"-> add it (local Agent Manager state must never be versioned)"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    checks = [
        ("markdown links", _check_markdown_links),
        ("harness files", _check_harness_files),
        ("AGENTS.md skill references", _check_agent_skills_reference),
        ("Claude/Copilot residue", _check_residue),
        ("god-file ratchet", _check_god_files),
        (".vscode/launch.json", _check_launch_json),
        ("kilo config", _check_kilo_config),
    ]
    all_errors: list[tuple[str, list[str]]] = []
    for name, fn in checks:
        errs = fn()
        if errs:
            all_errors.append((name, errs))

    if not all_errors:
        print("[validate_harness] OK: all harness checks passed.")
        return 0

    print(f"[validate_harness] FAILED: {sum(len(e) for _, e in all_errors)} error(s).")
    for name, errs in all_errors:
        print(f"  [{name}]")
        for e in errs:
            print(f"    - {e}")
    print(
        "Fix the errors above (each message states the remediation), then re-run: "
        "uv run python scripts/validate_harness.py"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
