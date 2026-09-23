#!/usr/bin/env python3
"""Harness validation — deterministic checks for the AI-agent harness files.

Complements the objective gate (pipeline + golden snapshot) with cheap,
CPU-only structural checks that turn judgment-only rules from AGENTS.md into
mechanical gates. Each failure message states WHAT failed and HOW to fix it
(remediation-injecting errors; every gate must be runnable, not remembered).

Checks (errors — fatal):
1. Markdown links in README.md + docs/*.md resolve to existing files/anchors.
2. Harness files exist: AGENTS.md, .kilo/skills/<referenced>, .kilo/command/,
   kilo.json, docs/6-HARNESS.md, docs/HARNESS_RULES.md, scripts/cycle_metrics.py.
3. Skills referenced by AGENTS.md exist in .kilo/skills/.
4. No Claude/Copilot harness residue (CLAUDE.md, .claude/, copilot files).
5. God-file ratchet: tracked .py files must not grow past 600 lines; any file
   listed in GOD_FILE_BASELINES must not grow from its recorded size.
6. .vscode/launch.json program paths point at existing files.
7. kilo.json + .kilo local-state exclusions present.
8. Session-state budget: MEMORY.md (live handoff) stays under 40 KB — fatal
   over budget (adapted from the ai-harness-eng harness; without the gate the
   file grows append-only and becomes a fixed per-session context cost).
9. No backward-compatibility shims (AGENTS.md § No backward-compatibility
   shims): the justification-comment markers shims are written with are
   grepped across tracked .py files; every hit is an error unless pinned in
   COMPAT_ALLOWLIST with a user-confirmed reason.
10. Code-quality + dead-code ratchet (scripts/quality_gates.py): C901 > 10,
   functions > 80 lines, PLR0913/0912/0915, unused parameters (`ARG`), bugbear
   `B`, and vulture dead code (>= 60%) are compared against the user-owned
   scripts/quality_baselines.json; a NEW or WORSENED violation fails,
   resolved/improved entries print a note.

Advisory checks (warnings — never fatal, adopted from the ai-harness-eng
harness 2026-09-10):
11. Privacy scan: machine-local path patterns across every tracked text file —
   home-directory paths and drive-absolute literals that are not documented
   examples/install dirs (the repo is public; each hit is reviewed by the user,
   never auto-edited). UTF-16 content is decoded; a text file that stays
   unreadable is reported as unscannable instead of passing silently. The check
   lives in scripts/privacy_scan.py (extracted when this file hit the 600-line
   god-file cap); scope and allowances are documented there.
12. Component-docs sync: every pipeline/*.py stage script is named in
   .vscode/launch.json or README/docs (undocumented components mask their
   own future evolution).

Exit-code contract (stable — hooks depend on it; do not change silently):
    0 = no errors, no warnings
    2 = warnings only (warnings NEVER fail validation; hooks must accept 2)
    1 = at least one error detected
Modes:
    validate_harness.py          full run, print diagnostics
    validate_harness.py --check  quiet mode for the pre-commit hook
                                 (errors go to stderr, exit 1; warnings exit 2)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import privacy_scan
import quality_gates

# God-file ratchet (lines). Files listed here are tracked debt: they must not
# GROW past their recorded line count; shrinking updates the baseline.
# 600 is the hard cap for any tracked .py file (AGENTS.md § Objective
# verification). Measure with wc -l, record the number, fix the message.
GOD_FILE_BASELINES: dict[str, int] = {
    "dardcollect/quality.py": 435,
    # debt registered 2026-09-08 at 642 lines; shrank to 464 on 2026-09-09 when
    # the clip/video writers moved to dardcollect/video_writers.py (issue #8 chunk),
    # then to 454 on 2026-09-23 (video_writers re-export shims removed), to 417
    # when the unused make_output_path/source_subdir_prefix were pruned, and to 402
    # when the orphaned duplicate _cleanup_files was removed.
    "dardcollect/pipeline_utils.py": 402,
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
    "docs/HARNESS_RULES.md",
    "scripts/cycle_metrics.py",
    "scripts/privacy_scan.py",
    "scripts/quality_gates.py",
    "scripts/quality_baselines.json",
    ".kilo/.gitignore",
    ".kilo/command/refactor-loop.md",
    ".kilo/FEATURE_WORKFLOW.md",
    ".kilo/skills/refactor-to-objective/SKILL.md",
    ".kilo/skills/keep-docs-navigable/SKILL.md",
]

# Session-state budget (ai-harness-eng pattern): the live handoff file must
# stay small; the narrative lives in the session chat + git history. User-owned
# constant — changing it is an explicit user edit of this line.
SESSION_STATE = REPO_ROOT / "MEMORY.md"
SESSION_STATE_MAX_BYTES = 40 * 1024


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
    for py in privacy_scan.tracked_files(REPO_ROOT, {".py"}):
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


def _check_quality_ratchet() -> list[str]:
    """No NEW or WORSENED code-quality violation (AGENTS.md code-quality gates).

    The criteria (complexity > 10, functions > `MAX_FUNCTION_LINES`, too many
    args/branches/statements, bugbear antipatterns, dead code) are collected by
    `scripts/quality_gates.py`; the user-owned baseline
    (`scripts/quality_baselines.json`) freezes the current debt, so this gate
    fails only when a violation is introduced or worsened. Resolved/improved
    entries are printed as notes so the baseline does not rot.
    """
    try:
        current = quality_gates.collect(REPO_ROOT)
        errors, notes = quality_gates.compare(current, quality_gates.load_baseline(REPO_ROOT))
    except quality_gates.QualityToolError as exc:
        return [
            f"code-quality ratchet cannot run: {exc} -> install dev deps "
            f"(`uv sync --extra dev`); the quality gate needs ruff and vulture"
        ]
    for note in notes:
        print(f"[validate_harness] note (quality ratchet): {note}", file=sys.stderr, flush=True)
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


def _check_session_state_size() -> list[str]:
    """The live session handoff (MEMORY.md) must stay within its size budget.

    Fatal over budget (the handoff is a fixed per-session context cost);
    advisory at >= 80% so recalibration is visible before the gate fires.
    """
    errors: list[str] = []
    if not SESSION_STATE.exists():
        errors.append(
            "missing session handoff: MEMORY.md -> recreate it "
            "(live handoff format: Where we are / Key decisions / Open items / "
            "Known quirks; see AGENTS.md § Session closure)"
        )
        return errors
    size = SESSION_STATE.stat().st_size
    if size > SESSION_STATE_MAX_BYTES:
        errors.append(
            f"session handoff too large: MEMORY.md is {size} bytes "
            f"(budget {SESSION_STATE_MAX_BYTES}) -> compact older entries "
            f"(full narrative lives in session chat + git history; never "
            f"delete facts silently), then re-run this script"
        )
    elif size >= SESSION_STATE_MAX_BYTES * 0.8:
        print(
            f"[validate_harness] note: MEMORY.md is {size} bytes "
            f"({size * 100 // SESSION_STATE_MAX_BYTES}% of budget) — "
            f"advisory: compact older entries at the next session close",
            file=sys.stderr,
            flush=True,
        )
    return errors


def _check_privacy_scan() -> list[str]:
    """Warning-level: personal-data patterns in committed text files.

    Delegates to scripts/privacy_scan.py (extracted 2026-09-22 when this file
    crossed the god-file cap). See that module for scope and allowances.
    """
    return privacy_scan.scan(REPO_ROOT)


# Marker phrases that shims are habitually justified with (AGENTS.md
# "No backward-compatibility shims"). A hit means a shim may have been added;
# remove the shim and update callers, or pin the line here with a reason.
# This file and its tests mention the phrases as data, so they are excluded
# by exact path below (see COMPAT_SELF_EXCLUDE).
COMPAT_MARKERS = re.compile(
    r"backwards?[-\s]?compat" + r"|kept for (?:API )?compat" + r"|no longer used" + r"|legacy",
    re.IGNORECASE,
)
# Legitimate markers pinned by (repo-relative path, stripped line) with a reason.
# Empty is the healthy state; add an entry only after the user confirms the code
# is a real contract (e.g. documented public API), never to silence a shim.
COMPAT_ALLOWLIST: dict[tuple[str, str], str] = {}
# Files that define/exercise this check mention the phrases as data, not as
# shims: the check's own source and its tests. Exclusion is by exact path.
COMPAT_SELF_EXCLUDE = frozenset({"scripts/validate_harness.py", "tests/test_validate_harness.py"})


def _check_compat_markers() -> list[str]:
    """Flag backward-compatibility shim markers in tracked .py files.

    Enforces AGENTS.md "No backward-compatibility shims": when a symbol is
    renamed/moved, update every caller instead of leaving a compat shim behind.
    The markers are the justification comments such shims are written with;
    a hit is an error unless pinned in COMPAT_ALLOWLIST with a reason.

    File discovery is shared with the privacy scan (`privacy_scan.tracked_files`)
    so the two checks cannot drift on "which committed files are scanned" or on
    NUL-safe path handling.
    """
    errors: list[str] = []
    for p in privacy_scan.tracked_files(REPO_ROOT, {".py"}):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel in COMPAT_SELF_EXCLUDE:
            continue
        if not p.exists():
            continue
        for lineno, line in enumerate(
            p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if not COMPAT_MARKERS.search(line):
                continue
            if (rel, line.strip()) in COMPAT_ALLOWLIST:
                continue
            errors.append(
                f"backward-compat shim marker in {rel}:{lineno} -> {line.strip()!r}; "
                f"remove the shim and update every caller (AGENTS.md: No "
                f"backward-compatibility shims), or pin it in COMPAT_ALLOWLIST "
                f"with a user-confirmed reason"
            )
    return errors


# Every pipeline stage script must be reachable from the documented surface:
# named in a launch.json debug config or in README/docs (undocumented
# components mask their own future evolution).
def _check_component_docs() -> list[str]:
    """Each pipeline/*.py stage must be named in launch.json or the docs."""
    errors: list[str] = []
    docs_text = ""
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        docs_text += readme.read_text(encoding="utf-8", errors="replace")
    for md in sorted((REPO_ROOT / "docs").glob("*.md")):
        docs_text += "\n" + md.read_text(encoding="utf-8", errors="replace")
    launch_text = ""
    launch = REPO_ROOT / ".vscode" / "launch.json"
    if launch.exists():
        launch_text = launch.read_text(encoding="utf-8-sig", errors="replace")
    for py in sorted((REPO_ROOT / "pipeline").glob("*.py")):
        stem = py.stem
        if stem not in docs_text and stem not in launch_text:
            errors.append(
                f"undocumented pipeline component: pipeline/{py.name} is named "
                f"neither in .vscode/launch.json nor in README/docs -> add a "
                f"launch config or a docs mention (undocumented components "
                f"mask their own future evolution)"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    check_mode = "--check" in (argv if argv is not None else sys.argv[1:])
    checks = [
        ("markdown links", _check_markdown_links),
        ("harness files", _check_harness_files),
        ("AGENTS.md skill references", _check_agent_skills_reference),
        ("Claude/Copilot residue", _check_residue),
        ("god-file ratchet", _check_god_files),
        (".vscode/launch.json", _check_launch_json),
        ("kilo config", _check_kilo_config),
        ("session-state budget", _check_session_state_size),
        ("backward-compat shims", _check_compat_markers),
        ("code-quality ratchet", _check_quality_ratchet),
    ]
    all_errors: list[tuple[str, list[str]]] = []
    all_warnings: list[tuple[str, list[str]]] = []
    for name, fn in checks:
        errs = fn()
        if errs:
            all_errors.append((name, errs))
    for w in _check_privacy_scan():
        all_warnings.append(("privacy scan", [w]))
    for w in _check_component_docs():
        all_warnings.append(("component-docs sync", [w]))

    if check_mode:
        # Quiet mode for the pre-commit hook: only failures are printed.
        if all_errors:
            for name, errs in all_errors:
                for e in errs:
                    print(f"[validate_harness] {name}: {e}", file=sys.stderr)
            return 1
        return 2 if all_warnings else 0

    for name, warns in all_warnings:
        print(f"[validate_harness] WARNING [{name}]")
        for w in warns:
            print(f"    - {w}")

    if not all_errors:
        n = sum(len(w) for _, w in all_warnings)
        print(
            "[validate_harness] OK: all harness checks passed."
            + (f" ({n} warning(s), advisory only.)" if n else "")
        )
        # Exit-code contract: warnings-only = 2 (hooks must accept 2).
        return 2 if all_warnings else 0

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
