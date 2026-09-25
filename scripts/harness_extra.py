#!/usr/bin/env python3
"""Extra harness checks for `scripts/validate_harness.py` (fatal).

Extracted into this module so `validate_harness.py` stays under the 600-line
god-file cap (AGENTS.md § Objective verification). Each function takes
REPO_ROOT and returns a list of error strings (empty = pass). Every failure
message states the remediation (remediation-injecting errors).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Permanent pipeline scripts. Any `scripts/*.py` outside this manifest (and
# outside `diag_*` diagnostics) is a transient leftover that must be deleted
# or promoted in the same cycle (ai-harness-eng transient-scripts lifecycle).
SCRIPT_MANIFEST = frozenset(
    {
        "scripts/benchmark_pipeline.py",
        "scripts/cycle_metrics.py",
        "scripts/diag_mutation_probe.py",
        "scripts/golden_snapshot.py",
        "scripts/harness_extra.py",
        "scripts/license_scan.py",
        "scripts/make_fixture_media.py",
        "scripts/make_test_config.py",
        "scripts/objective_gate.py",
        "scripts/privacy_scan.py",
        "scripts/quality_gates.py",
        "scripts/reclaim_processed_sources.py",
        "scripts/redownload_sources.py",
        "scripts/run_pipeline.py",
        "scripts/validate_harness.py",
    }
)

# Enforcement vocabulary for docs/HARNESS_RULES.md (ai-harness-eng rule_index
# Enforcement column). Every rule row must name the artifact that fails on
# violation, or `advisory` when only prose enforces it.
ENFORCEMENT_VOCAB = frozenset(
    {
        "validate_harness.py",
        "quality_gates.py",
        "privacy_scan.py",
        "license_scan.py",
        "objective_gate.py",
        "golden_snapshot.py",
        "pre-commit hook",
        "regression tests",
        "advisory",
    }
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str | None]:
    """Parse a `---` YAML frontmatter block strictly (no dependency).

    Returns (fields, error). Strict: block must open with `---`, close with
    `---`, every content line must be `key: value` with a non-empty value.
    An unquoted `: ` inside a value is rejected (the silent-loss class: a
    permissive reader accepts it locally while packaging boundaries drop it).
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "missing opening `---`"
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, "missing closing `---`"
    fields: dict[str, str] = {}
    for lineno, line in enumerate(lines[1:end], 2):
        if not line.strip():
            continue
        if ":" not in line or line.startswith((" ", "\t")):
            return {}, f"line {lineno} is not `key: value`: {line.strip()!r}"
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key or not value:
            return {}, f"line {lineno} has empty key or value: {line.strip()!r}"
        if ": " in value and not (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            return {}, f"line {lineno} has unquoted `: ` inside the value: {line.strip()!r}"
        fields[key] = value
    return fields, None


def check_skill_frontmatter(repo_root: Path) -> list[str]:
    """Every `.kilo/skills/*/SKILL.md` parses strictly with name == directory."""
    errors: list[str] = []
    skills_dir = repo_root / ".kilo" / "skills"
    if not skills_dir.is_dir():
        return ["no `.kilo/skills/` directory -> restore it from git history"]
    for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_file = skill_dir / "SKILL.md"
        rel = skill_file.relative_to(repo_root).as_posix()
        if not skill_file.exists():
            errors.append(
                f"skill without SKILL.md: {skill_dir.name}/ -> create {rel} or delete the directory"
            )
            continue
        fields, err = _parse_frontmatter(skill_file.read_text(encoding="utf-8", errors="replace"))
        if err is not None:
            errors.append(f"invalid skill frontmatter in {rel}: {err} -> fix the block")
            continue
        for needed in ("name", "description"):
            if needed not in fields:
                errors.append(
                    f"invalid skill frontmatter in {rel}: missing `{needed}` "
                    f"-> add it to the `---` block"
                )
        if "name" in fields and fields["name"] != skill_dir.name:
            errors.append(
                f"invalid skill frontmatter in {rel}: `name: {fields['name']}` "
                f"does not match its directory `{skill_dir.name}` -> rename one of them"
            )
    return errors


def check_script_manifest(repo_root: Path) -> list[str]:
    """Every `scripts/*.py` is in SCRIPT_MANIFEST or is a `diag_*` diagnostic."""
    errors: list[str] = []
    scripts_dir = repo_root / "scripts"
    if not scripts_dir.is_dir():
        return errors
    for py in sorted(scripts_dir.glob("*.py")):
        rel = py.relative_to(repo_root).as_posix()
        if rel in SCRIPT_MANIFEST or py.name.startswith("diag_"):
            continue
        errors.append(
            f"transient script outside the manifest: {rel} -> delete it in the "
            f"same cycle or promote it by adding it to SCRIPT_MANIFEST in "
            f"scripts/harness_extra.py"
        )
    return errors


def check_rule_enforcement(repo_root: Path) -> list[str]:
    """Every `docs/HARNESS_RULES.md` row carries an Enforcement cell from vocab."""
    errors: list[str] = []
    rules = repo_root / "docs" / "HARNESS_RULES.md"
    if not rules.exists():
        return ["missing harness file: docs/HARNESS_RULES.md -> restore from git history"]
    lines = rules.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [ln for ln in lines if ln.startswith("|") and not ln.startswith("| Rule")]
    if not rows:
        return ["docs/HARNESS_RULES.md has no rule rows -> restore it from git history"]
    for lineno, row in enumerate(lines, 1):
        if not row.startswith("|") or row.startswith("| Rule") or row.startswith("|---"):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 4:
            errors.append(
                f"docs/HARNESS_RULES.md:{lineno} has {len(cells)} cells, need 4 "
                f"(Rule | Failure | Date | Enforcement) -> add the Enforcement cell"
            )
            continue
        enforcement = cells[3].split("(")[0].strip().strip("`")
        if enforcement not in ENFORCEMENT_VOCAB:
            errors.append(
                f"docs/HARNESS_RULES.md:{lineno} Enforcement {cells[3]!r} is outside "
                f"the vocabulary {sorted(ENFORCEMENT_VOCAB)} -> use a fixed value "
                f"(`advisory` when only prose enforces the rule)"
            )
    return errors


def check_volatile_numbers(repo_root: Path, total: int) -> list[str]:
    """Live docs must cite the live check total, never a stale number."""
    errors: list[str] = []
    live = [
        repo_root / "README.md",
        repo_root / "AGENTS.md",
        repo_root / "docs" / "6-HARNESS.md",
    ]
    phrase_re = re.compile(r"(?<![/\d])(\d+)\s+checks\b")
    for f in live:
        if not f.exists():
            continue
        rel = f.relative_to(repo_root).as_posix()
        for lineno, line in enumerate(
            f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for m in phrase_re.finditer(line):
                if int(m.group(1)) != total:
                    errors.append(
                        f"stale check total in {rel}:{lineno}: says {m.group(1)} "
                        f"checks but the runner has {total} -> update the number "
                        f"or rephrase the claim"
                    )
    return errors


def check_launch_json(repo_root: Path) -> list[str]:
    """launch.json program paths must point at existing files."""
    errors: list[str] = []
    launch = repo_root / ".vscode" / "launch.json"
    if not launch.exists():
        return []  # optional file
    text = launch.read_text(encoding="utf-8-sig", errors="replace")
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
        if not (repo_root / program).exists():
            errors.append(
                f".vscode/launch.json program not found: {program} "
                f"(config '{cfg.get('name', '?')}') -> update the path to "
                f"match pipeline/ + scripts/ reality"
            )
    return errors


def check_kilo_config(repo_root: Path) -> list[str]:
    """kilo.json exists, is valid JSON, and .kilo local state is excluded."""
    errors: list[str] = []
    kilo = repo_root / "kilo.json"
    if not kilo.exists():
        return ["missing harness file: kilo.json -> recreate it (permissions)"]
    try:
        json.loads(kilo.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in kilo.json: {exc} -> fix the syntax"]
    gi = repo_root / ".kilo" / ".gitignore"
    gi_text = gi.read_text(encoding="utf-8", errors="replace") if gi.exists() else ""
    for needed in ("agent-manager.json", "worktrees/", "__pycache__/"):
        if needed not in gi_text:
            errors.append(
                f".kilo/.gitignore must exclude local state: missing '{needed}' "
                f"-> add it (local Agent Manager state must never be versioned)"
            )
    return errors


def check_session_state_size(repo_root: Path, max_bytes: int) -> list[str]:
    """The live session handoff (MEMORY.md) must stay within its size budget."""
    errors: list[str] = []
    state = repo_root / "MEMORY.md"
    if not state.exists():
        return [
            "missing session handoff: MEMORY.md -> recreate it "
            "(live handoff format: Where we are / Key decisions / Open items / "
            "Known quirks; see AGENTS.md § Session closure)"
        ]
    size = state.stat().st_size
    if size > max_bytes:
        errors.append(
            f"session handoff too large: MEMORY.md is {size} bytes "
            f"(budget {max_bytes}) -> compact older entries "
            f"(full narrative lives in session chat + git history; never "
            f"delete facts silently), then re-run this script"
        )
    elif size >= max_bytes * 0.8:
        print(
            f"[validate_harness] note: MEMORY.md is {size} bytes "
            f"({size * 100 // max_bytes}% of budget) — "
            f"advisory: compact older entries at the next session close",
            file=sys.stderr,
            flush=True,
        )
    return errors
