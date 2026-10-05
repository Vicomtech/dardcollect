#!/usr/bin/env python3
"""Mechanical code-quality ratchet (used by `scripts/validate_harness.py`).

AGENTS.md declares code-quality criteria (cyclomatic complexity, function
length) and "dead code pruned", but until 2026-09-23 nothing enforced them:
`C901` was not in ruff's `select`, there was no function-length check, and no
dead-code tool was wired at all. This module turns those criteria into one
mechanical ratchet.

There is **no anonymous frozen debt**. Every tolerated violation must be an
**explicit, reasoned exception** that is itself part of the rules: the registry
(`scripts/quality_baselines.json`) maps each exception to a rule id in
``EXCEPTION_RULES`` below, and a rule may only exist when it states a real design
reason. The gate fails when a violation is NEW, WORSENED, cites an unknown rule,
or when an exception is stale (no longer matches a violation — remove it). A
violation with no justified exception must be fixed, not waived.

Metrics and their baseline keys (``"<file>|<symbol>"``):

  - ``c901``    ruff C901, cyclomatic complexity > 10   (value = complexity)
  - ``plr0913`` ruff too many arguments > 5             (value = argument count)
  - ``plr0912`` ruff too many branches > 12             (value = branch count)
  - ``plr0915`` ruff too many statements > 50           (value = statement count)
  - ``fnlen``   function/method longer than 80 lines    (value = line count)
  - ``bugbear`` ruff ``B`` antipatterns (flake8-bugbear) (value = 1)
  - ``unusedarg`` ruff ``ARG`` unused parameters             (value = 1)
  - ``deadcode`` vulture unused symbol >= 60% confidence (value = 1)

Scope is the tracked ``.py`` set (via ``privacy_scan.tracked_files``), the same
authority the privacy and shim checks use.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import privacy_scan

BASELINE_NAME = "scripts/quality_baselines.json"
# Thresholds live in pyproject.toml ([tool.ruff.lint.mccabe], per-file-ignores)
# so plain ruff runs and the ratchet share one source of truth.
MAX_FUNCTION_LINES = 80
VULTURE_CONFIDENCE = 60
RUFF_SELECT = "C901,PLR0913,PLR0912,PLR0915,B,ARG"

# ── Codified exceptions ───────────────────────────────────────────────────────
# THE RULE: a tolerated violation is only legitimate when a *named rule* below
# justifies it. An entry in the baseline that no rule matches is an error, so
# debt can never be frozen anonymously — either fix the violation, or add a rule
# that states a real design reason and matches it. Each rule is
# ``rule_id: (regex over "<metric>|<file>|<symbol>", reason)``.
#
# The registry is EMPTY: the 2026-09-25 zero-debt pass fixed all 30 violations
# (request/state dataclasses + extracted phase helpers), so every historic rule
# was removed. Keep it empty — a new entry needs a real design reason, and
# "it is pre-existing" is not one.
EXCEPTION_RULES: dict[str, tuple[str, str]] = {}


def _rule_for(metric: str, key: str) -> str | None:
    """Rule id whose pattern matches ``<metric>|<key>``, or None if uncodified."""
    probe = f"{metric}|{key}"
    for rule_id, (pattern, _reason) in EXCEPTION_RULES.items():
        if re.match(pattern, probe):
            return rule_id
    return None


# ruff code -> metric name. Bugbear is matched by prefix (see _metric_for_code),
# not enumerated, so every B code is covered — enumerating let new antipatterns
# (B006 mutable defaults, ...) slip through while the gate claimed to cover `B`.
RUFF_METRICS = {
    "C901": "c901",
    "PLR0913": "plr0913",
    "PLR0912": "plr0912",
    "PLR0915": "plr0915",
}
# Messages of the form "(19 > 10)" carry the numeric severity in group 1.
_SEVERITY = re.compile(r"\((\d+)\s*>\s*\d+\)")
_DEF_NAME = re.compile(r"\bdef\s+(\w+)")
_VULTURE_LINE = re.compile(r"^(?P<file>.+?):(?P<line>\d+): unused \w+ '(?P<name>[^']+)'")


def _metric_for_code(code: str) -> str | None:
    """Map a ruff code to its ratchet metric (bugbear/ARG by prefix, else explicit)."""
    if code.startswith("B"):
        return "bugbear"
    if code.startswith("ARG"):
        return "unusedarg"
    return RUFF_METRICS.get(code)


class QualityToolError(RuntimeError):
    """A required analysis tool could not be run."""


def _relpath(path: str | Path, repo_root: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _run_json(cmd: list[str], repo_root: Path) -> list[dict]:
    """Run a command that prints a JSON array; raise clearly if it cannot run."""
    try:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    except OSError as exc:  # tool not installed / not executable
        raise QualityToolError(f"cannot run {cmd[0]}: {exc}") from exc
    if not proc.stdout.strip():
        raise QualityToolError(f"{cmd[0]} produced no output; stderr: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise QualityToolError(f"{cmd[0]} did not emit JSON: {exc}") from exc


def _severity(message: str) -> int:
    match = _SEVERITY.search(message)
    return int(match.group(1)) if match else 1


def _def_name_at(path: Path, row: int) -> str:
    """Function/method name on *row* (ruff PLR rows point at the `def` line)."""
    try:
        line = path.read_text(encoding="utf-8", errors="replace").splitlines()[row - 1]
    except (OSError, IndexError):
        return f"<line {row}>"
    match = _DEF_NAME.search(line)
    return match.group(1) if match else f"<line {row}>"


def _ruff_findings(repo_root: Path) -> dict[str, dict[str, int]]:
    """ruff C901 / PLR / B findings, keyed by metric then `<file>|<symbol>`."""
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        ".",
        "--select",
        RUFF_SELECT,
        "--no-cache",
        "--output-format",
        "json",
    ]
    findings: dict[str, dict[str, int]] = {}
    for item in _run_json(cmd, repo_root):
        metric = _metric_for_code(item["code"])
        if metric is None:
            continue
        rel = _relpath(item["filename"], repo_root)
        path = repo_root / rel
        row = item["location"]["row"]
        if metric in ("bugbear", "unusedarg"):
            symbol = _enclosing_symbol(path, row)
            value = 1
            if metric == "unusedarg":
                arg = _unused_arg_name(item["message"])
                symbol = f"{symbol}({arg})" if arg else symbol
        else:
            symbol = _enclosing_symbol(path, row)
            value = _severity(item["message"])
        key = f"{rel}|{symbol}"
        bucket = findings.setdefault(metric, {})
        bucket[key] = max(bucket.get(key, 0), value)
    return findings


_UNUSED_ARG = re.compile(r"Unused (?:function|method|lambda) argument: `([^`]+)`")


def _unused_arg_name(message: str) -> str:
    """Argument name from a ruff ARG message (e.g. "Unused function argument: `x`")."""
    match = _UNUSED_ARG.search(message)
    return match.group(1) if match else ""


def _enclosing_symbol(path: Path, row: int) -> str:
    """Name of the innermost function/method containing *row*.

    Used for both PLR rows (which point at the `def` line) and bugbear rows
    (which point at an arbitrary line inside the function). Keying on the
    enclosing function — not the line number/text — keeps the baseline stable
    when unrelated edits shift lines, so the ratchet does not fire on cosmetic
    diffs. Module-level hits fall back to the source line text.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"<line {row}>"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _def_name_at(path, row)
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.end_lineno is None:
            continue
        if node.lineno <= row <= node.end_lineno:
            # Innermost = the one whose start line is largest.
            if best is None or node.lineno > best[0]:
                best = (node.lineno, node.name)
    if best is not None:
        return best[1]
    # Module-level hit: keep the line text so the key stays human-readable.
    try:
        line = source.splitlines()[row - 1]
    except IndexError:
        return f"<line {row}>"
    return f"line {row}: {line.strip()[:40]}"


def _function_lengths(repo_root: Path) -> dict[str, dict[str, int]]:
    """Functions whose *code* exceeds MAX_FUNCTION_LINES, keyed by file|symbol.

    Length is measured **excluding the leading docstring**: this gate targets
    logic concentration (see the refactor-to-objective skill: "the god-method
    gate is about logic concentration, not raw line count"), and a long
    docstring is the documentation this repo mandates, not concentrated logic.
    Counting it would push well-documented functions over the cap and pressure
    authors to delete rationale. Bodies are otherwise counted verbatim.
    """
    findings: dict[str, dict[str, int]] = {}
    for path in privacy_scan.tracked_files(repo_root, {".py"}):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        rel = path.relative_to(repo_root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.end_lineno is None:
                continue
            code_start = node.lineno
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                and body[0].end_lineno is not None
            ):
                code_start = body[0].end_lineno + 1
            length = node.end_lineno - code_start + 1
            if length > MAX_FUNCTION_LINES:
                findings.setdefault("fnlen", {})[f"{rel}|{node.name}"] = length
    return findings


def _vulture_targets(repo_root: Path) -> list[str]:
    """Top-level paths containing tracked .py files (never .venv, never tests/).

    ``tests/`` is excluded: pytest discovers tests and resolves fixtures by name
    at runtime, so vulture cannot see the "caller" of a test function or a
    fixture/helper and would report them as dead. That is dynamic injection, not
    dead code.
    """
    targets: set[str] = set()
    for path in privacy_scan.tracked_files(repo_root, {".py"}):
        rel = path.relative_to(repo_root)
        if rel.parts[0] == "tests":
            continue
        targets.add(rel.parts[0] if len(rel.parts) > 1 else rel.name)
    return sorted(targets)


# Symbols vulture cannot see a caller for, but that are real contracts, keyed by
# "<repo-relative path>|<symbol>". Each carries the reason it is not dead code.
# Add an entry only for (a) framework overrides the runtime calls, or (b) API
# documented in docs/5-LIBRARY-API.md — never to silence a real dead symbol.
VULTURE_ALLOWLIST: dict[str, str] = {
    # BaseHTTPRequestHandler / socketserver callbacks (invoked by the stdlib).
    "viewer/serve.py|send_head": "socketserver hook called by the stdlib handler",
    "viewer/serve.py|log_message": "socketserver hook called by the stdlib handler",
    "viewer/serve.py|daemon_threads": "socketserver class attribute read by ThreadingMixIn",
    # Public library API (README + docs/5-LIBRARY-API.md), not called by the pipeline.
    "dardcollect/audio.py|transcribe_file": "documented public API (docs/5-LIBRARY-API.md)",
    "dardcollect/audio.py|transcribe_segment": "documented public API (docs/5-LIBRARY-API.md)",
}


def _vulture_findings(repo_root: Path) -> dict[str, dict[str, int]]:
    """vulture unused symbols at >= VULTURE_CONFIDENCE, keyed by file|symbol."""
    targets = _vulture_targets(repo_root)
    if not targets:
        return {}
    cmd = [
        sys.executable,
        "-m",
        "vulture",
        *targets,
        "--min-confidence",
        str(VULTURE_CONFIDENCE),
    ]
    try:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    except OSError as exc:
        raise QualityToolError(f"cannot run vulture: {exc}") from exc
    findings: dict[str, dict[str, int]] = {}
    for line in proc.stdout.splitlines():
        match = _VULTURE_LINE.match(line.strip())
        if match is None:
            continue
        rel = _relpath(match.group("file"), repo_root)
        key = f"{rel}|{match.group('name')}"
        if key in VULTURE_ALLOWLIST:
            continue
        findings.setdefault("deadcode", {})[key] = 1
    return findings


def collect(repo_root: Path) -> dict[str, dict[str, int]]:
    """Current violations per metric. Raises QualityToolError if a tool is missing."""
    findings: dict[str, dict[str, int]] = {}
    for source in (_ruff_findings, _function_lengths, _vulture_findings):
        for metric, bucket in source(repo_root).items():
            findings.setdefault(metric, {}).update(bucket)
    return findings


def _compare_metric(
    metric: str,
    base: dict[str, int],
    cur: dict[str, int],
    errors: list[str],
    notes: list[str],
) -> list[str]:
    """Compare one metric's current vs pinned values; returns covered rule ids."""
    covered: list[str] = []
    for key, value in sorted(cur.items()):
        if key not in base:
            errors.append(
                f"NEW {metric} violation: {key} (value {value}); fix it, or add a rule "
                f"to EXCEPTION_RULES (scripts/quality_gates.py) AND pin it in "
                f"{BASELINE_NAME} — a tolerated violation must be a codified exception"
            )
        elif value > base[key]:
            errors.append(
                f"WORSENED {metric}: {key} {base[key]} -> {value}; fix it, or raise the "
                f"pinned value in {BASELINE_NAME} under its existing codified rule"
            )
    for key, value in sorted(base.items()):
        if key not in cur:
            notes.append(f"{metric}: {key} resolved (was {value}) — remove from baseline")
            continue
        if cur[key] < value:
            notes.append(f"{metric}: {key} improved {value} -> {cur[key]} — lower the baseline")
        rule = _rule_for(metric, key)
        if rule is None:
            errors.append(
                f"UNCODIFIED exception: {metric} {key} is pinned in {BASELINE_NAME} but "
                f"no rule in EXCEPTION_RULES justifies it; fix the violation, or add a "
                f"rule stating a real reason"
            )
        else:
            covered.append(rule)
    return covered


def compare(
    current: dict[str, dict[str, int]],
    baseline: dict[str, dict[str, int]],
) -> tuple[list[str], list[str]]:
    """Ratchet comparison -> (errors for new/worse, notes for resolved/improved).

    Also enforces the no-anonymous-debt rule: every *baseline* entry (a tolerated
    violation) must match a codified rule in ``EXCEPTION_RULES``. An entry no rule
    covers is an error — fix the violation, or add a rule that states a real
    reason. A rule that matches nothing is a note (remove the stale rule).
    """
    errors: list[str] = []
    notes: list[str] = []
    covered: set[str] = set()
    for metric in sorted(set(current) | set(baseline)):
        covered.update(
            _compare_metric(
                metric,
                baseline.get(metric, {}),
                current.get(metric, {}),
                errors,
                notes,
            )
        )
    for rule_id in EXCEPTION_RULES:
        if rule_id not in covered:
            notes.append(f"rule '{rule_id}' matches no pinned violation — remove the stale rule")
    return errors, notes


def load_baseline(repo_root: Path) -> dict[str, dict[str, int]]:
    path = repo_root / BASELINE_NAME
    if not path.exists():
        raise QualityToolError(
            f"missing baseline {BASELINE_NAME}; create it with "
            f"`python scripts/quality_gates.py --write-baseline`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        current = collect(repo_root)
        if "--write-baseline" in args:
            (repo_root / BASELINE_NAME).write_text(
                json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"wrote {BASELINE_NAME}")
            return 0
        errors, notes = compare(current, load_baseline(repo_root))
    except QualityToolError as exc:
        print(f"[quality_gates] ERROR: {exc}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"[quality_gates] note: {note}")
    for error in errors:
        print(f"[quality_gates] {error}", file=sys.stderr)
    if errors:
        print(f"[quality_gates] FAILED: {len(errors)} new/worsened violation(s).", file=sys.stderr)
        return 1
    print("[quality_gates] OK: no new or worsened violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
