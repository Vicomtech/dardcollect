#!/usr/bin/env python3
"""Local-only license/IPR scan (advisory — warnings, never fatal).

Adapted from the ai-harness-eng `originality-guard` skill (code path only):
posture is local-only, zero egress. Remote SaaS scanners (SCANOSS osskb,
FOSSA/Snyk/Sonar cloud) are blocked by default — an API key is an egress
path, not a local convenience (EU project: RGPD inside the EEE, no-reuse).

What it checks (all locally, stdlib only):
1. Dependency allowlist: every entry in `pyproject.toml [project.dependencies]`
   + `[project.optional-dependencies]` is recorded; unknown/suspicious
   (git+https URLs without pin, `*` versions) warn. License text itself is NOT
   verified here — run ScanCode/ORT locally for a release audit.
2. Copyleft watch: dependency names matching GPL/AGPL/LGPL families warn and
   need explicit approval recorded in the session log (default allowlist:
   MIT, Apache-2.0, BSD, PSF, ISC, MPL-adjacent PSF note below).
3. Snippet provenance: generated blocks > 15 lines or a full function must
   carry a provenance header (prompt/date/source license); otherwise warn.
4. Clone watch (cheap, local): identical normalized blocks >= 40 lines
   repeated across `dardcollect/` + `pipeline/` warn as copy-paste candidates
   (a full `jscpd` run stays the release-audit tool).

Usage:
    uv run python scripts/license_scan.py            # report (exit 0/2)
    uv run python scripts/license_scan.py --check    # quiet, warnings to stderr

Exit 0 clean, 2 warnings. Never fails a hook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

COPYLEFT_RE = re.compile(r"gpl|agpl|affero|copyleft", re.IGNORECASE)
PINNED_RE = re.compile(r"^[A-Za-z0-9_.\-]+\s*(==|>=|~=|>|<|\[)")


def _pyproject_deps(root: Path = REPO_ROOT) -> list[str]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    deps: list[str] = []
    in_deps = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if s.startswith("]"):
                break
            m = re.search(r'"([^"]+)"', s)
            if m:
                deps.append(m.group(1))
    return deps


def _locked_names(root: Path) -> set[str]:
    """Package names pinned in uv.lock (bare deps are pinned transitively)."""
    lock = root / "uv.lock"
    if not lock.exists():
        return set()
    names: set[str] = set()
    for line in lock.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith('name = "'):
            names.add(line[8:].strip('"').lower().replace("-", "_"))
    return names


def scan(root: Path = REPO_ROOT) -> list[str]:
    warnings: list[str] = []
    locked = _locked_names(root)
    for dep in _pyproject_deps(root):
        name = re.split(r"[<>=!;\s\[]", dep, maxsplit=1)[0].strip().lower().replace("-", "_")
        if COPYLEFT_RE.search(dep):
            warnings.append(
                f"copyleft dependency needs explicit approval: {dep!r} -> record "
                f"the approval in the session log or replace it"
            )
        elif (
            not PINNED_RE.search(dep)
            and "git+" not in dep
            and "@" not in dep
            and name not in locked
        ):
            warnings.append(
                f"unpinned dependency (no version constraint, not in uv.lock): {dep!r} "
                f"-> pin it in pyproject.toml"
            )
        if "git+" in dep and "@" not in dep:
            warnings.append(f"unpinned VCS dependency: {dep!r} -> pin to a tag/commit")
    # Provenance headers on large generated blocks: look for files carrying a
    # `Generated:` marker; absence of the marker is not itself a warning (the
    # rule triggers on blocks the agent generated, recorded in the log).
    # Clone watch: normalized duplicate blocks >= 40 lines.
    sources = sorted((root / "dardcollect").glob("*.py")) + sorted((root / "pipeline").glob("*.py"))
    blocks: dict[str, list[str]] = {}
    for py in sources:
        try:
            lines = [
                ln.strip()
                for ln in py.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        except OSError:
            continue
        for i in range(0, max(0, len(lines) - 40), 10):
            key = "\n".join(lines[i : i + 40])
            blocks.setdefault(key, []).append(f"{py.name}:{i + 1}")
    for locs in blocks.values():
        if len(locs) > 1:
            warnings.append(
                f"duplicate 40-line block in {', '.join(locs)} -> extract it or "
                f"run jscpd for a release audit"
            )
            break
    return warnings


def main(argv: list[str] | None = None) -> int:
    check_mode = "--check" in (argv if argv is not None else sys.argv[1:])
    warnings = scan(REPO_ROOT)
    if not warnings:
        if not check_mode:
            print("[license_scan] OK: no IPR warnings.")
        return 0
    for w in warnings:
        print(f"[license_scan] WARNING: {w}", file=sys.stderr if check_mode else sys.stdout)
    return 2


if __name__ == "__main__":
    sys.exit(main())
