#!/usr/bin/env python3
"""pre-commit / CI wrapper around `validate_harness.py --check`.

Exit-code contract (scripts/validate_harness.py): 0 = clean, 2 = warnings-only
(advisory checks), 1 = hard failures. pre-commit fails a hook on ANY nonzero
exit, so this wrapper accepts rc==2 (warnings print in the report) and forwards
everything else unchanged.

With `--sync-mirror` (the pre-commit path) it first refreshes the derived skill
mirror: `scripts/skill_mounts.py --sync` regenerates `.claude/skills/` from the
canonical `.agents/skills/` tree and this wrapper stages it, so the canonical
copy always wins and a stale mirror cannot be committed. CI runs the wrapper
WITHOUT the flag, so a mirror that a `--no-verify` commit let through is still
reported by the `skill mounts` gate instead of being silently repaired.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIRROR_DIR = ".claude/skills"


def _refresh_skill_mirror() -> int:
    """Regenerate every declared skill mount from the canonical tree, then stage it."""
    sync = subprocess.run(
        [sys.executable, "scripts/skill_mounts.py", "--sync"],
        cwd=str(REPO_ROOT),
    )
    if sync.returncode != 0:
        print(
            "hook_validate_harness: skill-mount refresh FAILED - blocking commit.",
            file=sys.stderr,
        )
        return 1
    # Stage the refreshed mirror so the commit carries the canonical content.
    # Best effort: the git index is irrelevant to CI's plain validation path.
    subprocess.run(
        ["git", "add", "-A", "--", MIRROR_DIR],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-mirror",
        action="store_true",
        help="refresh and stage the derived skill mounts before validating "
        "(pre-commit path; CI leaves it off so drift is reported, not repaired)",
    )
    args = parser.parse_args(argv)

    if args.sync_mirror:
        rc = _refresh_skill_mirror()
        if rc != 0:
            return rc

    proc = subprocess.run(
        [sys.executable, "scripts/validate_harness.py", "--check"],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode == 2:
        return 0
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
