#!/usr/bin/env python3
"""pre-commit wrapper for validate_harness.py --check.

Exit-code contract (scripts/validate_harness.py): 0 = clean, 2 = warnings-only
(advisory checks), 1 = hard failures. pre-commit fails a hook on ANY nonzero
exit, so this wrapper accepts rc==2 (warnings print in the report) and
forwards everything else unchanged. CI runs the same wrapper so a
warnings-only run does not fail the workflow either.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_harness.py", "--check"],
        cwd=str(REPO_ROOT),
    )
    if proc.returncode == 2:
        return 0
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
