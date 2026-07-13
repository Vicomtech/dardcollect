#!/usr/bin/env python3
"""Fresh fixture objective gate — one command, exits 0 only if both stages pass.

    uv run python scripts/objective_gate.py            # fresh: wipe + pipeline + golden
    uv run python scripts/objective_gate.py --no-wipe  # resumable re-run (NOT a regression check)

Why FRESH: pipeline stages are resumable (``.done`` sentinels skip completed
work). A re-run on an existing ``DARD_test`` skips completed stages and the
golden compare can pass against STALE outputs from a prior session — masking
regressions that only surface when stages actually execute. (This session hit
exactly that: the gate looked green against pre-existing outputs while a fresh
run crashed on a ``flush_segments`` bug + a recursive-scan bug that the skip
path hid.) Always verify a chunk with a fresh ``DARD_test``; use ``--no-wipe``
only for fast iteration, never as a regression sign-off.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DARD_TEST = REPO_ROOT / "DARD_test"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.test.yaml"
GOLDEN_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "golden_manifest.json"


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "Fresh fixture objective gate."
    )
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="re-run on existing DARD_test (resumable; NOT a valid regression check)",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    if not args.no_wipe:
        if DARD_TEST.exists():
            print(f"[objective_gate] wiping {DARD_TEST} (fresh run)", flush=True)
            shutil.rmtree(DARD_TEST)
    else:
        print(
            "[objective_gate] --no-wipe: re-running on existing DARD_test "
            "(NOT a valid regression check)",
            flush=True,
        )

    if not Path(args.config).exists():
        print(f"[objective_gate] config not found: {args.config}", file=sys.stderr)
        return 2

    py = sys.executable
    rc1 = _run([py, "scripts/run_pipeline.py", "--config", args.config])
    if rc1 != 0:
        print(f"[objective_gate] pipeline FAILED (rc={rc1})", file=sys.stderr)
        return 1
    rc2 = _run(
        [
            py,
            "scripts/golden_snapshot.py",
            "--dard-root",
            "DARD_test",
            "compare",
            str(GOLDEN_MANIFEST),
            "--validate",
        ]
    )
    if rc2 != 0:
        print(f"[objective_gate] golden compare FAILED (rc={rc2})", file=sys.stderr)
        return 1
    print("[objective_gate] PASS: fresh pipeline EXIT 0 + golden compare 0 hard-fail", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
