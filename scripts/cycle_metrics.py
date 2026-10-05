#!/usr/bin/env python3
"""Cycle metrics logging — one JSON entry per work cycle (adapted from the
ai-harness-eng harness; minimalist port).

The script deliberately accepts no model argument: model and provider
identifiers must stay out of the metrics log — the guarantee is structural,
not procedural.

Usage:
    uv run python scripts/cycle_metrics.py log --phases 2,3,4,5 --files a.py,b.py --status ok
    uv run python scripts/cycle_metrics.py summary --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE / ".kilo" / "_metrics"
METRICS_FILE = METRICS_DIR / "cycle_log.json"


def load_log() -> list[dict]:
    if not METRICS_FILE.exists():
        return []
    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_log(log: list[dict]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def log_event(args: argparse.Namespace) -> None:
    log = load_log()
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phases": [p.strip() for p in args.phases.split(",") if p.strip()],
        "files_processed": [f.strip() for f in args.files.split(",") if f.strip()],
        "status": args.status or "unknown",
        "notes": args.notes or "",
    }
    if args.minutes is not None:
        entry["minutes"] = args.minutes
    log.append(entry)
    save_log(log)
    print(f"OK: cycle logged in {METRICS_FILE}")


def show_summary(args: argparse.Namespace) -> None:
    log = load_log()
    if not log:
        print("No records.")
        return
    for entry in log[-args.limit :]:
        ts = entry.get("timestamp", "?")
        phases = ",".join(entry.get("phases", []))
        n_files = len(entry.get("files_processed", []))
        status = entry.get("status", "?")
        minutes = entry.get("minutes")
        mins = f" {minutes:.1f} min" if minutes is not None else ""
        print(f"{ts}  phases={phases} files={n_files} status={status}{mins}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cycle metrics logging")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_log = sub.add_parser("log", help="Log one work cycle")
    p_log.add_argument("--phases", default="", help="Phases traversed (e.g.: 2,3,4,5)")
    p_log.add_argument("--files", default="", help="Files processed (comma-separated)")
    p_log.add_argument("--status", default="ok", help="Cycle status (ok/error)")
    p_log.add_argument("--minutes", type=float, default=None, help="Elapsed minutes (optional)")
    p_log.add_argument("--notes", default="", help="Additional notes")
    p_sum = sub.add_parser("summary", help="Show recent cycles")
    p_sum.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "log":
        log_event(args)
    else:
        show_summary(args)
    sys.exit(0)
