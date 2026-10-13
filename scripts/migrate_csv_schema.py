"""One-shot migration for existing traceability CSVs.

The traceability CSVs became the lean *event log* (identity + links +
input/output paths + run-level values); measures the sidecar already stores are
no longer duplicated. Two rounds changed headers:

- clips: ``archive_org_identifier`` -> ``download_uuid`` (the upstream manifest
  row's uuid), and the derived measures (fps, frame span, seconds, duration,
  persons-per-frame) were dropped;
- crops / image detections / frames / documents: derived measures dropped
  (see ``MIGRATIONS`` below).

A logger appends without rewriting the header, so a stale header would land
values in the wrong columns; ``_guard_header`` refuses such a file. This script
rewrites the header and drops the removed columns in place.

This migration is **metadata-only**: it renames/drops columns, it does not touch
the media, the sidecars, or the row identities. Rows keep their ``uuid``.

Usage
-----
    uv run python scripts/migrate_csv_schema.py --dard-root DARD            # dry run
    uv run python scripts/migrate_csv_schema.py --dard-root DARD --apply    # write

Idempotent: a file whose header is already current is left untouched.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_NAME = "downloads.csv"

# Per CSV: columns to rename (old -> new) and columns to drop. The output header
# is the existing header with renames applied, then the dropped columns removed.
MIGRATIONS: dict[str, dict] = {
    "clips_extraction.csv": {
        "rename": {"archive_org_identifier": "download_uuid"},
        "drop": {
            "fps",
            "start_frame",
            "end_frame",
            "start_seconds",
            "duration_seconds",
            "max_persons_per_frame",
        },
        "resolve_download_uuid": True,
    },
    "video_face_crops_extraction.csv": {"drop": {"crop_id", "face_bbox", "confidence"}},
    "image_face_crops_extraction.csv": {"drop": {"bbox_in_source", "bbox_confidence"}},
    "image_person_detection.csv": {"drop": {"source_image", "num_persons"}},
    "frames_extraction.csv": {"drop": {"timestamp_seconds"}},
    "document_text_extraction.csv": {"drop": {"text_length", "word_count"}},
}


def _load_manifest_lookup(dard_root: Path) -> dict[str, str]:
    """Map {basename(filename_downloaded): manifest-row uuid} from downloads.csv."""
    lookup: dict[str, str] = {}
    manifest = dard_root / MANIFEST_NAME
    if not manifest.exists():
        return lookup
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = Path(row.get("filename_downloaded", "")).name
            uid = row.get("uuid", "")
            if name and uid:
                lookup[name] = uid
    return lookup


def _target_header(header: list[str], spec: dict) -> list[str]:
    rename = spec.get("rename", {})
    drop = spec.get("drop", set())
    out: list[str] = []
    for col in header:
        new = rename.get(col, col)
        if new not in drop:
            out.append(new)
    return out


def _migrate_one(
    csv_path: Path, spec: dict, manifest: dict[str, str], apply: bool
) -> tuple[int, bool]:
    """Rewrite one CSV's header (renames + drops). Returns (row count, changed)."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    target = _target_header(header, spec)
    if header == target:
        return len(rows), False

    rename = spec.get("rename", {})
    drop = spec.get("drop", set())
    for row in rows:
        for old, new in rename.items():
            if old in row:
                row[new] = row[old]
        for col in list(row):
            if col in drop or (col in rename and col != rename[col]):
                row.pop(col, None)
        if spec.get("resolve_download_uuid"):
            name = Path(row.get("source_video", "")).name
            row["download_uuid"] = manifest.get(name, "")

    print(f"{'rewriting' if apply else 'would rewrite'} {csv_path} ({len(rows)} rows)")
    if apply:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=target, restval="")
            writer.writeheader()
            writer.writerows(rows)
    return len(rows), True


def main() -> int:  # pragma: no cover - exercised via _migrate_one in tests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dard-root", required=True, help="DARD output root to migrate")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    dard_root = Path(args.dard_root)
    if not dard_root.exists():
        raise SystemExit(f"--dard-root does not exist: {dard_root}")

    manifest = _load_manifest_lookup(dard_root)
    changed = 0
    seen = 0
    for name, spec in MIGRATIONS.items():
        for csv_path in sorted(dard_root.rglob(name)):
            seen += 1
            _, did_change = _migrate_one(csv_path, spec, manifest, args.apply)
            changed += int(did_change)

    if not seen:
        print(f"No known traceability CSVs found under {dard_root} — nothing to migrate.")
    elif not args.apply and changed:
        print("Dry run — re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
