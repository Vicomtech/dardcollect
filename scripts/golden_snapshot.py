#!/usr/bin/env python3
"""Golden snapshot harness for DARDcollect provenance outputs.

Capture/compare the traceability CSVs + JSON sidecars produced under ``DARD/``
against a normalized baseline manifest under ``snapshots/``. Because every
artifact carries a random UUID v4 and an ISO timestamp, byte-identical
comparison is impossible across re-runs; this harness normalizes volatile
fields (UUIDs, timestamps, path separators) before hashing, so a
behavior-preserving re-run produces a matching manifest.

The manifest stores one SHA-256 per file (not the file contents), so it is
small and committable. The raw ``DARD/`` outputs stay gitignored.

Usage::

    python scripts/golden_snapshot.py capture snapshots/golden_manifest.json
    python scripts/golden_snapshot.py compare snapshots/golden_manifest.json
    python scripts/golden_snapshot.py compare snapshots/golden_manifest.json --validate

``compare`` exits non-zero if any file is added/removed/changed (or, with
``--validate``, if a sidecar fails its JSON Schema). ``capture`` overwrites the
manifest at the destination path.

This script imports only the stdlib + (optionally) ``dardcollect.fair`` for
schema validation — no GPU, no models. Keep it that way so the golden gate
runs in any environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Volatility ──────────────────────────────────────────────────────────────
# UUID v4 columns/keys (random per run) and ISO timestamp columns/keys —
# their values are replaced with a placeholder before hashing.
VOLATILE_COLS = {
    "uuid",
    "download_uuid",
    "parent_uuid",
    "detection_uuid",
    "timestamp",
    "downloaded_at",
    "download_stage_timestamp",
    "extracted_at",
    "annotated_at",
}
# Columns whose values are filesystem paths — normalise ``\`` → ``/`` so a
# baseline captured on Windows compares against a run on Linux/macOS.
PATH_COLS = {
    "output_path",
    "source_image_path",
    "source_path",
    "source_video",
    "filename_downloaded",
    "source_image",
}
VOLATILE_KEYS = VOLATILE_COLS  # same set, applied recursively to JSON
PLACEHOLDER = "<v>"

# Files that are resume/progress state, not pipeline outputs — excluded.
PROGRESS_GLOB = "*_progress.json"

MANIFEST_VERSION = "1.0"


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── CSV normalisation ───────────────────────────────────────────────────────


def _canonical_csv(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    norm_rows: list[str] = []
    for row in rows:
        out: dict[str, str] = {}
        for key in fieldnames:
            val = row.get(key, "")
            if key in VOLATILE_COLS:
                val = PLACEHOLDER
            elif key in PATH_COLS:
                val = _normalise_path(val)
            out[key] = val
        # Stable per-row key: the whole normalised row, so sorting is
        # independent of write order.
        norm_rows.append(json.dumps(out, sort_keys=True, ensure_ascii=False))
    norm_rows.sort()
    header = ",".join(fieldnames)
    return header + "\n" + "\n".join(norm_rows)


# ── JSON normalisation ─────────────────────────────────────────────────────


def _normalise_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in VOLATILE_KEYS:
                out[key] = PLACEHOLDER
            else:
                out[key] = _normalise_json(value)
        return out
    if isinstance(obj, list):
        return [_normalise_json(v) for v in obj]
    if isinstance(obj, str):
        return _normalise_path(obj)
    return obj


def _canonical_json(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    norm = _normalise_json(data)
    return json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ── surface discovery ───────────────────────────────────────────────────────


@dataclass
class Surface:
    csvs: list[Path]
    sidecars: list[Path]


def discover(dard_root: Path) -> Surface:
    csvs = sorted(p for p in dard_root.rglob("*.csv") if p.is_file())
    sidecars = sorted(
        p for p in dard_root.rglob("*.json") if p.is_file() and not p.match(PROGRESS_GLOB)
    )
    return Surface(csvs=csvs, sidecars=sidecars)


def _digest(path: Path) -> str:
    text = _canonical_json(path) if path.suffix == ".json" else _canonical_csv(path)
    return _hash(text)


def _build(dard_root: Path) -> dict:
    surface = discover(dard_root)
    csv_map = {str(p.relative_to(dard_root)): _digest(p) for p in surface.csvs}
    sidecar_map = {str(p.relative_to(dard_root)): _digest(p) for p in surface.sidecars}
    return {
        "manifest_version": MANIFEST_VERSION,
        "dard_root": str(dard_root),
        "csv": csv_map,
        "sidecars": sidecar_map,
        "counts": {"csv": len(csv_map), "sidecars": len(sidecar_map)},
    }


# ── optional schema validation ──────────────────────────────────────────────

# Path heuristics → schema_type (loaded from schemas/ via dardcollect.fair).
_SCHEMA_BY_SUFFIX = {
    ".transcription.json": "transcription",
    ".quality.json": "quality_annotation",
}
# person_clip and face_crop sidecars have no distinguishing suffix; map by
# directory instead.
_SCHEMA_BY_DIR = {
    "extracted_person_clips": "person_clip",
    "extracted_image_detections": "image_detection",
    "video_face_crops": "face_crop",
    "image_face_crops": "face_crop",
}


def _schema_for(rel: str) -> str | None:
    # MagFace unified-score sidecars are not the OFIQ quality_annotation nor a
    # face_crop sidecar — no ratified schema for them; skip.
    if rel.endswith(".magface.json"):
        return None
    for suffix, st in _SCHEMA_BY_SUFFIX.items():
        if rel.endswith(suffix):
            return st
    parts = rel.replace("\\", "/").split("/")
    if parts and parts[0] in _SCHEMA_BY_DIR:
        return _SCHEMA_BY_DIR[parts[0]]
    return None


def _validate_sidecars(dard_root: Path, sidecars: list[Path]) -> dict[str, str]:
    """Return {relpath: error_message} for sidecars that fail their schema.

    Imports dardcollect.fair lazily so the harness stays GPU-free without it.
    """
    try:
        from dardcollect.fair import load_schema, validate_against_schema
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"[validate] could not import dardcollect.fair: {exc}", file=sys.stderr)
        return {}
    errors: dict[str, str] = {}
    for p in sidecars:
        rel = str(p.relative_to(dard_root))
        st = _schema_for(rel)
        if st is None:
            continue
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            load_schema(st)  # ensure schema exists
            validate_against_schema(data, st)
        except Exception as exc:
            # jsonschema.ValidationError.__str__ includes the full validated
            # instance (huge); keep only the first diagnostic line.
            first = (str(exc).splitlines() or [""])[0][:140]
            errors[rel] = f"{type(exc).__name__}: {first}"
    return errors


# ── commands ────────────────────────────────────────────────────────────────


def cmd_capture(dard_root: Path, manifest_path: Path) -> int:
    manifest = _build(dard_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[capture] {manifest['counts']['csv']} CSVs, "
        f"{manifest['counts']['sidecars']} sidecars -> {manifest_path}"
    )
    return 0


def cmd_compare(dard_root: Path, manifest_path: Path, validate: bool = False) -> int:
    if not manifest_path.exists():
        print(f"[compare] baseline manifest not found: {manifest_path}", file=sys.stderr)
        print(
            f"  capture one first: python scripts/golden_snapshot.py capture {manifest_path}",
            file=sys.stderr,
        )
        return 2
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _build(dard_root)

    b_csv, c_csv = baseline["csv"], current["csv"]
    b_sc, c_sc = baseline["sidecars"], current["sidecars"]

    diffs: list[str] = []
    matches = 0
    for key in sorted(set(b_csv) | set(c_csv)):
        if key not in c_csv:
            diffs.append(f"  CSV removed:   {key}")
        elif key not in b_csv:
            diffs.append(f"  CSV added:     {key}")
        elif b_csv[key] != c_csv[key]:
            diffs.append(f"  CSV changed:   {key}")
        else:
            matches += 1
    for key in sorted(set(b_sc) | set(c_sc)):
        if key not in c_sc:
            diffs.append(f"  sidecar removed: {key}")
        elif key not in b_sc:
            diffs.append(f"  sidecar added:   {key}")
        elif b_sc[key] != c_sc[key]:
            diffs.append(f"  sidecar changed: {key}")
        else:
            matches += 1

    schema_errors: dict[str, str] = {}
    if validate:
        surface = discover(dard_root)
        schema_errors = _validate_sidecars(dard_root, surface.sidecars)
        for rel, err in schema_errors.items():
            diffs.append(f"  sidecar schema-invalid: {rel}  ({err})")

    print(
        f"[compare] {matches} files match baseline; "
        f"{len(diffs)} differences ({len(schema_errors)} schema-invalid)."
    )
    for line in diffs[:50]:
        print(line)
    if len(diffs) > 50:
        print(f"  ... and {len(diffs) - 50} more.")

    return 0 if not diffs else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "DARDcollect golden snapshot harness"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="capture current DARD/ as a baseline manifest")
    cap.add_argument("manifest", type=Path)
    cmp = sub.add_parser("compare", help="compare current DARD/ against a baseline")
    cmp.add_argument("manifest", type=Path)
    cmp.add_argument("--validate", action="store_true", help="also JSON-Schema-validate sidecars")
    parser.add_argument(
        "--dard-root",
        type=Path,
        default=Path("DARD"),
        help="DARD output root (default: ./DARD)",
    )
    args = parser.parse_args(argv)

    dard_root: Path = args.dard_root
    if not dard_root.exists():
        print(f"error: DARD root not found: {dard_root}", file=sys.stderr)
        return 2

    if args.cmd == "capture":
        return cmd_capture(dard_root, args.manifest)
    return cmd_compare(dard_root, args.manifest, validate=args.validate)


if __name__ == "__main__":
    raise SystemExit(main())
