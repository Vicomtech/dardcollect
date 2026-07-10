#!/usr/bin/env python3
"""Golden snapshot harness for DARDcollect provenance outputs.

Capture/compare the traceability CSVs + JSON sidecars produced under ``DARD/``
against a normalized baseline manifest. UUIDs v4, ISO timestamps, and path
separators are normalized before hashing.

``compare`` is **non-determinism-tolerant**: GPU inference (TensorRT/CUDA) is
non-deterministic across runs, so inference-derived fields (keypoints, scores,
bboxes, transcription text, OCR) and even detection/crop counts drift run-to-
run. The harness therefore treats hash diffs / added-or-removed sidecars as
**drift** (informational), and only **hard-fails** on real regressions: a
baseline CSV is missing (a stage didn't produce it), sidecar volume out of
bounds (collapse or >4x swing), a sidecar's ``parent_*.uuid`` doesn't resolve
(broken provenance), or (with ``--validate``) a sidecar fails its JSON Schema.
Use ``--strict`` to fail on any drift (byte-exact mode for a deterministic
surface).

The manifest stores one SHA-256 per file (not the file contents), so it is
small. The raw ``DARD/`` outputs stay gitignored.

Usage::

    python scripts/golden_snapshot.py capture snapshots/golden_manifest.json
    python scripts/golden_snapshot.py compare snapshots/golden_manifest.json --validate
    python scripts/golden_snapshot.py compare snapshots/golden_manifest.json --validate --strict

``capture`` overwrites the manifest at the destination path. This script
imports only the stdlib + (optionally) ``dardcollect.fair`` for schema
validation — no GPU, no models. Keep it that way so the gate runs anywhere.
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


def _rel_key(p: Path, dard_root: Path) -> str:
    """Manifest key: repo-relative path with POSIX separators so a baseline
    captured on Windows (backslash) compares against a run on Linux/macOS
    (forward slash). Without this, every file shows as added/missing across OSes."""
    return str(p.relative_to(dard_root)).replace("\\", "/")


def _build(dard_root: Path) -> dict:
    surface = discover(dard_root)
    csv_map = {_rel_key(p, dard_root): _digest(p) for p in surface.csvs}
    sidecar_map = {_rel_key(p, dard_root): _digest(p) for p in surface.sidecars}
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


# ── provenance + volume (non-determinism-tolerant signals) ──────────────────

# CSV columns that hold artifact UUIDs (own + parent refs) — used to build the
# set of UUIDs that provenance parent-links must resolve against.
_UUID_CSV_COLS = ("uuid", "download_uuid", "parent_uuid", "detection_uuid")
# Sidecar keys whose .uuid is a parent reference that must resolve.
_PARENT_KEYS = ("parent_clip", "parent_crop", "parent_audio")


def _collect_uuids(surface: Surface) -> set[str]:
    uuids: set[str] = set()
    for p in surface.csvs:
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for col in _UUID_CSV_COLS:
                    val = row.get(col)
                    if val:
                        uuids.add(val)
    for p in surface.sidecars:
        try:
            with p.open(encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("uuid"):
            uuids.add(d["uuid"])
    return uuids


def _provenance_check(dard_root: Path, surface: Surface) -> list[str]:
    """Return messages for sidecars whose parent_*.uuid does not resolve to any
    artifact UUID in the output tree. Structural + non-determinism-tolerant."""
    uuids = _collect_uuids(surface)
    broken: list[str] = []
    for p in surface.sidecars:
        try:
            with p.open(encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for pk in _PARENT_KEYS:
            parent = d.get(pk)
            if isinstance(parent, dict) and parent.get("uuid"):
                if parent["uuid"] not in uuids:
                    broken.append(
                        f"  broken provenance: {p.relative_to(dard_root)} "
                        f"{pk}.uuid {parent['uuid']} not found"
                    )
    return broken


def _volume_check(
    baseline: dict, current: dict, bounds: tuple[float, float] = (0.25, 4.0)
) -> list[str]:
    """CSV count must match exactly (CSVs are structurally fixed per stage);
    sidecar count must stay within `bounds` of the baseline (detection/crop
    counts vary with GPU non-determinism, but a 4x swing or collapse to ~0 is a
    regression)."""
    issues: list[str] = []
    b_csv, c_csv = len(baseline["csv"]), len(current["csv"])
    if b_csv != c_csv:
        issues.append(
            f"  CSV count changed: baseline {b_csv} -> current {c_csv} (a stage gained/lost a CSV)"
        )
    b_sc = baseline["counts"]["sidecars"]
    c_sc = current["counts"]["sidecars"]
    if b_sc > 0:
        ratio = c_sc / b_sc
        if ratio < bounds[0] or ratio > bounds[1]:
            issues.append(
                f"  sidecar volume out of bounds: baseline {b_sc} -> current {c_sc} "
                f"(ratio {ratio:.2f}; expected {bounds[0]}x-{bounds[1]}x)"
            )
    elif c_sc > 0:
        issues.append(f"  sidecar volume: baseline 0 -> current {c_sc}")
    return issues


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


def cmd_compare(
    dard_root: Path, manifest_path: Path, validate: bool = False, strict: bool = False
) -> int:
    """Compare current DARD/ against a baseline.

    GPU inference (TensorRT/CUDA) is non-deterministic across runs, so
    byte-identical comparison of inference-derived fields (keypoints, scores,
    bboxes, transcription text, OCR) is NOT expected — those show as **drift**
    (informational, not a failure unless ``--strict``).

    Hard failures (real regressions): a baseline CSV is missing (a stage didn't
    produce it), sidecar volume out of bounds (collapse or >4x swing), a
    sidecar's parent_*.uuid doesn't resolve (broken provenance), or (with
    ``--validate``) a sidecar fails its JSON Schema.
    """
    if not manifest_path.exists():
        print(f"[compare] baseline manifest not found: {manifest_path}", file=sys.stderr)
        print(
            f"  capture one first: python scripts/golden_snapshot.py capture {manifest_path}",
            file=sys.stderr,
        )
        return 2
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _build(dard_root)
    surface = discover(dard_root)

    b_csv, c_csv = baseline["csv"], current["csv"]
    b_sc, c_sc = baseline["sidecars"], current["sidecars"]

    # Drift = hash diffs + added/removed files. Informational unless --strict.
    drift: list[str] = []
    matches = 0
    for key in sorted(set(b_csv) | set(c_csv)):
        if key not in c_csv:
            drift.append(f"  CSV removed: {key}")
        elif key not in b_csv:
            drift.append(f"  CSV added:   {key}")
        elif b_csv[key] != c_csv[key]:
            drift.append(f"  CSV drift:   {key}")
        else:
            matches += 1
    for key in sorted(set(b_sc) | set(c_sc)):
        if key not in c_sc:
            drift.append(f"  sidecar removed: {key}")
        elif key not in b_sc:
            drift.append(f"  sidecar added:   {key}")
        elif b_sc[key] != c_sc[key]:
            drift.append(f"  sidecar drift:   {key}")
        else:
            matches += 1

    # Hard failures: missing CSV, volume out of bounds, broken provenance, schema.
    hard: list[str] = []
    for key in baseline["csv"]:
        if key not in current["csv"]:
            hard.append(f"  MISSING CSV (stage regression): {key}")
    hard += _volume_check(baseline, current)
    hard += _provenance_check(dard_root, surface)
    schema_errors: dict[str, str] = {}
    if validate:
        schema_errors = _validate_sidecars(dard_root, surface.sidecars)
        for rel, err in schema_errors.items():
            hard.append(f"  sidecar schema-invalid: {rel}  ({err})")

    if strict:
        hard += drift

    print(
        f"[compare] {matches} match; {len(drift)} drift "
        f"(GPU non-determinism, informational{' — FAILED per --strict' if strict else ''}); "
        f"{len(hard)} hard-fail ({len(schema_errors)} schema-invalid)."
    )
    for line in hard[:50]:
        print(line)
    if len(hard) > 50:
        print(f"  ... and {len(hard) - 50} more hard-fail.")
    if drift and not strict:
        print("[compare] drift (informational; use --strict to fail on it):")
        for line in drift[:20]:
            print(line)
        if len(drift) > 20:
            print(f"  ... and {len(drift) - 20} more drift.")

    return 0 if not hard else 1


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
    cmp.add_argument(
        "--strict",
        action="store_true",
        help=(
            "fail on any drift (hash diff / added / removed) — byte-exact mode "
            "for deterministic surfaces"
        ),
    )
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
    return cmd_compare(dard_root, args.manifest, validate=args.validate, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
