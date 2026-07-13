#!/usr/bin/env python3
"""Generate ``configs/config.test.yaml`` from ``configs/config.archive_all.yaml``.

Produces the fast fixture-gate config.

The test config is the production config with input/output paths redirected to
the committed fixture media (``tests/fixtures/media/``) and a throwaway output
tree (``DARD_test/``). Regenerate whenever ``configs/config.archive_all.yaml``
changes so the test config never goes stale — do NOT hand-edit
``configs/config.test.yaml``.

Usage::

    python scripts/make_test_config.py            # writes configs/config.test.yaml

Idempotent: overwrites the output. Run once per machine (the output is
gitignored — it is a derived artifact, not source).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Order matters: substitute specific media-subdir paths before the generic
# base-output dir, and derived DARD/ outputs so they don't collide with the
# media substitutions.
SUBSTITUTIONS: list[tuple[str, str]] = [
    ("DARD/archive_org_public_domain/videos", "tests/fixtures/media/videos"),
    ("DARD/archive_org_public_domain/images", "tests/fixtures/media/images"),
    ("DARD/archive_org_public_domain/audio", "tests/fixtures/media/audio"),
    ("DARD/archive_org_public_domain/texts", "tests/fixtures/media/texts"),
    ("DARD/archive_org_public_domain", "tests/fixtures/media"),
    ("DARD/extracted_person_clips", "DARD_test/extracted_person_clips"),
    ("DARD/extracted_image_detections", "DARD_test/extracted_image_detections"),
    ("DARD/video_face_crops", "DARD_test/video_face_crops"),
    ("DARD/image_face_crops", "DARD_test/image_face_crops"),
    ("DARD/filtered_video_face_crops", "DARD_test/filtered_video_face_crops"),
    ("DARD/filtered_image_face_crops", "DARD_test/filtered_image_face_crops"),
    ("DARD/audio_transcriptions", "DARD_test/audio_transcriptions"),
    ("DARD/preprocessed_documents", "DARD_test/preprocessed_documents"),
    ("DARD/extracted_frames", "DARD_test/extracted_frames"),
]


def main(argv: list[str] | None = None) -> int:
    src_path = REPO_ROOT / "configs" / "config.archive_all.yaml"
    out_path = REPO_ROOT / "configs" / "config.test.yaml"
    if not src_path.exists():
        print(f"error: {src_path} not found", file=sys.stderr)
        return 2
    src = src_path.read_text(encoding="utf-8")
    for old, new in SUBSTITUTIONS:
        src = src.replace(old, new)
    out_path.write_text(src, encoding="utf-8")
    leftover = [line for line in src.splitlines() if "DARD/" in line and "DARD_test" not in line]
    if leftover:
        print("warning: stray DARD/ lines in generated config:", file=sys.stderr)
        for line in leftover[:5]:
            print(f"  {line}", file=sys.stderr)
    print(f"[make_test_config] wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
