#!/usr/bin/env python3
"""Generate ``configs/config.test.yaml`` from ``configs/config.archive_all.yaml``.

Produces the fast fixture-gate config.

The test config is the production config with input/output paths redirected to
the committed fixture media (``tests/fixtures/media/``) and a throwaway output
tree (``DARD_test/``). Both path conventions are handled: literal ``DARD/...``
strings and ``root:`` + ``{root}/...`` templating (the production config
convention, see SUBSTITUTIONS). Regenerate whenever
``configs/config.archive_all.yaml`` changes so the test config never goes
stale — do NOT hand-edit ``configs/config.test.yaml``.

Usage::

    python scripts/make_test_config.py            # writes configs/config.test.yaml

Idempotent: overwrites the output. Run once per machine (the output is
gitignored — it is a derived artifact, not source).

Fail-loud: if after substitution any line still contains a production path
(``C:/data``) or an unresolved ``{root}`` template, generation raises
:class:`TemplateMismatch` (exit non-zero) instead of writing a hollow gate
config. This is the fix requested by issue #7.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Order matters: substitute specific media-subdir paths before the generic
# base-output dir, and derived DARD/ outputs so they don't collide with the
# media substitutions.
#
# Two path conventions are supported (the production config moved from literal
# ``DARD/...`` strings to ``root:`` + ``{root}/...`` templating):
#   - literal ``DARD/<name>`` strings are replaced as-is;
#   - templated ``{root}/<name>`` strings are rewritten to literal
#     ``tests/fixtures/media/<name>`` / ``DARD_test/<name>`` paths (the test
#     config carries its own explicit paths; ``root`` itself is repointed
#     below so any remaining ``{root}`` references stay coherent).
SUBSTITUTIONS: list[tuple[str, str]] = [
    # Templated derived paths first ({root}/...), so the literal DARD/ rules
    # below cannot pre-empt them. {root} itself is left untouched here and
    # redirected by the (root) substitution at the end.
    ("{root}/archive_org_public_domain/videos", "tests/fixtures/media/videos"),
    ("{root}/archive_org_public_domain/images", "tests/fixtures/media/images"),
    ("{root}/archive_org_public_domain/audio", "tests/fixtures/media/audio"),
    ("{root}/archive_org_public_domain/texts", "tests/fixtures/media/texts"),
    ("{root}/archive_org_public_domain", "tests/fixtures/media"),
    ("{root}/extracted_person_clips", "DARD_test/extracted_person_clips"),
    ("{root}/extracted_image_detections", "DARD_test/extracted_image_detections"),
    ("{root}/video_face_crops", "DARD_test/video_face_crops"),
    ("{root}/image_face_crops", "DARD_test/image_face_crops"),
    ("{root}/filtered_video_face_crops", "DARD_test/filtered_video_face_crops"),
    ("{root}/filtered_image_face_crops", "DARD_test/filtered_image_face_crops"),
    ("{root}/audio_transcriptions", "DARD_test/audio_transcriptions"),
    ("{root}/preprocessed_documents", "DARD_test/preprocessed_documents"),
    ("{root}/extracted_frames", "DARD_test/extracted_frames"),
    # Literal DARD/ paths (legacy base_output_dir convention).
    # The absolute production base_output_dir (C:/data/DARD/...) must be
    # redirected FIRST: the generic DARD/... literal below would otherwise
    # match inside it (C:/data/DARD/archive_org_public_domain contains the
    # literal substring) and leave a C:/data/tests/... prefix behind.
    (
        'base_output_dir: "C:/data/DARD/archive_org_public_domain"',
        'base_output_dir: "tests/fixtures/media"',
    ),
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
    # Repoint {root} last: any remaining templated path now resolves under
    # DARD_test/ instead of the production dataset root.
    ("{root}", "DARD_test"),
    ('root: "C:/data/DARD"', 'root: "DARD"'),
]


class TemplateMismatch(RuntimeError):
    """Raised when substitution left unresolved production/templated paths.

    A generated test config must never silently carry a ``C:/data/...`` path or
    an unresolved ``{root}`` template: the fixture gate would silently read
    production data (or skip stages) instead of failing.
    """


def main(argv: list[str] | None = None) -> int:
    src_path = REPO_ROOT / "configs" / "config.archive_all.yaml"
    out_path = REPO_ROOT / "configs" / "config.test.yaml"
    if not src_path.exists():
        print(f"error: {src_path} not found", file=sys.stderr)
        return 2
    src = src_path.read_text(encoding="utf-8")
    for old, new in SUBSTITUTIONS:
        src = src.replace(old, new)
    leftover = [line for line in src.splitlines() if _unresolved(line)]
    if leftover:
        raise TemplateMismatch(
            "generated test config still contains production/templated paths "
            f"({len(leftover)} line(s)) — extend the SUBSTITUTIONS list in "
            "scripts/make_test_config.py to cover them:\n  " + "\n  ".join(leftover[:10])
        )
    out_path.write_text(src, encoding="utf-8")
    print(f"[make_test_config] wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


def _unresolved(line: str) -> bool:
    """A line is unresolved if it still references production or templated paths.

    Checked across the whole line (comments included): a stale ``C:/data/...``
    path in a comment would still document a wrong path in the generated
    config. The ``make_test_config`` exclusion lets the generator's own name
    appear in comments without tripping the check.
    """
    return "C:/data" in line or "{root}" in line


if __name__ == "__main__":
    raise SystemExit(main())
