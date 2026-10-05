"""Interrupted-move reconciliation for the quality-filter stage (extracted).

``filter_face_crops_by_quality.py`` grew past the god-file cap with the
reconcile logic; the recoverable-move helpers live here (pipeline → library
direction only). The move trio (crop → sidecar → magface → CSV row) is not
atomic: a crash between two moves leaves an incomplete set in output_dir that
the forward pass cannot see. Reconcile returns stranded media to input_dir so
the next pass re-assesses it as a whole.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from dardcollect.face_crop_discovery import find_face_crops

logger = logging.getLogger(__name__)


def reconcile_partial_moves(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Repair interrupted forward moves before the filter's forward pass.

    Walks *output_dir* and, for every crop whose sidecar/magface did NOT land,
    moves the stranded media back to *input_dir* so the next forward pass
    re-assesses it as a whole. Crops whose full set IS present are left alone
    (they are complete, already moved).

    Returns ``(repaired, stranded)``.
    """
    output_crops = find_face_crops(output_dir)
    repaired = 0
    stranded = 0
    for dest_crop in output_crops:
        rel_parent = dest_crop.parent.relative_to(output_dir)
        src_dir = input_dir / rel_parent
        src_crop = src_dir / dest_crop.name
        dest_sidecar = dest_crop.with_suffix(".json")
        dest_magface = dest_crop.with_suffix(".magface.json")
        if dest_sidecar.exists() and dest_magface.exists():
            continue  # complete set — genuinely moved
        # Incomplete set: put the media back so the forward pass sees it again.
        try:
            if src_crop.exists():
                logger.warning(
                    "Reconcile: %s exists in BOTH input and output dirs — the "
                    "input copy is authoritative; removing the incomplete "
                    "output copy for a whole-set re-assessment",
                    dest_crop.name,
                )
                dest_crop.unlink()
            else:
                src_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest_crop), str(src_crop))
                repaired += 1
            logger.warning(
                "Reconcile: incomplete set for %s (sidecar=%s magface=%s) — media returned "
                "to input_dir for a whole-set re-assessment",
                dest_crop.name,
                dest_sidecar.exists(),
                dest_magface.exists(),
            )
        except Exception as exc:
            logger.error("Reconcile: failed to repair %s: %s", dest_crop.name, exc)
            stranded += 1
    return repaired, stranded
