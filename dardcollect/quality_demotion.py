"""Demotion of already-filtered face crops (opt-in, issue #6).

When ``demote_on_raise: true``, a re-run of the quality-filter stage re-evaluates
crops already moved to ``output_dir`` against the CURRENT threshold using their
cached ``.magface.json`` and moves back to ``input_dir`` — crop + ``.json`` +
``.magface.json`` — those that no longer pass. That makes raising
``quality_threshold`` take effect in the already-filtered set too.

Extracted from ``pipeline/filter_face_crops_by_quality.py`` (2026-09-23) so the
stage script stays under the 600-line cap. The library never imports pipeline
stage scripts; this is the allowed direction (pipeline → library).

Name collisions on the reverse move are never overwritten: both copies are left
in place with a loud warning.
"""

import json
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from dardcollect.face_crop_discovery import find_face_crops

logger = logging.getLogger(__name__)


def demote_crop(dest_crop: Path, input_dir: Path, output_dir: Path) -> str:
    """Move one already-filtered crop (+ sidecars) back to input_dir.

    Returns a status string: ``"demoted"`` on success, ``"demote_collision"``
    when the source path already exists in input_dir (never overwritten — both
    copies are left in place and logged loudly), ``"demote_error"`` on a move
    failure.
    """
    crop = Path(dest_crop)
    try:
        rel_parent = crop.relative_to(output_dir).parent
    except ValueError:
        rel_parent = Path()
    src_sidecar = crop.with_suffix(".json")
    src_magface = crop.with_suffix(".magface.json")

    dest_dir = input_dir / rel_parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    targets = [(crop, dest_dir / crop.name)]
    if src_sidecar.exists():
        targets.append((src_sidecar, dest_dir / src_sidecar.name))
    if src_magface.exists():
        targets.append((src_magface, dest_dir / src_magface.name))

    try:
        for src, dest in targets:
            if dest.exists():
                logger.warning(
                    "Demote collision: %s already exists in input_dir — leaving both copies",
                    dest,
                )
                return "demote_collision"
            shutil.move(str(src), str(dest))
    except Exception as exc:
        logger.error("Failed to demote %s: %s", crop.name, exc)
        return "demote_error"

    logger.info(
        "DEMOTED %s (below current threshold) → %s",
        crop.name,
        dest_dir,
    )
    return "demoted"


def _cached_max_score(magface_path: Path) -> float | None:
    """Max score from a cached ``.magface.json``, or None if it cannot be read."""
    if not magface_path.exists():
        logger.warning(
            "Demote: %s has no .magface.json — cannot re-evaluate, crop left in place",
            magface_path.name,
        )
        return None
    try:
        with open(magface_path, encoding="utf-8") as f:
            magface_data = json.load(f)
    except Exception as exc:
        logger.warning(
            "Demote: cannot read %s (%s) — crop left in place",
            magface_path.name,
            exc,
        )
        return None
    unified = magface_data.get("unified_score", {})
    if isinstance(unified, dict) and "max" in unified:
        return float(unified["max"])
    return None


def demote_output_crops(
    modality: str,
    output_dir: Path,
    quality_threshold: float,
    input_dir: Path,
) -> int:
    """Re-evaluate crops already in output_dir against the current threshold.

    For each crop, reads its cached ``.magface.json`` (written when it passed)
    and moves it + sidecars back to input_dir when the cached max score no longer
    meets the threshold. Crops without a cached score are left in place with a
    warning (nothing can be re-evaluated without re-scoring, which would be a
    fresh assessment, not a demotion).

    Returns the number of demoted crops.
    """
    candidates = find_face_crops(output_dir)
    if not candidates:
        logger.info("[%s] Demote: no filtered crops to re-evaluate", modality)
        return 0

    demoted = 0
    collisions = 0
    errors = 0
    for dest_crop in tqdm(candidates, desc=f"Demote re-check ({modality})", unit="crop"):
        max_score = _cached_max_score(dest_crop.with_suffix(".magface.json"))
        if max_score is None or max_score >= quality_threshold:
            continue  # cannot re-evaluate, or still passes — keep it

        status = demote_crop(dest_crop, input_dir, output_dir)
        if status == "demoted":
            demoted += 1
        elif status == "demote_collision":
            collisions += 1
        else:
            errors += 1

    logger.info(
        "[%s] Demote re-check done. Demoted: %d  Collisions (left both): %d  Errors: %d",
        modality,
        demoted,
        collisions,
        errors,
    )
    return demoted
