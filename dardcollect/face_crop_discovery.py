"""Single source of truth for discovering OFIQ face crops on disk.

The filter stage, the opt-in demotion reconcile, and the quality-annotation stage
all need "the face crops under this directory, masks excluded". Defining that set
in three places let them drift: a new crop extension or a renamed mask would make
the filter and the demotion disagree on what a crop is, so demotion could move a
mask back as a crop or miss crops the filter would process. Extensions, the mask
suffix, and the scan live here once.
"""

from pathlib import Path

# Masks are written as ``<stem>_mask.png`` / ``<stem>_trackNNN_mask.png`` by
# generate_face_masks.py; they are not crops and must be excluded everywhere.
MASK_SUFFIX = "_mask.png"
CROP_GLOBS = ("*_face_*.mp4", "*_face_*.jpg", "*_face_*.png")


def find_face_crops(root: Path) -> list[Path]:
    """All face crops under *root* (dedup, sorted, mask files excluded)."""
    crops = sorted({p for glob in CROP_GLOBS for p in root.rglob(glob)})
    return [p for p in crops if not p.name.endswith(MASK_SUFFIX)]
