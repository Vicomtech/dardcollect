# Face Mask Generation — System Card

## Overview

**System:** Face Mask Generation  
**Type:** Rule-based algorithm  
**Purpose:** Generate binary face-contour masks from existing sidecar keypoints  
**Provider:** DARDcollect

---

## Description

This system generates binary face-contour masks (255 = face region, 0 = background) from face crops and extracted frames. It does not run a detector in this stage. Instead, it reuses keypoints already produced upstream and stored in sidecar JSON files.

**Input:** Face crop/frame images with `.json` sidecars containing keypoints  
**Output:** Binary PNG masks written alongside inputs as `<name>_mask.png`  
**Mask value range:** 0-255 (binary: 0 = black background, 255 = white face region)

---

## Technical Details

- **Detector:** None (keypoint reuse from sidecars)
- **Algorithm:**
  1. Read crop/frame image.
  2. Load sidecar keypoints + keypoint scores (`133 x 2` + `133`).
  3. Keep face-landmark subset (indices 23-90) above confidence threshold.
  4. Compute convex hull over valid face landmarks.
  5. Create binary mask: white (255) inside hull, black (0) elsewhere.
  6. Save as `<name>_mask.png` next to the input image.
- **Resumability:** Skips inputs that already have `_mask.png`.

---

## Performance & Limitations

- **Speed:** CPU-only and lightweight (no model inference).
- **Accuracy:** Depends on upstream keypoint quality and confidence.
- **Limitations:**
  - Requires valid sidecar keypoints; missing/invalid sidecars are skipped.
  - Very sparse or low-confidence face landmarks can produce empty masks.
  - Convex hull is a geometric contour approximation, not semantic segmentation.

---

## FAIR Provenance

- **Input tracking:** Masks inherit provenance from parent crops/frames via filename + sidecar linkage.
- **CSV integration:** Masks do not appear in lineage CSVs (derived annotations, not primary artifacts).
- **Metadata:** Mask dimensions match the source image dimensions.

---

## EU AI Act Annex IV Compliance

**High-Risk AI System:** No — this stage is a deterministic rule-based post-process with no learned model execution.

---

## References

- Implementation: [pipeline/generate_face_masks.py](../../pipeline/generate_face_masks.py)
