# Face Mask Generation — System Card

## Overview

**System:** Face Mask Generation
**Type:** Rule-based algorithm
**Purpose:** Generate binary ArcFace-region masks from face-crop sidecars (white = the ArcFace quad)
**Provider:** DARDcollect

---

## Description

This system generates binary face-contour masks (255 = face region, 0 = background) from face crops and extracted frames. It does not run a detector in this stage. Instead, it reuses keypoints already produced upstream and stored in sidecar JSON files.

**Input:** Face crop sidecars (video `frame_data` / image top-level) carrying `face_crop_corners_arcface`
**Output:** Binary PNG masks alongside inputs — one per annotated video-crop frame (`<stem>_fNNNNNN_mask.png`), one per image crop (`<stem>_mask.png`)
**Mask value range:** 0-255 (binary: 0 = black background, 255 = white ArcFace region)

---

## Technical Details

- **Detector:** None (keypoint reuse from sidecars)
- **Algorithm:**
  1. Load the crop sidecar (no image decode needed — the quad is constant per crop).
  2. Read `face_crop_corners_arcface` (per `frame_data` entry for video, top-level for images).
  3. Create binary mask: white (255) inside the quad, black (0) elsewhere.
  4. Save next to the input (`<stem>_fNNNNNN_mask.png` per video frame, `<stem>_mask.png` per image).
- **Source frames** (`extracted_frames`) get one OFIQ quad mask per detected identity (`<frame stem>_track<id>_mask.png`).
- **Resumability:** Skips masks that already exist.

---

## Performance & Limitations

- **Speed:** CPU-only and lightweight (no model inference).
- **Accuracy:** Depends on upstream keypoint quality and confidence.
- **Limitations:**
  - Requires `face_crop_corners_arcface` in the sidecar; frames/entries without it are skipped.
  - The quad is constant per crop, so per-frame masks of one video are identical by construction.
  - The quad is a geometric region (the ArcFace identity ROI), not semantic segmentation.

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
