# Face Mask Generation — System Card

## Overview

**System:** Face Mask Generation  
**Type:** Rule-based algorithm  
**Purpose:** Generate binary bounding box masks for detected faces in cropped face images  
**Provider:** DARDcollect  

---

## Description

This system generates binary face bounding box masks (255 = face region, 0 = background) from cropped face images. It uses the YOLOX-Tiny person detector to identify the face bounding box within each crop and creates a corresponding binary mask image.

**Input:** Face crop images (video or image modality)  
**Output:** Binary PNG mask files with same dimensions as input crops  
**Mask value range:** 0–255 (binary: 0 = black background, 255 = white face region)

---

## Technical Details

- **Detector:** YOLOX-Tiny (HumanArt) — same as main detection pipeline
- **Algorithm:** 
  1. Read face crop image
  2. Run YOLOX-Tiny face detection on crop
  3. Extract bounding box of first detected face
  4. Create binary mask: white (255) inside bbox, black (0) elsewhere
  5. Save as `<crop_name>_mask.png` alongside crop
  
- **Resumability:** Skips crops that already have corresponding `_mask.png` files

---

## Performance & Limitations

- **Speed:** <100ms per crop (GPU-accelerated)
- **Accuracy:** Bounding box accuracy inherited from YOLOX-Tiny detector
- **Limitations:**
  - Assumes one face per crop (standard for face crop dataset)
  - Bbox may not perfectly align with face boundaries (detection model dependent)
  - No special handling for partial occlusions or profile views

---

## FAIR Provenance

- **Input tracking:** Masks inherit provenance from parent face crops via 1:1 filename mapping
- **CSV integration:** Masks do NOT appear in lineage CSVs (they are derived annotations, not primary artifacts)
- **Metadata:** Mask file dimensions must match crop dimensions (validated at write time)

---

## EU AI Act Annex IV Compliance

**High-Risk AI System:** No — this is a rule-based algorithm with no learned models (uses pre-trained detector).

---

## References

- YOLOX-Tiny HumanArt model: See [dardcollect/models/README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md](README_yolox_tiny_8xb8-300e_humanart-6f3252f9.md)
