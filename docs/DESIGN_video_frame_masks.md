# Design — Video pre-processing masks (source-video frames + face-crop masks)

Status: **ratified and implemented 2026-08-04**. Written per [`.kilo/FEATURE_WORKFLOW.md`](../.kilo/FEATURE_WORKFLOW.md).

## Problem

The feature request asks for:

> Extract and save **N consecutive frames** from each video. For each extracted frame,
> detect whether a face is present. If a face is detected, generate and save a
> corresponding **bounding box mask** using the same filename as the frame — 255 inside
> the face bounding box, 0 everywhere else. A head-and-shoulders bounding box is also
> acceptable.

What ships today diverges in three ways, each measured against real data:

| # | Requirement | Implemented | Consequence |
| :-- | :--- | :--- | :--- |
| 1 | N consecutive frames | **every** frame of every clip (~330/clip) | 4.4 M frames ≈ 2.6 TB on the full clip set; 219 GB even on the filtered subset |
| 2 | Mask over the face region | `cv2.convexHull` of face landmarks 23–90 | landmarks trace the face oval and stop at the brow, so the scalp is cut off |
| 3 | Frames of the **video** | frames of 616×616 OFIQ face crops | the face fills the crop, so its bbox is ~the whole image — the mask discriminates nothing, and "if a face is detected" is always true |

Requirement 1 is the root cause of the disk pressure. It is a missing parameter, not a
storage problem.

## Architecture

Reuses the existing `frames` stage slot and its FAIR chain; changes what it reads and
what it writes. Person-clip sidecars already carry everything needed:

```
source_video : /…/archive_org_public_domain/videos/eng/<title>.mp4
start_frame  : 3486          end_frame: 4102
frame_data   : { "3486": [ {bbox, keypoints, keypoint_scores, score, track_id}, … ], … }
                 ^ keys are ABSOLUTE source-video frame numbers
```

Per clip sidecar:

1. Pick the first run of **N consecutive absolute frames** inside `[start_frame, end_frame]`
   whose `frame_data` entries contain a detection with usable face keypoints
   (`dardcollect.source_frames._has_usable_face`). This is the
   "timestamps where a person is known to be present" selection — it reuses detections the
   pipeline already paid for, no re-inference.
2. Seek those absolute frame numbers in `source_video` and write each frame as PNG.
3. Write **one mask per detection**, not one per frame: each tracked identity in that
   frame gets its own file, filled white inside that detection's ``face_crop_corners_ofiq``
   quadrilateral and black elsewhere. Those four points are recorded by
   ``dardcollect/face_geometry.py`` when it cuts the identity's face crop video, in source
   video coordinates — so the mask and the crop are the *same* region, not two
   approximations of it. Verified against real data: reprojecting the quad out of the
   source frame reproduces the crop video frame to a mean absolute difference of 3/255
   (p95 = 7), which is H.264 recompression, not geometry.

   The quad is **rotated** (OFIQ levels the eyes) and covers the whole head, hair
   included. Boxes derived from the face landmarks do not: landmarks 23-90 trace the face
   oval and stop at the brow, so every landmark-derived box cut the scalp off. That was
   the reason this design went through three rejected shapes before landing here.

   A detection with no ``face_crop_corners_ofiq`` gets **no** mask — it produced no face
   crop. That is the request's "if a face is detected" branch, and it stays observable in
   the summary counts.

Output is keyed by source video and absolute frame, so clips that overlap in time
converge on the same file instead of duplicating it:

```
extracted_frames/<lang>/<video_stem>/frame_003486.png
                                     frame_003486.json
                                     frame_003486_track007_mask.png   <- filled OFIQ crop quad
                                     frame_003486_track012_mask.png
```

Two deliberate deviations from the request, both ratified 2026-08-04:

- **Per-detection masks break "the same filename as the frame."** A frame with three
  people cannot have one mask named after it and still separate the identities, so the
  frame stem is suffixed with the track id. Correspondence is still by filename.
- **`track_id` is per-clip, not global.** The same person appearing in two clips gets two
  different ids, so these masks group detections *within* a frame, not identities across
  the dataset. Cross-clip re-identification is not part of this pipeline.

## FAIR impact

- No new CSV. `frames_extraction.csv` (lineage link 3, `docs/2-LINEAGE.md`) keeps its shape;
  `clip_uuid` still resolves, and each row gains the absolute `frame_number` it already has.
- Frame sidecars keep `add_fair_metadata` + `validate_against_schema` at write time.
- `docs/3-ANNOTATIONS.md` used to document masks at `video_face_crops/…_mask.png` (one per
  crop), which never matched the code. Corrected to the per-frame, per-identity location.
- The `video_face_crops` / `image_face_crops` modalities in `generate_face_masks.py` are
  unaffected and still find no `.jpg/.png` in a directory of `.mp4`; that mismatch is
  out of scope here and should be tracked separately.

## Resumability

Unchanged in kind: a frame is skipped when its `.png` and `.json` both exist (plus the
manifest re-listing fix already in `dardcollect/frames.py`). Because output is keyed by
source video + absolute frame, a rerun is idempotent even when a different clip selects an
overlapping window. Masks skip on `mask_path.exists()`.

## Cost

Measured on real data: 160 KB per source-resolution frame PNG, 2 KB per mask.

| N (per clip) | frames | disk |
| ---: | ---: | ---: |
| 5 | ~67 k | **~11 GB** |
| 10 | ~134 k | ~22 GB |
| 30 | ~402 k | ~65 GB |

`N = 5` against 79 GB free leaves the budget comfortable. `frame_extraction.min_free_disk_gb`
(added 2026-08-04) aborts the stage cleanly rather than filling a shared quota.

## Test plan

- Unit (`tests/test_source_frame_masks.py`, 12): N-consecutive selection respects `N` and
  breaks on gaps and faceless frames; masks are binary `{0,255}` and **not** axis-aligned
  (a rotated quad must not produce uniform row widths); a detection without crop corners
  produces no mask; reruns are idempotent; output path derives from source video +
  absolute frame. Each was verified to fail when its fix is reverted.
- Real data: reprojecting `face_crop_corners_ofiq` out of the source frame reproduces the
  actual crop video frame to a mean absolute difference of 3/255 (p95 = 7).
- Objective gate on the fixture — **currently blocked**: `configs/config.test.yaml` and
  `tests/fixtures/golden_manifest.json` do not exist on this machine.

## Decisions (ratified 2026-08-04)

1. **N is per clip**, i.e. per detected-person segment: 13,406 × 5 ≈ 67 k frames. Per
   source video would have given 103 × 5 = 515 frames — too few for a dataset.
2. **The OFIQ face-crop quad**, reusing ``face_crop_corners_ofiq`` from each detection.
   No tuning parameter, and consistent by construction with the crop videos. Rejected on
   the way here: the person ``bbox`` (whole body), a head-and-shoulders box from pose
   keypoints (cut at the brow), and a pose-skeleton silhouette (approximate, needed two
   invented thickness parameters).
3. **One mask per detection / identity**, suffixed with `track_id` (see above).

4. **The existence of a face crop gates the mask.** A detection filmed from behind has a
   person box and pose keypoints but no crop, so it gets no mask. This is the request's
   "if a face is detected" branch, and it subsumes the earlier keypoint-confidence gate:
   the crop corners only exist for detections that already passed the face-crop criteria.
   Revisit if recall on back-turned or very small subjects matters.

   Because the gate is a choice, it must be **visible in the output**, not silent: the
   stage reports how many detections were skipped, so its cost is measurable without
   re-running anything.
