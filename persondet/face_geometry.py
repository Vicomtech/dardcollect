"""
ArcFace and OFIQ face alignment geometry shared between extract_person_clips and
extract_face_crops.

Provides a single source of truth for the two canonical landmark formats and the
function that converts smoothed keypoints into the 4 source-frame corners of a
face crop region. Up to 5 landmarks are used when available (both eyes, nose tip,
and mouth corners at COCO-133 indices 71/77); alignment degrades gracefully to
2-eye-only when face keypoints are absent or low-confidence.

Two aligned crop modes are supported:

  "arcface" (ARCFACE_SIZE = 112×112) — insightface/ArcFace convention.  Tight,
    face-filling crops suitable as direct input to MagFace (IResNet50) and
    ArcFace identity embedding.  Used by filter_face_crops_by_quality.py.

  "ofiq" (OFIQ_SIZE = 616×616) — BSI-OFIQ convention.  Wider framing designed
    to match the internal aligned-face format expected by OFIQ quality measures:
    sharpness, expression neutrality, head pose, compression artifacts,
    background uniformity, and face occlusion.  Used by annotate_face_quality.py.

  "unaligned" — axis-aligned square centred on the midface; not used by the
    standard pipeline but retained for debugging.

The corners are stored in source-frame pixel coordinates (independent of the
final output_size), so downstream scripts can reconstruct the affine warp for
any desired resolution.

Why two formats?  The OFIQ quality models (sharpness CNN, expression
EfficientNet, head pose MobileNet, etc.) were calibrated on OFIQ's 616×616
aligned faces where the face occupies roughly the central 40% of the canvas and
the eyes sit at y≈272.  ArcFace crops are too tight for those models.
Conversely, MagFace (IResNet50) was trained on ArcFace-aligned 112×112 crops
and expects that framing.  The two subdirectories produced by
extract_face_crops.py feed the correct format to each downstream step.
"""

import cv2
import numpy as np

# Keypoint indices (COCO-133 wholebody convention)
_KPT_NOSE = 0
_KPT_L_EYE = 1  # person's left eye  = viewer's right
_KPT_R_EYE = 2  # person's right eye = viewer's left
# Face keypoints (COCO-133 = 23 + dlib-68 index, viewer's perspective)
_KPT_MOUTH_VL = 71  # dlib 48, viewer's left  mouth corner
_KPT_MOUTH_VR = 77  # dlib 54, viewer's right mouth corner

# ── ArcFace 112×112 canonical (insightface convention) ────────────────────────
ARCFACE_SIZE = 112

# x increases left→right (viewer's perspective).
# Person's left eye is on the right side of the image (viewer's right).
_ALIGN_ARCFACE_INDICES = [_KPT_L_EYE, _KPT_R_EYE, _KPT_NOSE, _KPT_MOUTH_VL, _KPT_MOUTH_VR]
_ALIGN_ARCFACE_DST = np.array(
    [
        [73.5318, 51.5014],  # person's left eye  → viewer's right  (x≈73)
        [38.2946, 51.6963],  # person's right eye → viewer's left   (x≈38)
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # viewer's left  mouth corner          (x≈41)
        [70.7299, 92.2041],  # viewer's right mouth corner          (x≈70)
    ],
    dtype=np.float32,
)

# ── OFIQ 616×616 canonical (BSI-OFIQ convention) ──────────────────────────────
OFIQ_SIZE = 616

# Eyes at y≈272, nose at y≈336, mouth corners at y≈402 on a 616-tall canvas.
# x=251 is viewer's left (person's right eye); x=364 is viewer's right (person's left eye).
_ALIGN_OFIQ_INDICES = [_KPT_R_EYE, _KPT_L_EYE, _KPT_NOSE, _KPT_MOUTH_VL, _KPT_MOUTH_VR]
_ALIGN_OFIQ_DST = np.array(
    [
        [251.0, 272.0],  # person's right eye → viewer's left  (x=251)
        [364.0, 272.0],  # person's left eye  → viewer's right (x=364)
        [308.0, 336.0],  # nose tip
        [262.0, 402.0],  # viewer's left  mouth corner
        [355.0, 402.0],  # viewer's right mouth corner
    ],
    dtype=np.float32,
)


def face_crop_corners(
    keypoints: np.ndarray,
    kpt_scores: np.ndarray,
    mode: str,
    keypoint_threshold: float,
    min_eye_distance_px: float,
    face_padding: float = 0.0,
) -> np.ndarray | None:
    """Return the 4 source-frame corners of the face crop region, or None.

    The returned array has shape (4, 2) with rows [TL, TR, BR, BL] in
    source-frame pixel coordinates. Pass the first three rows to
    cv2.getAffineTransform with the corresponding output corners:

        src = corners[:3]              # TL, TR, BR in source frame
        dst = [[0,0],[S,0],[S,S]]      # output square corners
        M   = cv2.getAffineTransform(src, dst)

    :param keypoints: (K, 2) array of keypoint coordinates.
    :param kpt_scores: (K,) array of keypoint confidence scores.
    :param mode: "arcface" (112×112), "ofiq" (616×616), or "unaligned".
    :param keypoint_threshold: Minimum score to accept a keypoint.
    :param min_eye_distance_px: Minimum inter-eye distance in pixels.
    :param face_padding: Extra padding factor; only used for mode="unaligned".
    :return: (4, 2) float32 corner array, or None on failure.
    """
    if kpt_scores[_KPT_L_EYE] < keypoint_threshold or kpt_scores[_KPT_R_EYE] < keypoint_threshold:
        return None

    l_eye = keypoints[_KPT_L_EYE].astype(np.float32)
    r_eye = keypoints[_KPT_R_EYE].astype(np.float32)

    eye_dist = float(np.linalg.norm(r_eye - l_eye))
    if eye_dist < min_eye_distance_px:
        return None

    if mode in ("arcface", "ofiq"):
        if mode == "arcface":
            indices, dst_pts_full, output_size = (
                _ALIGN_ARCFACE_INDICES,
                _ALIGN_ARCFACE_DST,
                ARCFACE_SIZE,
            )
        else:
            indices, dst_pts_full, output_size = _ALIGN_OFIQ_INDICES, _ALIGN_OFIQ_DST, OFIQ_SIZE

        n_kpts = len(kpt_scores)
        src_list, dst_list = [], []
        for kpt_idx, canonical in zip(indices, dst_pts_full):
            if kpt_idx >= n_kpts or kpt_scores[kpt_idx] < keypoint_threshold:
                continue
            src_list.append(keypoints[kpt_idx].astype(np.float32))
            dst_list.append(canonical)

        src_pts = np.array(src_list, dtype=np.float32)
        dst_pts = np.array(dst_list, dtype=np.float32)
        M, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            return None
        M_inv = cv2.invertAffineTransform(M)
        S = float(output_size)
        out_corners = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float32)
        src_corners = cv2.transform(out_corners.reshape(1, -1, 2), M_inv).reshape(-1, 2)
    else:  # "unaligned"
        eye_mid = (l_eye + r_eye) * 0.5
        face_center = eye_mid + np.array([0.0, eye_dist * 0.5], dtype=np.float32)
        half = eye_dist * 1.5 * (1.0 + face_padding)
        cx, cy = float(face_center[0]), float(face_center[1])
        src_corners = np.array(
            [
                [cx - half, cy - half],  # TL
                [cx + half, cy - half],  # TR
                [cx + half, cy + half],  # BR
                [cx - half, cy + half],  # BL
            ],
            dtype=np.float32,
        )

    return src_corners
