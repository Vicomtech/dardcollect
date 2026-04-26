"""
ArcFace face alignment geometry shared between extract_person_clips and
extract_face_crops.

Provides a single source of truth for the canonical landmark positions and the
function that converts smoothed keypoints into the 4 source-frame corners of
the face crop region.  Up to 5 landmarks are used when available (both eyes,
nose tip, and mouth corners at COCO-133 indices 71/77); alignment degrades
gracefully to 2-eye-only when face keypoints are absent or low-confidence.

Canonical positions follow the insightface/ArcFace convention (112×112).  This
produces tight, face-filling crops suitable as direct input to MagFace and, with
preprocessing adaptation, to OFIQ quality measures.  OFIQ's own 616×616
canonical cannot be used here: it assumes an already-cropped portrait photo as
input, whereas dardcollect extracts faces from full video frames where the face
may occupy only a small fraction of the scene.

The corners are stored in source-frame pixel coordinates (independent of the
final output_size), so downstream scripts can reconstruct the affine warp for
any desired resolution.
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

# Canonical canvas size (insightface ArcFace convention).
_CANONICAL_SIZE = 112

# 5-point ArcFace canonical positions on 112×112 (insightface convention).
# All coordinates are from the viewer's perspective.
# Order matches _ALIGN_5PT_INDICES below.
_ALIGN_5PT_INDICES = [_KPT_L_EYE, _KPT_R_EYE, _KPT_NOSE, _KPT_MOUTH_VL, _KPT_MOUTH_VR]
_ALIGN_5PT_DST = np.array(
    [
        [73.5318, 51.5014],  # person's left eye  → viewer's right  (x≈73)
        [38.2946, 51.6963],  # person's right eye → viewer's left   (x≈38)
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # viewer's left  mouth corner          (x≈41)
        [70.7299, 92.2041],  # viewer's right mouth corner          (x≈70)
    ],
    dtype=np.float32,
)


def face_crop_corners(
    keypoints: np.ndarray,
    kpt_scores: np.ndarray,
    align_face: bool,
    face_padding: float,
    keypoint_threshold: float,
    min_eye_distance_px: float,
) -> np.ndarray | None:
    """Return the 4 source-frame corners of the face crop region, or None.

    The returned array has shape (4, 2) with rows [TL, TR, BR, BL] in
    source-frame pixel coordinates.  They are computed independently of the
    final output_size — pass the first three rows to cv2.getAffineTransform
    together with the corresponding output corners to reconstruct M for any
    desired resolution:

        src = corners[:3]                        # TL, TR, BR in source frame
        dst = [[0,0],[S,0],[S,S]]                # output square corners
        M   = cv2.getAffineTransform(src, dst)

    :param keypoints: (K, 2) array of keypoint coordinates.
    :param kpt_scores: (K,) array of keypoint confidence scores.
    :param align_face: If True use ArcFace similarity transform; if False use
                       axis-aligned square centred on the midface.
    :param face_padding: Extra padding factor for the non-aligned path.
    :param keypoint_threshold: Minimum score to accept an eye keypoint.
    :param min_eye_distance_px: Minimum inter-eye distance in pixels.
    :return: (4, 2) float32 corner array, or None on failure.
    """
    if kpt_scores[_KPT_L_EYE] < keypoint_threshold or kpt_scores[_KPT_R_EYE] < keypoint_threshold:
        return None

    l_eye = keypoints[_KPT_L_EYE].astype(np.float32)
    r_eye = keypoints[_KPT_R_EYE].astype(np.float32)

    eye_dist = float(np.linalg.norm(r_eye - l_eye))
    if eye_dist < min_eye_distance_px:
        return None

    if align_face:
        # Collect whichever of the 5 canonical landmarks pass threshold.
        # Eyes are already validated above; nose/mouth are opportunistic.
        n_kpts = len(kpt_scores)
        src_list, dst_list = [], []
        for kpt_idx, canonical in zip(_ALIGN_5PT_INDICES, _ALIGN_5PT_DST):
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
        S = float(_CANONICAL_SIZE)
        out_corners = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float32)
        src_corners = cv2.transform(out_corners.reshape(1, -1, 2), M_inv).reshape(-1, 2)
    else:
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
