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
    ArcFace identity embedding.

  "ofiq" (OFIQ_SIZE = 616×616) — BSI-OFIQ convention.  Wider framing designed
    to match the internal aligned-face format expected by OFIQ quality measures:
    sharpness, expression neutrality, head pose, compression artifacts,
    background uniformity, and face occlusion.  Output of extract_face_crops.py.

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
and expects that framing.

extract_face_crops.py produces only OFIQ-format videos.  Because both formats
align to fixed canonical positions, the ArcFace region is always the same
parallelogram within any OFIQ frame.  ARCFACE_CROP_CORNERS_IN_OFIQ provides
that constant region; arcface_from_ofiq_frame() extracts the 112×112 crop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dardcollect.config import FaceCropConfig
    from dardcollect.tracker import Segment

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


# ── Precomputed ArcFace region in OFIQ canonical space ────────────────────────
# Both formats use fixed canonical positions, so the ArcFace 112×112 region is
# always the same parallelogram in every OFIQ 616×616 frame.  Precompute once.

_OFIQ_KPT_SRC = np.array(
    [[251.0, 272.0], [364.0, 272.0], [308.0, 336.0], [262.0, 402.0], [355.0, 402.0]],
    dtype=np.float32,
)  # R_EYE, L_EYE, NOSE, MOUTH_VL, MOUTH_VR — OFIQ canonical positions
_ARCFACE_KPT_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)  # same landmarks in ArcFace canonical positions
_M_ofiq_to_arcface, _ = cv2.estimateAffinePartial2D(
    _OFIQ_KPT_SRC, _ARCFACE_KPT_DST, method=cv2.LMEDS
)
_M_arcface_in_ofiq = cv2.invertAffineTransform(_M_ofiq_to_arcface)
_arcface_out_corners = np.array(
    [[0.0, 0.0], [ARCFACE_SIZE, 0.0], [ARCFACE_SIZE, ARCFACE_SIZE], [0.0, ARCFACE_SIZE]],
    dtype=np.float32,
)
ARCFACE_CROP_CORNERS_IN_OFIQ: np.ndarray = cv2.transform(
    _arcface_out_corners.reshape(1, -1, 2), _M_arcface_in_ofiq
).reshape(-1, 2)
"""4 corners (TL, TR, BR, BL) of the ArcFace 112x112 region in OFIQ 616x616 pixel space.

Constant for all OFIQ-aligned frames.  Store in sidecar JSONs so consumers can
extract ArcFace crops without re-estimating the transform.  Pass corners[:3] to
cv2.getAffineTransform to warp an OFIQ frame to 112x112 ArcFace format.
"""


# ── Script-level geometry helpers (consolidated from extraction scripts) ──────


def _bbox_iou(a: list, b: list) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _transform_bbox(bbox: list, M: np.ndarray) -> list:
    """Transform [x1, y1, x2, y2] bbox through affine matrix M, returning axis-aligned result."""
    x1, y1, x2, y2 = bbox
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
    transformed = (M @ corners.T).T
    tx1, ty1 = transformed[:, 0].min(), transformed[:, 1].min()
    tx2, ty2 = transformed[:, 0].max(), transformed[:, 1].max()
    return [round(tx1, 2), round(ty1, 2), round(tx2, 2), round(ty2, 2)]


def _corners_to_warp(
    frame: np.ndarray,
    corners: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Warp *frame* to an output_size square given 4 source-frame corners [TL,TR,BR,BL]."""
    S = output_size
    src = corners[:3].astype(np.float32)
    dst = np.array([[0, 0], [S, 0], [S, S]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _transform_keypoints(
    keypoints: list,
    keypoint_scores: list,
    keypoints_source_array: np.ndarray,
    kpt_scores_array: np.ndarray,
    output_size: int,
) -> tuple[list, list, np.ndarray | None]:
    """Transform keypoints to the output crop space using the OFIQ alignment transform.

    Returns:
        (transformed_keypoints, keypoint_scores, affine_matrix) where affine_matrix
        is the 2×3 matrix used, or None if the transform could not be computed.
    """
    if len(keypoints_source_array) == 0:
        return [], keypoint_scores, None

    indices = _ALIGN_OFIQ_INDICES
    dst_pts_full = _ALIGN_OFIQ_DST

    n_kpts = len(kpt_scores_array)
    src_list, dst_list = [], []
    for kpt_idx, canonical in zip(indices, dst_pts_full):
        if kpt_idx >= n_kpts or kpt_scores_array[kpt_idx] < 0.2:
            continue
        src_list.append(keypoints_source_array[kpt_idx].astype(np.float32))
        dst_list.append(canonical)

    if len(src_list) < 3:
        return keypoints, keypoint_scores, None

    src_pts = np.array(src_list, dtype=np.float32)
    dst_pts = np.array(dst_list, dtype=np.float32)
    M, _inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

    if M is None:
        return keypoints, keypoint_scores, None

    transformed = []
    for kpt in keypoints_source_array:
        pt = np.array([float(kpt[0]), float(kpt[1]), 1.0])
        transformed_pt = M @ pt
        transformed.append([float(transformed_pt[0]), float(transformed_pt[1])])

    return transformed, keypoint_scores, M


def _get_or_compute_corners(
    det: dict,
    face_config,
    frame_id: str = "",
) -> np.ndarray | None:
    """Get corners from detection dict or compute from keypoints.

    Args:
        det: Detection dict with optional 'face_crop_corners_ofiq' field.
        face_config: Config object with ``min_eye_distance_px`` and
            optionally ``pose_keypoint_threshold`` attributes.
        frame_id: Optional frame ID for logging.

    Returns:
        (4, 2) float32 corner array or None if computation fails.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    # Return pre-computed corners if available
    if "face_crop_corners_ofiq" in det:
        corners = np.array(det["face_crop_corners_ofiq"], dtype=np.float32)
        if corners.shape == (4, 2):
            return corners

    # Otherwise compute from keypoints
    keypoints = det.get("keypoints", [])
    keypoint_scores = det.get("keypoint_scores", [])

    if not keypoints or not keypoint_scores:
        return None

    kpts_array = np.array(keypoints, dtype=np.float32)
    scores_array = np.array(keypoint_scores, dtype=np.float32)

    # Use a lower threshold (0.2 instead of 0.3) to capture more marginal detections
    corners = face_crop_corners(
        keypoints=kpts_array,
        kpt_scores=scores_array,
        mode="ofiq",
        keypoint_threshold=0.2,
        min_eye_distance_px=face_config.min_eye_distance_px,
    )

    if corners is None and len(keypoint_scores) >= 3:
        eye_scores = [
            keypoint_scores[1] if len(keypoint_scores) > 1 else 0,
            keypoint_scores[2] if len(keypoint_scores) > 2 else 0,
        ]
        _log.debug(
            "[%s] Corner computation failed. Eye scores: %s, threshold: 0.2",
            frame_id,
            eye_scores,
        )

    return corners


def _annotate_face_crop_corners(seg: Segment, fcfg: FaceCropConfig) -> None:
    """Add face crop corners (arcface and ofiq) to each detection entry.

    Called after smooth_segment_keypoints so corners reflect the smoothed
    positions.  Corners are 4 source-frame points [TL, TR, BR, BL] stored
    as a list of [x, y] pairs, independent of output_size.
    """
    for frame_detections in seg.frame_data.values():
        for person in frame_detections:
            if "keypoints" not in person or "keypoint_scores" not in person:
                continue
            kpts = np.array(person["keypoints"], dtype=np.float32)
            kscores = np.array(person["keypoint_scores"], dtype=np.float32)
            for mode in ("arcface", "ofiq"):
                corners = face_crop_corners(
                    kpts,
                    kscores,
                    mode=mode,
                    keypoint_threshold=fcfg.pose_keypoint_threshold,
                    min_eye_distance_px=fcfg.min_eye_distance_px,
                )
                if corners is not None:
                    person[f"face_crop_corners_{mode}"] = [
                        [round(float(x), 2), round(float(y), 2)] for x, y in corners
                    ]


def arcface_from_ofiq_frame(ofiq_frame: np.ndarray) -> np.ndarray:
    """Extract a 112×112 ArcFace-aligned crop from a 616×616 OFIQ-aligned frame."""
    src = ARCFACE_CROP_CORNERS_IN_OFIQ[:3].astype(np.float32)
    dst = np.array(
        [[0.0, 0.0], [float(ARCFACE_SIZE), 0.0], [float(ARCFACE_SIZE), float(ARCFACE_SIZE)]],
        dtype=np.float32,
    )
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        ofiq_frame,
        M,
        (ARCFACE_SIZE, ARCFACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
