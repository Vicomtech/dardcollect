"""
OFIQ per-frame quality measures (the 7 ISO/IEC 29794-5 dimensions) + their
shared constants + the OFIQ sigmoid calibration.

Split out of `quality.py` so neither file is a god-file. `quality.py` imports
these measure functions (one-way dependency — no circular import) and calls them
from `score_frame_all`.
"""

from typing import cast

import cv2
import numpy as np
import onnxruntime as ort


def _ofiq_sigmoid(x: float, h: float, a: float, s: float, x0: float, w: float) -> float:
    sig = 1.0 / (1.0 + np.exp((x0 - x) / w))
    score = h * (a + s * sig)
    return float(np.clip(score, 0.0, 100.0))


_HP_MEAN = np.array(
    [
        3.4926363e-04,
        2.5279013e-07,
        -6.8751979e-07,
        6.0167957e01,
        -6.2955132e-07,
        5.7572004e-04,
        -5.0853912e-05,
    ],
    dtype=np.float32,
)
_HP_STD = np.array(
    [
        1.76321526e-04,
        6.73794348e-05,
        4.47084894e-04,
        2.65502319e01,
        1.23137695e-04,
        4.49302170e-05,
        7.92367064e-05,
    ],
    dtype=np.float32,
)

# BiSeNet class indices for head-covering pixels
_BISENET_CLOTH_IDX = 16
_BISENET_HAT_IDX = 18

# ImageNet mean/std (BGR order for OpenCV)
_IMAGENET_MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255.0
_IMAGENET_STD_BGR = np.array([0.225, 0.224, 0.229], dtype=np.float32) * 255.0


def _sharpness_score(frame_bgr: np.ndarray, rtrees: cv2.ml.RTrees, n_trees: int) -> float:
    """Compute OFIQ Sharpness raw score from feature vector, then calibrate."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.ones_like(gray, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    features: list[float] = []

    # Laplacian features (kernel sizes 1, 3, 5, 7, 9)
    for k in (1, 3, 5, 7, 9):
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=k)
        m, s = cv2.meanStdDev(np.abs(lap), mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    # MeanDiff features (kernel sizes 3, 5, 7)
    for k in (3, 5, 7):
        b = cv2.blur(gray, (k, k))
        diff = cv2.absdiff(gray.astype(np.float64), b.astype(np.float64))
        m, s = cv2.meanStdDev(diff, mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    # Sobel features (kernel sizes 1, 3, 5, 7, 9)
    for k in (1, 3, 5, 7, 9):
        sob = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 1, ksize=k)
        m, s = cv2.meanStdDev(np.abs(sob), mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    feat = np.array(features, dtype=np.float32).reshape(1, -1)
    pred_result = np.zeros((1, 1), dtype=np.float32)
    rtrees.predict(feat, pred_result, flags=cv2.ml.StatModel_RAW_OUTPUT)
    raw = float(n_trees) - float(pred_result[0, 0])
    # OFIQ sigmoid: h=1, a=-14.0, s=115.0, x0=-20.0, w=15.0
    return _ofiq_sigmoid(raw, h=1.0, a=-14.0, s=115.0, x0=-20.0, w=15.0)


def _compression_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ CompressionArtifacts score."""
    img = cv2.resize(frame_bgr, (248, 248), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb -= np.array([123.7, 116.3, 103.5], dtype=np.float32)
    img_rgb /= np.array([58.4, 57.1, 57.4], dtype=np.float32)
    blob = cv2.dnn.blobFromImage(img_rgb, scalefactor=1.0, size=(248, 248), mean=0, swapRB=False)
    output = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: blob}))
    raw = float(output[0][0, 0])
    # OFIQ sigmoid: h=1, a=-0.0278, s=103.0, x0=0.3308, w=0.092
    return _ofiq_sigmoid(raw, h=1.0, a=-0.0278, s=103.0, x0=0.3308, w=0.092)


def _expression_neutrality_score(
    frame_bgr: np.ndarray,
    enet_b0: ort.InferenceSession,
    enet_b2: ort.InferenceSession,
    adaboost: cv2.ml.Boost,
) -> float:
    """Compute OFIQ ExpressionNeutrality score."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    # B0: 224×224
    img_b0 = cv2.resize(img_norm, (224, 224), interpolation=cv2.INTER_LINEAR)
    blob_b0 = cv2.dnn.blobFromImage(img_b0, scalefactor=1.0, size=(224, 224), mean=0, swapRB=False)
    feat1 = cast(
        np.ndarray, enet_b0.run(None, {enet_b0.get_inputs()[0].name: blob_b0})[0]
    )  # (1, 1280)

    # B2: 260×260
    img_b2 = cv2.resize(img_norm, (260, 260), interpolation=cv2.INTER_LINEAR)
    blob_b2 = cv2.dnn.blobFromImage(img_b2, scalefactor=1.0, size=(260, 260), mean=0, swapRB=False)
    feat2 = cast(
        np.ndarray, enet_b2.run(None, {enet_b2.get_inputs()[0].name: blob_b2})[0]
    )  # (1, 1408)

    features = np.concatenate([feat1, feat2], axis=1).astype(np.float32)  # (1, 2688)
    pred_result = np.zeros((1, 1), dtype=np.float32)
    adaboost.predict(features, pred_result, flags=cv2.ml.DTrees_PREDICT_SUM)
    raw = float(pred_result[0, 0])
    # OFIQ sigmoid: h=100, x0=-5000.0, w=5000.0, a=0 (default), s=1 (default)
    return _ofiq_sigmoid(raw, h=100.0, a=0.0, s=1.0, x0=-5000.0, w=5000.0)


def _no_head_coverings_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ NoHeadCoverings score via BiSeNet face parsing."""
    img = cv2.resize(frame_bgr, (400, 400), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb - _IMAGENET_MEAN_BGR[[2, 1, 0]]) / _IMAGENET_STD_BGR[[2, 1, 0]]
    blob = cv2.dnn.blobFromImage(img_norm, scalefactor=1.0, size=(400, 400), mean=0, swapRB=False)
    logits = cast(
        np.ndarray, session.run(None, {session.get_inputs()[0].name: blob})[0]
    )  # (1, 19, 400, 400)

    seg_map = np.argmax(logits[0], axis=0).astype(np.uint8)  # (400, 400)

    # OFIQ crops the bottom 204 rows before counting; apply proportional crop (51%)
    crop_bottom = round(204 / 400 * seg_map.shape[0])
    seg_cropped = seg_map[: seg_map.shape[0] - crop_bottom, :]

    total = seg_cropped.size
    cloth_px = int(np.sum(seg_cropped == _BISENET_CLOTH_IDX))
    hat_px = int(np.sum(seg_cropped == _BISENET_HAT_IDX))
    raw = (cloth_px + hat_px) / total if total > 0 else 0.0

    # Piecewise quality mapping from OFIQ (T0=0.0, T1=0.95, x0=0.02, w=0.1)
    T0, T1, x0, w = 0.0, 0.95, 0.02, 0.1
    if raw <= T0:
        return 100.0
    if raw >= T1:
        return 0.0
    s = 1.0 / (1.0 + np.exp((x0 - raw) / w))
    s0 = 1.0 / (1.0 + np.exp((x0 - T0) / w))
    s1 = 1.0 / (1.0 + np.exp((x0 - T1) / w))
    q = (s1 - s) / (s1 - s0)
    return float(np.clip(round(100.0 * q), 0.0, 100.0))


def _face_occlusion_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ FaceOcclusionPrevention score.

    Returns 100 when no occlusion is detected, 0 for fully occluded.
    """
    img = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    # swapRB=True converts BGR→RGB; normalize [0, 1]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(224, 224), mean=0, swapRB=True)
    out = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: blob}))
    # OFIQ: negate output, then THRESH_BINARY_INV(thresh=0) gives 1 when original ≥ 0.
    # So raw_output ≥ 0 → non-occluded face; raw_output < 0 → occluded / background.
    raw_map = out[-1][0, 0]  # (H, W)
    non_occluded_frac = float(np.sum(raw_map >= 0)) / raw_map.size
    return float(np.clip(round(100.0 * non_occluded_frac), 0.0, 100.0))


def _head_pose_angles(
    frame_bgr: np.ndarray, session: ort.InferenceSession
) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) Euler angles in degrees using 3DDFAV2.

    Preprocessing and decoding from OFIQ HeadPose3DDFAV2.cpp.
    """
    img = cv2.resize(frame_bgr, (120, 120), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = (img - 127.5) / 128.0
    # HWC → NCHW (BGR channel order, matching C++ code)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(120, 120), mean=0, swapRB=False)
    raw = cast(np.ndarray, session.run(None, {session.get_inputs()[0].name: blob})[0])  # (1, 62)

    # Denormalise first 7 parameters
    params = raw[0, :7] * _HP_STD + _HP_MEAN

    # Build rotation matrix from two row vectors
    r0 = params[:3]
    r1 = params[4:]
    norm0 = np.linalg.norm(r0)
    norm1 = np.linalg.norm(r1)
    if norm0 < 1e-8 or norm1 < 1e-8:
        return 0.0, 0.0, 0.0
    r0 = r0 / norm0
    r1 = r1 / norm1
    r2 = np.cross(r0, r1)
    R = np.stack([r0, r1, r2]).T  # 3×3, transposed as in C++ code

    # Euler angles (ZYX convention as in OFIQ)
    THRES = 0.9975
    r31 = float(R[2, 0])
    if -THRES <= r31 <= THRES:
        pitch = np.arcsin(r31)
        c = np.cos(pitch)
        yaw = -np.arctan2(R[2, 1] / c, R[2, 2] / c)
        roll = -np.arctan2(R[1, 0] / c, R[0, 0] / c)
    elif r31 < -THRES:
        pitch = -np.pi / 2
        yaw = -np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    else:
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0

    return (
        float(yaw * 180.0 / np.pi),
        float(pitch * 180.0 / np.pi),
        float(roll * 180.0 / np.pi),
    )


def _angle_to_quality(angle_deg: float) -> float:
    """OFIQ HeadPose quality: round(100 * max(0, cos(angle))²)."""
    c = np.cos(angle_deg * np.pi / 180.0)
    c = max(0.0, float(c))
    return float(round(100.0 * c * c))
