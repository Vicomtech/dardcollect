"""Post-processing utilities for object detection and pose estimation outputs.

Provides:
    - `simcc_decode`: Decode SimCC keypoint logits to coordinates and confidence.
    - `apply_ofiq_sigmoid_calibration`: Calibrate raw MagFace scores to OFIQ scale.
"""

import numpy as np


def simcc_decode(simcc_x, simcc_y, split_ratio=2.0):
    """Decode SimCC (Simple Coordinate Classification) logits to keypoints.

    SimCC represents keypoint coordinates as classification problems along x and y
    axes separately. This function finds the argmax positions and scales them back
    to the model input space.

    Args:
        simcc_x: X-axis classification logits, shape (1, K, W) where K is number
            of keypoints and W is the discretized x dimension.
        simcc_y: Y-axis classification logits, shape (1, K, H) where H is the
            discretized y dimension.
        split_ratio: Downsampling ratio of the SimCC head relative to the input
            resolution (default: 2.0, meaning the head is half the input size).

    Returns:
        tuple: (keypoints, scores)
            - keypoints: ndarray of shape (K, 2) with [x, y] coordinates in
              model-input space (not original image space).
            - scores: ndarray of shape (K,) with confidence scores per keypoint,
              computed as the minimum of the x and y max-logit values.
    """
    x_locs = np.argmax(simcc_x[0], axis=-1).astype(np.float32)
    y_locs = np.argmax(simcc_y[0], axis=-1).astype(np.float32)
    scores = np.minimum(np.max(simcc_x[0], axis=-1), np.max(simcc_y[0], axis=-1))
    kpts = np.stack([x_locs / split_ratio, y_locs / split_ratio], axis=-1)
    return kpts, scores


def apply_ofiq_sigmoid_calibration(
    raw_score: float, h: float = 100.0, x0: float = 23.0, w: float = 2.6
) -> float:
    """Calibrate a raw MagFace quality score to the OFIQ unified scale [0, 100].

    Maps the raw MagFace (IResNet50) model output to the OFIQ unified quality score
    following ISO/IEC 29794-5 using a sigmoid calibration function.

    Calibration formula::
        Q(x) = h / (1 + exp((x0 - x) / w))

    Default parameters are derived from OFIQ reference calibration:
        - h = 100.0: Score range [0, 100]
        - x0 = 23.0: Raw score at 50% quality
        - w = 2.6: Curve steepness

    Args:
        raw_score: Raw MagFace model output (typically in range ~[0, 50]).
        h: Sigmoid height/scale parameter (default: 100.0).
        x0: Sigmoid center point (default: 23.0).
        w: Sigmoid width/steepness parameter (default: 2.6).

    Returns:
        float: Calibrated quality score clamped to [0, 100].
    """
    sigmoid_val = 1.0 / (1.0 + np.exp((x0 - raw_score) / w))
    return max(0.0, min(100.0, h * sigmoid_val))
