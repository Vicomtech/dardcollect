"""
Kalman filter and tracking primitives for OC-SORT.

Extracted from tracker_ocsort.py to keep module size within limits.
Provides KalmanFilter, TrackState, and the numpy IoU fallback.
"""

from enum import Enum

import numpy as np
import scipy.linalg


def _numpy_bbox_ious(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    n, m = len(boxes1), len(boxes2)
    ious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            x1 = max(boxes1[i, 0], boxes2[j, 0])
            y1 = max(boxes1[i, 1], boxes2[j, 1])
            x2 = min(boxes1[i, 2], boxes2[j, 2])
            y2 = min(boxes1[i, 3], boxes2[j, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area1 = (boxes1[i, 2] - boxes1[i, 0]) * (boxes1[i, 3] - boxes1[i, 1])
            area2 = (boxes2[j, 2] - boxes2[j, 0]) * (boxes2[j, 3] - boxes2[j, 1])
            union = area1 + area2 - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious


class KalmanFilter:
    """Kalman filter for tracking bounding boxes in image space.

    8-dimensional state space: x, y, a, h, vx, vy, va, vh
    (center position, aspect ratio, height, and velocities).
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        )
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
