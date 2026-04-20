"""
Body keypoint estimation module using CIGPose Wholebody ONNX.
"""

import json
import logging

import cv2
import numpy as np
import onnxruntime as ort

from .config import DetectorConfig
from .onnx_utils import get_preferred_providers
from .postprocessing import simcc_decode

logger = logging.getLogger(__name__)

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class PoseEstimator:
    """CIGPose Wholebody estimator."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        model_path: str | None = None,
        mode: str = "performance",
        log_level: int | str = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        if not model_path:
            raise ValueError("PoseEstimator requires 'model_path'.")

        self._logger.info("Loading pose model: %s", model_path)

        gpu_id = config.gpu_id if config else 0
        providers = get_preferred_providers(device_id=gpu_id)
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            self._logger.warning("Failed to load pose model with high-performance providers: %s", e)
            self._logger.warning("Falling back to standard CUDA/CPU...")
            fallback_providers = [
                ("CUDAExecutionProvider", {"device_id": gpu_id}),
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(model_path, providers=fallback_providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        meta_str = self.session.get_modelmeta().custom_metadata_map.get("cigpose_meta")
        if meta_str:
            self.split_ratio = json.loads(meta_str).get("split_ratio", 2.0)
        else:
            self.split_ratio = 2.0

        self._logger.info(
            "Pose input shape: %s, split_ratio: %s", self.input_shape, self.split_ratio
        )

    def get_keypoints(
        self,
        image: np.ndarray,
        bbox: list[float],
        score_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1

        aspect = self.input_w / self.input_h
        if bw / max(bh, 1) > aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        bw *= 1.25
        bh *= 1.25

        im_h, im_w = image.shape[:2]
        sx1 = int(max(0, cx - bw / 2))
        sy1 = int(max(0, cy - bh / 2))
        sx2 = int(min(im_w, cx + bw / 2))
        sy2 = int(min(im_h, cy + bh / 2))

        crop = image[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            crop = image

        resized = cv2.resize(crop, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[np.newaxis]

        outputs = self.session.run(None, {self.input_name: tensor})
        kpts, scores = simcc_decode(outputs[0], outputs[1], self.split_ratio)

        scale_x = (sx2 - sx1) / self.input_w
        scale_y = (sy2 - sy1) / self.input_h
        kpts[:, 0] = kpts[:, 0] * scale_x + sx1
        kpts[:, 1] = kpts[:, 1] * scale_y + sy1

        return kpts.astype(np.float32), scores.astype(np.float32)

    def check_mouth_open(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        min_score: float = 0.3,
        open_threshold: float = 0.05,
    ) -> bool:
        """Check if mouth is open based on COCO-133 lip keypoints."""
        K_UPPER_LIP = 85
        K_LOWER_LIP = 89

        if len(scores) < 90:
            return False

        if scores[K_UPPER_LIP] < min_score or scores[K_LOWER_LIP] < min_score:
            return False

        mouth_dist = np.linalg.norm(keypoints[K_UPPER_LIP] - keypoints[K_LOWER_LIP])

        if scores[1] > min_score and scores[2] > min_score:
            ref_dist = np.linalg.norm(keypoints[1] - keypoints[2])
        elif scores[3] > min_score and scores[4] > min_score:
            ref_dist = np.linalg.norm(keypoints[3] - keypoints[4]) * 0.5
        else:
            return False

        if ref_dist == 0:
            return False

        return bool((mouth_dist / ref_dist) > open_threshold)
