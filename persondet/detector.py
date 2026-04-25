"""
Person detection module using YOLOX-Tiny HumanArt ONNX (end2end, NMS included).
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort

from .config import DetectorConfig
from .onnx_utils import get_preferred_providers

logger = logging.getLogger(__name__)

_PERSON_CLASS = 0


class PersonDetector:
    """Person detector using YOLOX-Tiny HumanArt end2end ONNX."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)

        if not model_path:
            raise ValueError("PersonDetector requires 'model_path'.")

        self._logger.info("Loading detector model: %s", model_path)

        gpu_id = config.gpu_id if config else 0
        providers = get_preferred_providers(device_id=gpu_id)
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            self._logger.warning("Failed to load model with high-performance providers: %s", e)
            self._logger.warning("Falling back to standard CUDA/CPU...")
            fallback_providers = [
                ("CUDAExecutionProvider", {"device_id": gpu_id}),
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(model_path, providers=fallback_providers)

        self._logger.info("Detector active providers: %s", self.session.get_providers())

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h = inp.shape[2]  # 416
        self.input_w = inp.shape[3]  # 416

    def _letterbox(self, img: np.ndarray):
        """Letterbox to model input size. Returns (tensor, ratio, pad_l, pad_t)."""
        h, w = img.shape[:2]
        ratio = min(self.input_h / h, self.input_w / w)
        new_w = round(w * ratio)
        new_h = round(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = self.input_w - new_w
        dh = self.input_h - new_h
        pad_l = dw // 2
        pad_t = dh // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_t,
            dh - pad_t,
            pad_l,
            dw - pad_l,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = padded.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
        return tensor, ratio, pad_l, pad_t

    def get_detections(
        self,
        image: np.ndarray,
        score_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return person bounding boxes [x1,y1,x2,y2] and scores."""
        h, w = image.shape[:2]

        tensor, ratio, pad_l, pad_t = self._letterbox(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        dets, labels = outputs[0], outputs[1]

        # dets:   (1, N, 5) = [x1, y1, x2, y2, score] in letterboxed space
        # labels: (1, N)    = class indices (NMS already applied by end2end model)
        # Convert from ONNX output (may be sparse or dense) to numpy arrays
        dets = np.asarray(dets)[0]  # (N, 5)
        labels = np.asarray(labels)[0]  # (N,)

        if len(dets) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = dets[:, :4].copy()
        scores = dets[:, 4]

        mask = (labels == _PERSON_CLASS) & (scores >= score_threshold)
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Undo letterbox: remove padding then undo scale
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / ratio

        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

        return boxes.astype(np.float32), scores.astype(np.float32)
