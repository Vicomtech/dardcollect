"""MagFace model utilities for face quality assessment.

MagFace (IResNet50) is trained on MS1MV2 with MagFace loss and measures how
confidently a face recognition model can embed a given crop — a standard proxy
for biometric sample quality following ISO/IEC 29794-5 (OFIQ).

Reference: https://github.com/IrvingMeng/MagFace
"""

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MAGFACE_INPUT_SIZE = 112  # IResNet50 input resolution


def load_magface(model_path: str, gpu_id: int) -> ort.InferenceSession:
    """Load the MagFace ONNX model onto the requested GPU (or CPU as fallback).

    :param model_path: Path to magface_iresnet50_norm.onnx, or "" for default.
    :param gpu_id: CUDA device ID.
    :return: ONNX Runtime inference session.
    :raises FileNotFoundError: If model file does not exist.
    """
    path = (
        Path(model_path)
        if model_path
        else Path(__file__).parent / "models" / "magface_iresnet50_norm.onnx"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"MagFace model not found at {path}.\n"
            f"Expected at persondet/models/magface_iresnet50_norm.onnx.\n"
            f"Obtain it from the OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project"
        )

    providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    logger.info("Loaded MagFace from %s", path)
    return session


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a BGR face crop for MagFace inference.

    Face crops should be ArcFace-aligned (e.g., from extract_face_crops.py).
    Resizes to model input size (112×112) and normalizes to [0, 1].

    :param frame_bgr: BGR uint8 numpy array, ArcFace-aligned.
    :return: Float32 array of shape (1, 3, 112, 112), values in [0, 1].
    """
    # Resize to model input size
    resized = cv2.resize(
        frame_bgr, (_MAGFACE_INPUT_SIZE, _MAGFACE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    )

    # Normalize to [0, 1]
    resized_float = resized.astype(np.float32) / 255.0

    # Convert to NCHW blob format
    return cv2.dnn.blobFromImage(
        resized_float,
        scalefactor=1.0,
        size=(_MAGFACE_INPUT_SIZE, _MAGFACE_INPUT_SIZE),
        mean=0,
        swapRB=False,
    )


def score_frame(session: ort.InferenceSession, frame_bgr: np.ndarray) -> float:
    """Score a single face crop frame with MagFace and apply OFIQ sigmoid calibration.

    :param session: Loaded MagFace ONNX session.
    :param frame_bgr: BGR uint8 numpy array, ArcFace-aligned.
    :return: Calibrated quality score in [0, 100] (higher = better). Returns 0.0 on failure.
    """
    from persondet.postprocessing import apply_ofiq_sigmoid_calibration

    try:
        inp = preprocess(frame_bgr)
        outputs = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: inp}))
        raw_score = float(outputs[0][0])
        return apply_ofiq_sigmoid_calibration(raw_score)
    except Exception as e:
        logger.debug("Error scoring frame: %s", e)
        return 0.0
