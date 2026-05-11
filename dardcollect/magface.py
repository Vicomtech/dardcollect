"""MagFace face quality assessment using ONNX Runtime.

MagFace (IResNet50) is trained on MS1MV2 with MagFace loss and measures how
confidently a face recognition model can embed a given crop. This serves as a
standard proxy for biometric sample quality following ISO/IEC 29794-5 (OFIQ).

The raw model output is calibrated via an OFIQ sigmoid to produce a score in
[0, 100], where higher values indicate better face image quality.

Reference: https://github.com/IrvingMeng/MagFace
"""

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort

from .onnx_utils import get_preferred_providers

logger = logging.getLogger(__name__)

_MAGFACE_INPUT_SIZE = 112  # IResNet50 input resolution


def load_magface(gpu_id: int) -> ort.InferenceSession:
    """Load the MagFace ONNX model from dardcollect/models/.

    Args:
        gpu_id: CUDA device ID for inference. Falls back to CPU if CUDA is unavailable.

    Returns:
        ort.InferenceSession: Loaded MagFace inference session.

    Raises:
        FileNotFoundError: If the model file does not exist at the expected path.
    """
    path = Path(__file__).parent / "models" / "magface_iresnet50_norm.onnx"

    if not path.exists():
        raise FileNotFoundError(
            f"MagFace model not found at {path}.\n"
            f"Expected at dardcollect/models/magface_iresnet50_norm.onnx.\n"
            f"Obtain it from the OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project"
        )

    providers = get_preferred_providers(device_id=gpu_id)

    using_trt = any("TensorrtExecutionProvider" in str(p) for p in providers)
    if using_trt:
        msg = (
            "⚠️  TensorRT is enabled — processing may pause while compiling GPU engines on first use"
        )
        logger.info(msg)

    logger.info("Loading MagFace from %s", path)
    return ort.InferenceSession(str(path), providers=providers)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a BGR face crop for MagFace inference.

    Resizes to 112×112, normalizes to [0, 1], and converts to NCHW blob format.
    Expects an ArcFace-aligned crop but will resize any input to the target size.

    Args:
        frame_bgr: BGR uint8 numpy array of the face crop.

    Returns:
        np.ndarray: Float32 NCHW blob of shape (1, 3, 112, 112), values in [0, 1].
    """
    resized = cv2.resize(
        frame_bgr, (_MAGFACE_INPUT_SIZE, _MAGFACE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    )
    resized_float = resized.astype(np.float32) / 255.0

    return cv2.dnn.blobFromImage(
        resized_float,
        scalefactor=1.0,
        size=(_MAGFACE_INPUT_SIZE, _MAGFACE_INPUT_SIZE),
        mean=0,
        swapRB=False,
    )


def score_frame(session: ort.InferenceSession, frame_bgr: np.ndarray) -> float:
    """Score a single face crop with MagFace quality assessment.

    Runs the MagFace model on the preprocessed crop and applies OFIQ sigmoid
    calibration to map the raw output to a [0, 100] quality score.

    Args:
        session: Loaded MagFace ONNX inference session.
        frame_bgr: BGR uint8 numpy array of the face crop (ArcFace-aligned).

    Returns:
        float: Calibrated quality score in [0, 100] (higher = better).
            Returns 0.0 on any inference or preprocessing failure.
    """
    from dardcollect.postprocessing import apply_ofiq_sigmoid_calibration

    try:
        inp = preprocess(frame_bgr)
        outputs = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: inp}))
        raw_score = float(outputs[0][0])
        return apply_ofiq_sigmoid_calibration(raw_score)
    except Exception as e:
        logger.debug("Error scoring frame: %s", e)
        return 0.0
