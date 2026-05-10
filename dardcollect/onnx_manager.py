"""
ONNX Runtime inference manager with environment-variable provider selection.

GPU selection is controlled by DETECTOR_USE_GPU (bool), DETECTOR_GPU_ID (int),
and DETECTOR_NUM_THREADS (int) environment variables.
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort
else:
    try:
        import onnxruntime as ort

        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False
        ort = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class ONNXInstance:
    """Holds the ONNX Runtime session and IO metadata."""

    inputs: Any = None  # Sequence of input node descriptors
    outputs: list[str] | None = None
    output_size: int = 0
    sess: Optional["ort.InferenceSession"] = None


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from environment variable.

    Maps "1"/"true"/"yes"→True, "0"/"false"/"no"→False.
    """
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variable.

    Returns *default* on missing/invalid value.
    """
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class ONNXManager:
    """ORT session wrapper used by the pose estimator and older pipeline components.

    Provider order: CUDA (when DETECTOR_USE_GPU=1) → CPU.
    Use dardcollect.onnx_utils.get_preferred_providers for TensorRT support.
    """

    def __init__(self, model_file: str) -> None:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self._logger = logging.getLogger(__name__)
        self._model_file = model_file
        self._instance = ONNXInstance()

        # Provider selection
        use_gpu = _env_bool("DETECTOR_USE_GPU", False)
        providers = []

        if use_gpu:
            # Try CUDA first
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                device_id = _env_int("DETECTOR_GPU_ID", 0)
                providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
                self._logger.info("Using CUDA GPU (device %d)", device_id)

        providers.append("CPUExecutionProvider")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Intra-op threads
        num_threads = _env_int("DETECTOR_NUM_THREADS", 0)
        if num_threads > 0:
            sess_options.intra_op_num_threads = num_threads

        self._logger.info("Loading ONNX model: %s", model_file)

        try:
            self._instance.sess = ort.InferenceSession(
                model_file,
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            self._logger.error("Failed to load model: %s", e)
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        self._instance.inputs = self._instance.sess.get_inputs()
        self._instance.outputs = [o.name for o in self._instance.sess.get_outputs()]

        # Last dimension of the first output = embedding/feature size
        out_shapes = [o.shape for o in self._instance.sess.get_outputs()]
        if out_shapes and len(out_shapes[0]) >= 2:
            self._instance.output_size = out_shapes[0][-1]

        self._logger.info("ONNX model loaded successfully")

    def get_vector_size(self) -> int:
        """Return the main output feature vector size (last output dimension)."""
        return self._instance.output_size

    def do_inference(self, input_blob: np.ndarray) -> Any:
        """Run inference on the given input tensor.

        :param input_blob: Input tensor with shape matching model input.
        :return: Output from the model (typically list of numpy arrays).
        """
        if self._instance.sess is None:
            raise RuntimeError("ONNX session not initialized")

        if self._instance.inputs is None:
            raise RuntimeError("ONNX inputs not initialized")

        input_name = self._instance.inputs[0].name
        return self._instance.sess.run(
            self._instance.outputs,
            {input_name: input_blob},
        )


def get_inference_engine(model_file: str) -> ONNXManager | None:
    """Return an ONNXManager for *model_file* if it is an ONNX file, else None."""
    if model_file.lower().endswith(".onnx"):
        return ONNXManager(model_file)
    return None
