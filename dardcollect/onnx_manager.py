"""
ONNX Runtime inference manager with environment-variable provider selection.

Provides a high-level wrapper around ONNX Runtime InferenceSession that handles:
- Automatic provider selection (CUDA vs CPU) based on environment variables
- GPU device ID configuration
- Thread pool configuration
- Model loading with error handling and logging

Environment variables:
    DETECTOR_USE_GPU: Set to "1", "true", or "yes" to enable GPU inference.
    DETECTOR_GPU_ID: Integer GPU device ID (default: 0).
    DETECTOR_NUM_THREADS: Number of intra-op threads (default: 0, auto).
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
    """Container for ONNX Runtime session and I/O metadata.

    Attributes:
        inputs: Sequence of input node descriptors from the model.
        outputs: List of output node names.
        output_size: Last dimension of the first output tensor (embedding/feature size).
        sess: The ONNX Runtime inference session.
    """

    inputs: Any = None  # Sequence of input node descriptors
    outputs: list[str] | None = None
    output_size: int = 0
    sess: Optional["ort.InferenceSession"] = None


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean value from an environment variable.

    Parses common string representations: "1"/"true"/"yes" → True,
    "0"/"false"/"no" → False. Falls back to *default* for any other value.

    Args:
        name: Name of the environment variable to read.
        default: Value to return if the variable is not set or unrecognized.

    Returns:
        bool: Parsed boolean value.
    """
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable.

    Args:
        name: Name of the environment variable to read.
        default: Value to return if the variable is not set or not a valid integer.

    Returns:
        int: Parsed integer value or *default* on failure.
    """
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class ONNXManager:
    """High-level wrapper for an ONNX Runtime inference session.

    Handles model loading, provider selection, session options, and inference.
    Provider priority: CUDA (when DETECTOR_USE_GPU=1) → CPU.

    Note:
        For TensorRT support, use `dardcollect.onnx_utils.get_preferred_providers`
        which includes TensorRT in the provider chain.
    """

    def __init__(self, model_file: str) -> None:
        """Initialize and load an ONNX model.

        Args:
            model_file: Path to the ONNX model file.

        Raises:
            ImportError: If onnxruntime is not installed.
            RuntimeError: If the model fails to load.
        """
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
        """Return the feature vector dimension of the model's primary output.

        Returns:
            int: Last dimension of the first output tensor (0 if unknown).
        """
        return self._instance.output_size

    def do_inference(self, input_blob: np.ndarray) -> Any:
        """Run inference on the given input tensor.

        Args:
            input_blob: Input tensor with shape matching the model's first input.

        Returns:
            Model output, typically a list of numpy arrays.

        Raises:
            RuntimeError: If the ONNX session or inputs are not initialized.
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
    """Create an ONNXManager for the given model file if it is an ONNX file.

    Args:
        model_file: Path to the model file.

    Returns:
        ONNXManager instance if the file has a .onnx extension, otherwise None.
    """
    if model_file.lower().endswith(".onnx"):
        return ONNXManager(model_file)
    return None
