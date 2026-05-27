"""ONNX Runtime provider selection.

Builds the ONNX Runtime execution-provider list in priority order
(TensorRT → CUDA → CPU). NVIDIA library preload is handled separately by
`dardcollect.gpu_setup.auto_preload_pypi_nvidia_libs`, which runs from
`dardcollect/__init__.py`.

TensorRT is enabled only when:
- ORT reports TensorrtExecutionProvider as available
- GPU compute capability is >= 7.5 (required by TensorRT 10)
- nvinfer can be loaded by ctypes (verified per-platform)
"""

import ctypes
import logging
import os
import sys

import onnxruntime as ort

logger = logging.getLogger(__name__)


def _trt_loadable() -> bool:
    """Verify the TensorRT shared library is loadable by name.

    `auto_preload_pypi_nvidia_libs` should have already loaded it with
    RTLD_GLOBAL (Linux) or via add_dll_directory (Windows). This call confirms
    it. If preload silently failed, ORT would later crash at
    `InferenceSession(..., providers=[TRT])` — this check disables TRT in the
    provider list before that happens.
    """
    if sys.platform == "linux":
        for cand in ("libnvinfer.so.10", "libnvinfer.so"):
            try:
                ctypes.CDLL(cand)
                return True
            except OSError:
                continue
        return False

    if sys.platform == "win32":
        for cand in ("nvinfer.dll", "nvinfer_10.dll", "nvinfer_8.dll"):
            try:
                ctypes.cdll.LoadLibrary(cand)
                return True
            except OSError:
                # Also accept if already loaded (gpu_setup may have preloaded it)
                try:
                    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                    if kernel32.GetModuleHandleW(cand):
                        return True
                except Exception:
                    pass
        return False

    return False


def get_preferred_providers(device_id: int = 0) -> list:
    """Build the ONNX Runtime execution provider list in priority order.

    Provider priority: TensorRT → CUDA → CPU. TensorRT is added only when ORT
    reports it as available, the GPU compute capability is >= 7.5, and the
    nvinfer library is loadable.

    When TensorRT is enabled, engine caches are written to .cache/trt_engines/
    in the current directory.

    Args:
        device_id: CUDA device ID to use for GPU providers.

    Returns:
        Ordered list of providers (each a string or (name, options) tuple).
    """
    available = ort.get_available_providers()
    preferred: list = []

    if "TensorrtExecutionProvider" in available:
        can_use_trt = True

        # TensorRT 10 dropped support for compute capability < 7.5 (V100 = 7.0)
        try:
            import torch

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(device_id)
                if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
                    logger.warning(
                        "Skipping TensorRT: GPU compute capability %s < 7.5 "
                        "(not supported by installed TRT version).",
                        cap,
                    )
                    can_use_trt = False
        except ImportError:
            pass

        if can_use_trt and not _trt_loadable():
            can_use_trt = False
            logger.debug(
                "TensorRT library (nvinfer) not loadable. Disabling TensorRT provider."
            )

        if can_use_trt:
            cache_dir = os.path.join(os.getcwd(), ".cache", "trt_engines")
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass

            preferred.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache_dir,
                        "trt_fp16_enable": True,
                        "device_id": device_id,
                    },
                )
            )

    if "CUDAExecutionProvider" in available:
        preferred.append(("CUDAExecutionProvider", {"device_id": device_id}))

    preferred.append("CPUExecutionProvider")

    return preferred
