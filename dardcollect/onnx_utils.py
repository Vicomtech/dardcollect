import logging
import os
import sys

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_preferred_providers(device_id: int = 0) -> list[str]:
    """Build the ONNX Runtime provider list in priority order: TensorRT → CUDA → CPU.

    TensorRT is only added when:
    - ORT reports it as available, AND
    - GPU compute capability >= 7.5 (TensorRT 10 requirement; V100 = 7.0 is excluded), AND
    - At least one nvinfer DLL can be loaded (Windows) or is assumed present (Linux).

    Engine cache is written to .cache/trt_engines/ in the current directory.
    """
    available_providers = ort.get_available_providers()

    # TensorrtExecutionProvider must precede CUDAExecutionProvider in the list.
    preferred = []

    if "TensorrtExecutionProvider" in available_providers:
        can_use_trt = True

        # TensorRT 10 dropped support for compute capability < 7.5 (V100 = 7.0)
        try:
            import torch

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(device_id)
                if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
                    logger.warning(
                        f"Skipping TensorRT: GPU compute capability {cap} < 7.5 "
                        f"(not supported by installed TRT version)."
                    )
                    can_use_trt = False
        except ImportError:
            pass

        # On Windows, verify nvinfer DLL is loadable before handing control to ORT.
        # ORT crashes loudly if it tries to initialize TRT without the DLL present.
        if can_use_trt and sys.platform == "win32":
            try:
                import ctypes

                candidates = ["nvinfer.dll", "nvinfer_10.dll", "nvinfer_8.dll"]
                found_trt = False
                failure_logs = []

                for cand in candidates:
                    try:
                        ctypes.cdll.LoadLibrary(cand)
                        found_trt = True
                        logger.debug(f"Successfully loaded TensorRT lib: {cand}")
                        break
                    except OSError as e:
                        # Also accept if the DLL was already loaded by gpu_setup.py
                        try:
                            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                            h_module = kernel32.GetModuleHandleW(cand)
                            if h_module:
                                found_trt = True
                                logger.debug(
                                    "Confirmed TensorRT lib %s is already loaded in memory.",
                                    cand,
                                )
                                break
                        except Exception:
                            pass
                        failure_logs.append(f"Failed to load {cand}: {e}")

                if not found_trt:
                    for fail_msg in failure_logs:
                        logger.debug(fail_msg)
                    can_use_trt = False
                    logger.debug(
                        "TensorRT libraries (nvinfer) not found/loadable. "
                        "Disabling TensorRT provider."
                    )

            except Exception as e:
                logger.debug(f"Error checking TensorRT availability: {e}")
                can_use_trt = False

        if can_use_trt:
            cache_dir = os.path.join(os.getcwd(), ".cache", "trt_engines")
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass

            options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
                "trt_fp16_enable": True,
                "device_id": device_id,
            }
            preferred.append(("TensorrtExecutionProvider", options))

    # 2. CUDA
    if "CUDAExecutionProvider" in available_providers:
        preferred.append(("CUDAExecutionProvider", {"device_id": device_id}))

    # 3. CPU
    preferred.append("CPUExecutionProvider")

    return preferred


def check_cuda_dependencies() -> None:
    """Placeholder — CUDA dependency checks are handled inside get_preferred_providers."""
    pass
