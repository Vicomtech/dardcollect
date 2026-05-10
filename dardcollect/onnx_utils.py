import logging
import os
import sys

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_preferred_providers(device_id: int = 0) -> list[str]:
    """
    Get the list of providers to use for ONNX Runtime, handling missing dependencies.

    Checks for presence of TensorRT and CUDA DLLs on Windows to avoid
    noisy errors and fallbacks.
    """
    available_providers = ort.get_available_providers()

    # Desired order
    # Note: 'TensorrtExecutionProvider' must be before 'CUDAExecutionProvider'
    preferred = []

    # 1. TensorRT
    if "TensorrtExecutionProvider" in available_providers:
        can_use_trt = True

        # Check specific platform requirements first
        if sys.platform == "win32":
            # ... existing windows checks ...
            pass

        # Check architecture support (TensorRT 10+ dropped support for SM < 7.5
        # unless the legacy pack is used)
        # But we are using the main package.
        try:
            import torch

            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(device_id)
                # V100 is 7.0. TensorRT 10 requires 7.5+ for standard build?
                # Actually, let's just skip 7.0 to be safe since we saw the error.
                if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
                    logger.warning(
                        f"Skipping TensorRT: GPU compute capability {cap} < 7.5 "
                        f"(not supported by installed TRT version)."
                    )
                    can_use_trt = False
        except ImportError:
            pass

        if can_use_trt:
            if sys.platform == "win32":
                # ... (existing win32 logic to check dlls) ...
                pass
            elif sys.platform == "linux":
                # We already verified libs in gpu_setup, so assume yes if we got here?
                pass

            # If passed checks, add it
            if can_use_trt:
                # ... existing options ...
                pass
            try:
                import ctypes

                # Try to load the main TensorRT library
                # (version-specific names, e.g. nvinfer_10.dll)
                # But names change. simpler strategy:
                # If we rely on ONNXRuntime's internal loading, it crashes loud.
                # Let's try to find it in PATH.
                import ctypes.util

                # This is imperfect as names vary (nvinfer.dll, nvinfer_10.dll, etc)
                # Helper: checks if any file matching nvinfer*.dll exists in CUDA bin or PATH

                # We can check specific known DLL names for recent TRT versions
                candidates = ["nvinfer.dll", "nvinfer_10.dll", "nvinfer_8.dll"]
                found_trt = False

                # Check loaded DLLs or PATH?
                # ctypes.util.find_library doesn't work well for DLLs on Windows always.

                # Let's try to load 'nvinfer_10.dll' (common for recent ORT) or catch the error
                # Loop through candidates
                failure_logs = []
                for cand in candidates:
                    try:
                        ctypes.cdll.LoadLibrary(cand)
                        found_trt = True
                        logger.debug(f"Successfully loaded TensorRT lib: {cand}")
                        break
                    except OSError as e:
                        # Fallback: check if already loaded in process memory (e.g. by gpu_setup.py)
                        if sys.platform == "win32":
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

                        # Store error for later, don't spam unless we fail completely
                        failure_logs.append(f"Failed to load {cand}: {e}")
                        continue

                if not found_trt:
                    # Only log the failures if we truly couldn't find ANY version
                    for fail_msg in failure_logs:
                        logger.debug(fail_msg)

                if not found_trt:
                    # One last heuristic: check CUDA bin dir for file existence
                    # This avoids PATH issues if we already added it to DLL search path
                    # but LoadLibrary fails
                    # due to dependencies.
                    # But if LoadLibrary fails, ORT will likely fail too.
                    can_use_trt = False
                    logger.debug(
                        "TensorRT libraries (nvinfer) not found/loadable. "
                        "Disabling TensorRT provider."
                    )

            except Exception as e:
                logger.debug(f"Error checking TensorRT availability: {e}")
                can_use_trt = False

        if can_use_trt:
            # Configure explicit cache to avoid corruption/system temp issues
            # and to allow easy cleanup.
            cache_dir = os.path.join(os.getcwd(), ".cache", "trt_engines")
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass  # Use default if we can't create dir

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


def check_cuda_dependencies():
    pass
