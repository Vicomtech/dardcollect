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
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


def ensure_simplified_onnx(model_path: str | Path) -> str:
    """Return path to a simplified copy of the ONNX model, cached as <stem>.simplified.onnx.

    Runs ONNX shape inference then onnxsim constant-folding entirely in memory,
    writing only the final result. onnxsim folds shape-dependent If nodes
    (e.g. Gather(Shape(x), idx) == 1) into static branches TensorRT can parse.

    The .simplified.onnx file is hardware-agnostic and can be distributed alongside
    the original model. Falls back gracefully to the original path if either tool
    is unavailable or if the simplified file doesn't exist yet.
    """
    try:
        import onnx
    except ImportError:
        return str(model_path)

    model_path = Path(model_path)
    prepared_path = model_path.with_name(model_path.stem + ".simplified.onnx")

    if prepared_path.exists() and prepared_path.stat().st_mtime >= model_path.stat().st_mtime:
        return str(prepared_path)

    logger.debug("Preparing %s for TensorRT (shape inference + onnxsim)...", model_path.name)
    try:
        # Step 1: shape inference in memory
        model = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))

        # Step 2: onnxsim constant-folding (folds If nodes with fixed input shapes)
        try:
            from onnxsim import simplify

            input_shapes = {
                inp.name: [d.dim_value or 1 for d in inp.type.tensor_type.shape.dim]
                for inp in model.graph.input
            }
            simplified, ok = simplify(model, overwrite_input_shapes=input_shapes)
            if ok:
                model = simplified
            else:
                logger.debug("onnxsim failed for %s — shape-inferred only", model_path.name)
        except ImportError:
            logger.debug("onnxsim not available for %s — shape-inferred only", model_path.name)

        onnx.save(model, str(prepared_path))
        return str(prepared_path)

    except Exception as e:
        logger.debug("Model preparation failed for %s: %s — using original", model_path.name, e)
        return str(model_path)


def create_ort_session(
    model_path: str | Path,
    providers: list,
    gpu_id: int = 0,
) -> ort.InferenceSession:
    """Create an ORT InferenceSession with shape inference and TRT→CUDA fallback.

    Runs ensure_simplified_onnx() first so TensorRT can parse models with
    dynamic graph patterns. Falls back to CUDA/CPU if TRT fails to load.
    """
    prepared = ensure_simplified_onnx(model_path)
    if prepared != str(model_path):
        logger.info("Using simplified model: %s", prepared)
    using_trt = any("TensorrtExecutionProvider" in str(p) for p in providers)
    fallback = [
        ("CUDAExecutionProvider", {"device_id": gpu_id}),
        "CPUExecutionProvider",
    ]

    if using_trt:
        logger.info(
            "⚠️  TensorRT is enabled — processing may pause while compiling GPU engines on first use"
        )
        # TRT's C++ parser writes directly to fd 2 on failure; suppress it so
        # structural incompatibilities (e.g. If-node dynamic shapes in CIGPose)
        # don't flood the log. ORT exception is still caught and handled below.
        sys.stderr.flush()
        saved_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            sess = ort.InferenceSession(prepared, providers=providers)
        except Exception as e:
            sess = None
            trt_error = e
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)

        if sess is not None:
            return sess
        logger.warning("TRT unavailable for %s — falling back to CUDA/CPU", Path(model_path).name)
        logger.debug("TRT error: %s", trt_error)
        return ort.InferenceSession(prepared, providers=fallback)

    try:
        return ort.InferenceSession(prepared, providers=providers)
    except Exception as e:
        logger.warning("Failed to load %s: %s — falling back to CUDA/CPU", Path(model_path).name, e)
        return ort.InferenceSession(prepared, providers=fallback)


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
            logger.debug("TensorRT library (nvinfer) not loadable. Disabling TensorRT provider.")

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
