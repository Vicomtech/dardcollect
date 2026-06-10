"""GPU library path setup for ONNX Runtime and TensorRT.

Configures environment variables and preloads CUDA, cuDNN, and TensorRT
libraries so that ONNX Runtime can find them at session creation time.

Two entry points:
    - `auto_preload_pypi_nvidia_libs()` — config-free, side-effect-safe.
      Searches site-packages for PyPI-installed NVIDIA libraries (nvidia-*-cu12,
      tensorrt, etc.) and preloads them. Called automatically from
      `dardcollect/__init__.py` so library users get GPU support without
      explicit setup. Idempotent.
    - `setup_gpu_paths(config_path)` — config-driven. Runs auto-preload, then
      on Windows additionally adds system CUDA / cuDNN / TensorRT directories
      from `config.yaml` to the DLL search path. Used by pipeline scripts.

Platform-specific behavior:
    - Linux: Preloads shared libraries with RTLD_GLOBAL so ONNX Runtime
      can resolve symbols from pip-installed NVIDIA packages.
    - Windows: Adds DLL directories via os.add_dll_directory() and preloads
      libraries with ctypes to resolve split-library dependencies.
"""

import ctypes
import glob
import importlib.util
import logging
import os
import site
import sys

import yaml

logger = logging.getLogger(__name__)


def _trigger_torch_cuda_preload() -> None:
    """Import torch to preload its bundled CUDA/cuDNN libs into the process.

    The torch wheel bundles `libcudart.so.12`, `libcublas.so.12`, `libcudnn.so.9`
    etc. in `site-packages/torch/lib/`. Torch's `__init__.py` ctypes-loads these
    when imported, making them visible to subsequent dlopens (e.g. ONNX
    Runtime's CUDA provider). No-op if torch is not installed.
    """
    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch  # noqa: F401  # triggers torch's CUDA lib preload

        logger.debug("Triggered torch CUDA lib preload")
    except Exception as e:
        logger.debug("Failed to import torch for CUDA preload: %s", e)


def _site_packages_dirs() -> list[str]:
    """Return all plausible site-packages directories for the current interpreter."""
    dirs = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        dirs.append(site.getusersitepackages())
    dirs.extend(glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages")))
    if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        dirs.extend(glob.glob(os.path.join(sys.base_prefix, "lib", "python*", "site-packages")))
    return dirs


def _preload_nvidia_lib_linux(package_name: str, lib_pattern: str) -> bool:
    """Find and preload NVIDIA libraries from pip site-packages on Linux.

    Searches site-packages/nvidia/{package_name}/lib and (for tensorrt)
    site-packages/tensorrt_libs / tensorrt_cu12_libs / tensorrt_cu13_libs,
    loading matching .so files with RTLD_GLOBAL.

    Args:
        package_name: NVIDIA package name (e.g., "cudnn", "tensorrt").
        lib_pattern: Glob pattern (e.g., "libcudnn.so*").

    Returns:
        True if at least one library was loaded.
    """
    found = False
    for sp in _site_packages_dirs():
        target_dirs = [os.path.join(sp, "nvidia", package_name, "lib")]
        if package_name == "tensorrt":
            target_dirs = [
                os.path.join(sp, "tensorrt_cu12_libs"),
                os.path.join(sp, "tensorrt_libs"),
                os.path.join(sp, "tensorrt_cu13_libs"),
            ]

        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                continue
            libs = glob.glob(os.path.join(target_dir, lib_pattern))
            if not libs:
                continue

            loaded_any = False
            failures: list[tuple[str, OSError]] = []
            for lib in sorted(libs):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    logger.debug("Preloaded %s from %s", os.path.basename(lib), target_dir)
                    found = True
                    loaded_any = True
                except OSError as e:
                    logger.debug("Failed to preload %s: %s", lib, e)
                    failures.append((lib, e))

            if not loaded_any and failures:
                logger.warning(
                    "Found %d %s lib(s) in %s but none loaded. First error: %s",
                    len(failures),
                    package_name,
                    target_dir,
                    failures[0][1],
                )

            if loaded_any:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                if target_dir not in current_ld.split(os.pathsep):
                    os.environ["LD_LIBRARY_PATH"] = f"{target_dir}{os.pathsep}{current_ld}"
                    logger.debug("Added %s to LD_LIBRARY_PATH", target_dir)
                break

    return found


def _preload_nvidia_lib_windows(package_name: str, dll_pattern: str) -> bool:
    """Find and preload NVIDIA DLLs from pip site-packages on Windows.

    Searches site-packages/nvidia/{package_name}/bin and (for tensorrt)
    site-packages/tensorrt_libs / tensorrt_cu12_libs, adding each directory
    via os.add_dll_directory and loading matching .dll files with ctypes.

    Args:
        package_name: NVIDIA package name (e.g., "cudnn", "tensorrt").
        dll_pattern: Glob pattern (e.g., "cudnn64_*.dll").

    Returns:
        True if at least one DLL was loaded.
    """
    found = False
    for sp in _site_packages_dirs():
        target_dirs = [os.path.join(sp, "nvidia", package_name, "bin")]
        if package_name == "tensorrt":
            target_dirs = [
                os.path.join(sp, "tensorrt_cu12_libs"),
                os.path.join(sp, "tensorrt_libs"),
                os.path.join(sp, "tensorrt_cu13_libs"),
            ]

        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                continue
            dlls = glob.glob(os.path.join(target_dir, dll_pattern))
            if not dlls:
                continue

            try:
                os.add_dll_directory(target_dir)
            except OSError as e:
                logger.debug("Failed to add DLL directory %s: %s", target_dir, e)

            loaded_any = False
            failures: list[tuple[str, OSError]] = []
            for dll in sorted(dlls):
                try:
                    ctypes.CDLL(dll)
                    logger.debug("Preloaded %s from %s", os.path.basename(dll), target_dir)
                    found = True
                    loaded_any = True
                except OSError as e:
                    logger.debug("Failed to preload %s: %s", dll, e)
                    failures.append((dll, e))

            if not loaded_any and failures:
                logger.warning(
                    "Found %d %s DLL(s) in %s but none loaded. First error: %s",
                    len(failures),
                    package_name,
                    target_dir,
                    failures[0][1],
                )

            if loaded_any:
                break

    return found


def auto_preload_pypi_nvidia_libs() -> None:
    """Preload PyPI-installed NVIDIA shared libraries.

    Config-free, idempotent, safe to call multiple times. Called automatically
    from `dardcollect/__init__.py` so library users get GPU lib resolution
    without needing to call `setup_gpu_paths()` explicitly.

    Strategy (order matters):
        1. Import torch first. Torch bundles its own CUDA + cuDNN libs in
           site-packages/torch/lib/ and preloads them via ctypes on import.
           Going first ensures the dynamic linker dedups by SONAME against
           torch's specific bundled versions (the ones torch was built
           against), avoiding ABI mismatches if standalone nvidia-*-cu12
           wheels declare different minor versions.
        2. Search site-packages for standalone NVIDIA wheels (nvidia-*-cu12,
           tensorrt) and preload anything found. With torch's libs already
           loaded, matching-SONAME libs become no-ops; only TensorRT (which
           torch does not bundle) actually loads.

    On systems without PyPI NVIDIA wheels and without torch, this is a no-op.
    """
    # 1. Torch first — its bundled CUDA libs anchor the SONAME resolution.
    _trigger_torch_cuda_preload()

    # 2. Standalone NVIDIA / TensorRT wheels (primarily for TensorRT, which
    #    torch does not bundle; CUDA libs here will dedup against torch's).
    if sys.platform == "linux":
        try:
            # cuBLAS
            _preload_nvidia_lib_linux("cublas", "libcublas.so*")
            _preload_nvidia_lib_linux("cublas", "libcublasLt.so*")
            # cuDNN
            _preload_nvidia_lib_linux("cudnn", "libcudnn.so*")
            _preload_nvidia_lib_linux("cudnn", "libcudnn_*.so*")
            # CUDA runtime + supporting libs
            _preload_nvidia_lib_linux("cuda_runtime", "libcudart.so*")
            _preload_nvidia_lib_linux("cufft", "libcufft.so*")
            _preload_nvidia_lib_linux("curand", "libcurand.so*")
            # TensorRT
            _preload_nvidia_lib_linux("tensorrt", "libnvinfer.so*")
            _preload_nvidia_lib_linux("tensorrt", "libnvinfer_plugin.so*")
            _preload_nvidia_lib_linux("tensorrt", "libnvinfer_builder_resource.so*")
            _preload_nvidia_lib_linux("tensorrt", "libnvonnxparser.so*")
        except Exception as e:
            logger.warning("Error preloading NVIDIA libs on Linux: %s", e)

    elif sys.platform == "win32":
        try:
            # cuBLAS
            _preload_nvidia_lib_windows("cublas", "cublas64_*.dll")
            _preload_nvidia_lib_windows("cublas", "cublasLt64_*.dll")
            # cuDNN
            _preload_nvidia_lib_windows("cudnn", "cudnn64_*.dll")
            _preload_nvidia_lib_windows("cudnn", "cudnn_*.dll")
            # CUDA runtime + supporting
            _preload_nvidia_lib_windows("cuda_runtime", "cudart64_*.dll")
            _preload_nvidia_lib_windows("cufft", "cufft64_*.dll")
            _preload_nvidia_lib_windows("curand", "curand64_*.dll")
            # TensorRT
            _preload_nvidia_lib_windows("tensorrt", "nvinfer*.dll")
            _preload_nvidia_lib_windows("tensorrt", "nvonnxparser*.dll")
        except Exception as e:
            logger.warning("Error preloading NVIDIA libs on Windows: %s", e)


def setup_gpu_paths(config_path: str = "config.yaml") -> None:
    """Configure environment variables and DLL paths for NVIDIA GPU support.

    Always preloads PyPI NVIDIA libraries via `auto_preload_pypi_nvidia_libs()`.
    On Windows, additionally reads `gpu_paths` from the given YAML config and
    adds system CUDA / cuDNN / TensorRT directories to the DLL search path.

    Args:
        config_path: Path to YAML config containing `gpu_paths`. Only used on
            Windows; ignored on Linux. Default: "config.yaml" in the cwd.

    Note:
        Safe to call multiple times; duplicate path additions are handled.
    """
    auto_preload_pypi_nvidia_libs()

    if sys.platform != "win32":
        return

    # Windows: optionally augment with system CUDA / TensorRT paths from config.yaml
    if not os.path.exists(config_path):
        logger.debug("Config file %s not found. Skipping system GPU path setup.", config_path)
        return

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        gpu_paths = config.get("gpu_paths", {}) or config.get("person_extraction", {}).get(
            "gpu_paths", {}
        )
        if not gpu_paths:
            logger.debug("No 'gpu_paths' found in config.")
            return

        cuda_bin = gpu_paths.get("cuda_bin") or (
            os.path.join(os.environ["CUDA_PATH"], "bin") if os.environ.get("CUDA_PATH") else None
        )
        trt_lib = gpu_paths.get("tensorrt_lib") or os.environ.get("TENSORRT_LIB_PATH")
        cudnn_bin = gpu_paths.get("cudnn_bin")

        paths_to_add: list[str] = []

        # Torch lib dir (priority — Torch ships its own CUDA, avoid DLL conflicts)
        torch_lib: str | None = None
        try:
            import importlib.util

            torch_spec = importlib.util.find_spec("torch")
            if torch_spec and torch_spec.submodule_search_locations:
                candidate = os.path.join(torch_spec.submodule_search_locations[0], "lib")
                if os.path.exists(candidate):
                    torch_lib = candidate
                    logger.debug("Found Torch lib: %s", torch_lib)
        except Exception as e:
            logger.debug("Failed to locate Torch lib dir: %s", e)

        if torch_lib:
            paths_to_add.append(torch_lib)
            use_system_libs = False
            logger.debug("Torch found. Skipping system CUDA/cuDNN to prevent DLL conflicts.")
        else:
            use_system_libs = True

        if use_system_libs and cuda_bin:
            paths_to_add.append(cuda_bin)
        if use_system_libs and cudnn_bin:
            paths_to_add.append(cudnn_bin)
        if trt_lib:
            paths_to_add.append(trt_lib)

        for p in paths_to_add:
            if p and os.path.exists(p):
                try:
                    os.add_dll_directory(p)
                except OSError as e:
                    logger.warning("Failed to add DLL directory %s: %s", p, e)

        # Preload system TRT DLLs (PyPI ones already loaded by auto_preload above)
        if trt_lib and os.path.exists(trt_lib):
            logger.debug("Preloading TensorRT DLLs from %s", trt_lib)
            for dll_path in glob.glob(os.path.join(trt_lib, "*.dll")):
                try:
                    ctypes.CDLL(dll_path)
                except OSError as e:
                    logger.debug("Failed to preload %s: %s", dll_path, e)

    except Exception as e:
        logger.warning("Error setting up system GPU paths on Windows: %s", e)
