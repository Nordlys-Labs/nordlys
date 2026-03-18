"""Device utilities for execution backends."""

from __future__ import annotations

import importlib.util
import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda"]


def get_device(device: DeviceType) -> DeviceType:
    """Validate and return the device.

    Args:
        device: Requested device type

    Returns:
        The validated device type

    Raises:
        ValueError: If device is not supported
    """
    if device not in ("cpu", "cuda"):
        msg = f"Unsupported device: {device}. Must be 'cpu' or 'cuda'."
        raise ValueError(msg)
    return device


def has_cuml_package() -> bool:
    """Check if the cuML package is installed.

    Returns:
        True if cuML package is available, False otherwise
    """
    return importlib.util.find_spec("cuml") is not None


def has_cupy_package() -> bool:
    """Check if the cupy package is installed.

    Returns:
        True if cupy package is available, False otherwise
    """
    return importlib.util.find_spec("cupy") is not None


def check_cuda_available() -> bool:
    """Check if CUDA is available for clustering (package + runtime).

    Returns:
        True if CUDA is available, False otherwise
    """
    if not has_cupy_package():
        return False
    try:
        import cupy

        return cupy.cuda.is_available()
    except Exception:
        return False


def require_cuda() -> None:
    """Require CUDA to be available.

    Raises:
        ImportError: If CUDA clustering is not available
    """
    if not has_cupy_package():
        logger.error("CUDA package not found", extra={"package": "cupy"})
        msg = (
            "CUDA clustering requires 'cupy' package. "
            "Install with: uv sync --extra cu13"
        )
        raise ImportError(msg)
    try:
        import cupy

        if not cupy.cuda.is_available():
            logger.error("CUDA runtime not available")
            msg = (
                "CUDA runtime not available. "
                "Ensure NVIDIA GPU is present and drivers are installed."
            )
            raise RuntimeError(msg)
    except ImportError:
        logger.error("CUDA package import failed")
        msg = (
            "CUDA clustering requires 'cupy' package. "
            "Install with: uv sync --extra cu13"
        )
        raise ImportError(msg)
