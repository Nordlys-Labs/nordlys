"""Device utilities for execution backends."""

from __future__ import annotations

import importlib.util
from typing import Literal

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


def check_cuda_available() -> bool:
    """Check if CUDA is available for clustering.

    Returns:
        True if CUDA dependencies are available, False otherwise
    """
    return importlib.util.find_spec("cuml") is not None


def require_cuda() -> None:
    """Require CUDA to be available.

    Raises:
        ImportError: If CUDA clustering is not available
    """
    if not check_cuda_available():
        msg = (
            "CUDA clustering requires 'cuml' package. "
            "Install with: pip install cuml-cu12"
        )
        raise ImportError(msg)
