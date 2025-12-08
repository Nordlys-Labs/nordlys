"""Profile loaders for adaptive router.

This module provides different strategies for loading router profiles:
- LocalFileProfileLoader: Load profiles from local JSON/YAML files
- MinIOProfileLoader: Load profiles from MinIO object storage
- ProfileLoader: Abstract base class for custom implementations
"""

from .base import ProfileLoader
from .local import LocalFileProfileLoader
from .minio import MinIOProfileLoader

__all__ = [
    "ProfileLoader",
    "LocalFileProfileLoader",
    "MinIOProfileLoader",
]
