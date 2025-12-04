"""Abstract base class for profile writers.

This module defines the ProfileWriter ABC that all profile format writers must implement.
It provides a consistent interface for writing RouterProfiles to various formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_router.models.storage import RouterProfile


class ProfileWriter(ABC):
    """Abstract base class for profile writers.

    Supports writing RouterProfiles to various formats.
    """

    @abstractmethod
    def write_to_path(self, profile: RouterProfile, path: Path) -> None:
        """Write profile to file path.

        Args:
            profile: RouterProfile to save
            path: Destination file path

        Raises:
            IOError: If write fails
        """
        raise NotImplementedError

    @abstractmethod
    def write_to_bytes(self, profile: RouterProfile) -> bytes:
        """Write profile to bytes (for MinIO/S3).

        Args:
            profile: RouterProfile to save

        Returns:
            Serialized profile as bytes
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> list[str]:
        """Return list of supported file extensions.

        Returns:
            List of extensions (e.g., ['.json'])
        """
        raise NotImplementedError
