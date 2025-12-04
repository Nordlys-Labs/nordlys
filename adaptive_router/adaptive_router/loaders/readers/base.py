"""Abstract base class for profile readers.

This module defines the ProfileReader ABC that all profile format readers must implement.
It provides a consistent interface for reading RouterProfiles from various formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from adaptive_router.models.storage import RouterProfile


class ProfileReader(ABC):
    """Abstract base class for profile readers.

    Supports reading RouterProfiles from various formats.
    """

    @abstractmethod
    def read_from_path(self, path: Path) -> RouterProfile:
        """Read profile from file path.

        Args:
            path: Source file path

        Returns:
            Loaded RouterProfile

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is invalid or profile validation fails
        """
        raise NotImplementedError

    @abstractmethod
    def read_from_bytes(self, data: bytes) -> RouterProfile:
        """Read profile from bytes (for MinIO/S3).

        Args:
            data: Raw bytes data

        Returns:
            Loaded RouterProfile

        Raises:
            ValueError: If format is invalid or profile validation fails
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
