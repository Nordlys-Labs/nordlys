"""Abstract base class for profile savers.

This module defines the ProfileSaver ABC that all profile format savers must implement.
It provides a consistent interface for saving RouterProfiles to various storage backends.
"""

from abc import ABC, abstractmethod

from adaptive_router.models.storage import RouterProfile


class ProfileSaver(ABC):
    """Abstract base class for profile savers.

    Supports saving RouterProfiles to various storage backends.
    Implementations should handle both local filesystem and remote storage.
    """

    @abstractmethod
    def save_profile(self, profile: RouterProfile, path: str) -> str:
        """Save profile to specified path.

        Args:
            profile: RouterProfile to save
            path: Destination path (local or remote URL)

        Returns:
            Path where profile was saved

        Raises:
            IOError: If save fails
            ValueError: If path format is invalid
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Check if saver is healthy and can save profiles.

        Returns:
            True if healthy, False otherwise
        """
        return True
