"""Local filesystem profile saver implementation."""

import logging
from pathlib import Path

from adaptive_router.savers.base import ProfileSaver
from adaptive_router.savers.writers import get_writer
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class LocalFileProfileSaver(ProfileSaver):
    """Saver for RouterProfiles to local filesystem.

    Auto-detects format from file extension (.json, .csv, .parquet)
    and uses appropriate writer from savers.writers module.
    """

    def save_profile(self, profile: RouterProfile, path: str) -> str:
        """Save profile to local filesystem.

        Args:
            profile: RouterProfile to save
            path: Local file path (relative or absolute)

        Returns:
            Path where profile was saved

        Raises:
            IOError: If write fails
            ValueError: If path format is invalid or unsupported format
        """
        output_path = Path(path)

        logger.info(f"Saving profile to local file: {output_path}")

        try:
            # Auto-detect format from file extension and get appropriate writer
            writer = get_writer(output_path)
            writer.write_to_path(profile, output_path)

            logger.info(
                f"Successfully saved profile to local file "
                f"(format: {output_path.suffix}, n_clusters: {profile.metadata.n_clusters})"
            )

            return str(output_path)

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            raise

    def health_check(self) -> bool:
        """Check if local filesystem is writable.

        Returns:
            True if filesystem is writable, False otherwise
        """
        try:
            # Try to create a temporary file in current directory
            test_file = Path(".adaptive_router_health_check.tmp")
            test_file.write_text("test")
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False
