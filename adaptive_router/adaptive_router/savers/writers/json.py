"""JSON profile writer implementation."""

import json
import logging
from pathlib import Path

from adaptive_router.savers.writers.base import ProfileWriter
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class JSONProfileWriter(ProfileWriter):
    """Writer for JSON format RouterProfiles."""

    def write_to_path(self, profile: RouterProfile, path: Path) -> None:
        """Write profile to JSON file path.

        Args:
            profile: RouterProfile to save
            path: Destination file path

        Raises:
            IOError: If write fails
        """
        logger.debug(f"Writing JSON profile to: {path}")

        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize profile to JSON
            profile_dict = profile.model_dump()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2, ensure_ascii=False)

            logger.debug(
                f"Successfully wrote JSON profile with {profile.metadata.n_clusters} clusters"
            )

        except Exception as e:
            error_msg = f"Failed to write JSON profile to {path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def write_to_bytes(self, profile: RouterProfile) -> bytes:
        """Write profile to JSON bytes (for MinIO/S3).

        Args:
            profile: RouterProfile to save

        Returns:
            Serialized profile as bytes
        """
        logger.debug("Writing JSON profile to bytes")

        try:
            # Serialize profile to JSON
            profile_dict = profile.model_dump()
            json_str = json.dumps(profile_dict, indent=2, ensure_ascii=False)
            data = json_str.encode("utf-8")

            logger.debug(
                f"Successfully wrote JSON profile with {profile.metadata.n_clusters} clusters"
            )
            return data

        except Exception as e:
            error_msg = f"Failed to serialize JSON profile: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".json"]
