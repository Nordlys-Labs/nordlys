"""JSON profile reader implementation."""

import json
import logging
from pathlib import Path

from adaptive_router.loaders.readers.base import ProfileReader
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class JSONProfileReader(ProfileReader):
    """Reader for JSON format RouterProfiles."""

    def read_from_path(self, path: Path) -> RouterProfile:
        """Read profile from JSON file path.

        Args:
            path: Path to JSON profile file

        Returns:
            Loaded RouterProfile

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or profile validation fails
        """
        logger.debug(f"Reading JSON profile from: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                profile_dict = json.load(f)

            profile = RouterProfile(**profile_dict)
            logger.debug(
                f"Successfully read JSON profile with {profile.metadata.n_clusters} clusters"
            )
            return profile

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in profile file {path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed for {path}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            raise

    def read_from_bytes(self, data: bytes) -> RouterProfile:
        """Read profile from JSON bytes (for MinIO/S3).

        Args:
            data: Raw JSON bytes

        Returns:
            Loaded RouterProfile

        Raises:
            ValueError: If JSON is invalid or profile validation fails
        """
        logger.debug(f"Reading JSON profile from bytes ({len(data)} bytes)")

        try:
            profile_dict = json.loads(data.decode("utf-8"))
            profile = RouterProfile(**profile_dict)
            logger.debug(
                f"Successfully read JSON profile with {profile.metadata.n_clusters} clusters"
            )
            return profile

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_msg = f"Invalid JSON data: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            raise

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".json"]
