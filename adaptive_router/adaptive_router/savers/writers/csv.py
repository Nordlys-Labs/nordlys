"""CSV profile writer implementation."""

import csv
import io
import json
import logging
from pathlib import Path

from adaptive_router.savers.writers.base import ProfileWriter
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class CSVProfileWriter(ProfileWriter):
    """Writer for CSV format RouterProfiles.

    CSV format stores the profile as JSON data in a single row.
    Output format: CSV with columns 'format' and 'data', where
    'format' is 'json' and 'data' contains the JSON-serialized profile.
    """

    def write_to_path(self, profile: RouterProfile, path: Path) -> None:
        """Write profile to CSV file path.

        Args:
            profile: RouterProfile to save
            path: Destination file path

        Raises:
            IOError: If write fails
        """
        logger.debug(f"Writing CSV profile to: {path}")

        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize profile to JSON
            profile_dict = profile.model_dump()
            json_data = json.dumps(profile_dict, ensure_ascii=False)

            # Write as CSV with single row
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["format", "data"])
                writer.writeheader()
                writer.writerow({"format": "json", "data": json_data})

            logger.debug(
                f"Successfully wrote CSV profile with {profile.metadata.n_clusters} clusters"
            )

        except Exception as e:
            error_msg = f"Failed to write CSV profile to {path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def write_to_bytes(self, profile: RouterProfile) -> bytes:
        """Write profile to CSV bytes (for MinIO/S3).

        Args:
            profile: RouterProfile to save

        Returns:
            Serialized profile as bytes
        """
        logger.debug("Writing CSV profile to bytes")

        try:
            # Serialize profile to JSON
            profile_dict = profile.model_dump()
            json_data = json.dumps(profile_dict, ensure_ascii=False)

            # Create CSV content in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["format", "data"])
            writer.writeheader()
            writer.writerow({"format": "json", "data": json_data})

            data = output.getvalue().encode("utf-8")
            logger.debug(
                f"Successfully wrote CSV profile with {profile.metadata.n_clusters} clusters"
            )
            return data

        except Exception as e:
            error_msg = f"Failed to serialize CSV profile: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".csv"]
