"""Parquet profile writer implementation."""

import json
import logging
from pathlib import Path

import polars as pl

from adaptive_router.savers.writers.base import ProfileWriter
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class ParquetProfileWriter(ProfileWriter):
    """Writer for Parquet format RouterProfiles.

    Parquet format stores the profile as JSON data in a single row.
    Output format: Parquet file with columns 'format' and 'data', where
    'format' is 'json' and 'data' contains the JSON-serialized profile.
    """

    def write_to_path(self, profile: RouterProfile, path: Path) -> None:
        """Write profile to Parquet file path.

        Args:
            profile: RouterProfile to save
            path: Destination file path

        Raises:
            IOError: If write fails
        """
        logger.debug(f"Writing Parquet profile to: {path}")

        try:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize profile to JSON
            profile_dict = profile.model_dump()
            json_data = json.dumps(profile_dict, ensure_ascii=False)

            # Create DataFrame with single row
            df = pl.DataFrame([{"format": "json", "data": json_data}])

            # Write to Parquet
            df.write_parquet(path)

            logger.debug(
                f"Successfully wrote Parquet profile with {profile.metadata.n_clusters} clusters"
            )

        except Exception as e:
            error_msg = f"Failed to write Parquet profile to {path}: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def write_to_bytes(self, profile: RouterProfile) -> bytes:
        """Write profile to Parquet bytes (for MinIO/S3).

        Args:
            profile: RouterProfile to save

        Returns:
            Serialized profile as bytes
        """
        logger.debug("Writing Parquet profile to bytes")

        try:
            # Serialize profile to JSON
            profile_dict = profile.model_dump()
            json_data = json.dumps(profile_dict, ensure_ascii=False)

            # Create DataFrame with single row
            df = pl.DataFrame([{"format": "json", "data": json_data}])

            # Write to bytes using BytesIO
            import io

            buffer = io.BytesIO()
            df.write_parquet(buffer)
            data = buffer.getvalue()

            logger.debug(
                f"Successfully wrote Parquet profile with {profile.metadata.n_clusters} clusters"
            )
            return data

        except Exception as e:
            error_msg = f"Failed to serialize Parquet profile: {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".parquet", ".pq"]
