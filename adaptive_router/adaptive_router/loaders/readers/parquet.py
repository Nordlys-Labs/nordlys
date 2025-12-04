"""Parquet profile reader implementation."""

import json
import logging
from pathlib import Path

import polars as pl

from adaptive_router.loaders.readers.base import ProfileReader
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class ParquetProfileReader(ProfileReader):
    """Reader for Parquet format RouterProfiles.

    Parquet format stores the profile as JSON data in a single row.
    Expected format: Parquet file with columns 'format' and 'data', where
    'format' is 'json' and 'data' contains the JSON-serialized profile.
    """

    def read_from_path(self, path: Path) -> RouterProfile:
        """Read profile from Parquet file path.

        Args:
            path: Path to Parquet profile file

        Returns:
            Loaded RouterProfile

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If Parquet format is invalid or profile validation fails
        """
        logger.debug(f"Reading Parquet profile from: {path}")

        try:
            # Read Parquet file
            df = pl.read_parquet(path)

            if df.is_empty():
                raise ValueError(f"Parquet file {path} is empty")

            if df.height > 1:
                logger.warning(
                    f"Parquet file {path} has multiple rows, using first row"
                )

            row = df.row(0, named=True)

            # Check format column
            if "format" not in row:
                raise ValueError(f"Parquet file {path} missing 'format' column")
            if row["format"] != "json":
                raise ValueError(
                    f"Unsupported format '{row['format']}' in Parquet file {path}"
                )

            # Check data column
            if "data" not in row:
                raise ValueError(f"Parquet file {path} missing 'data' column")

            # Parse JSON data
            profile_dict = json.loads(row["data"])
            profile = RouterProfile(**profile_dict)
            logger.debug(
                f"Successfully read Parquet profile with {profile.metadata.n_clusters} clusters"
            )
            return profile

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON data in Parquet file {path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed for {path}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            raise

    def read_from_bytes(self, data: bytes) -> RouterProfile:
        """Read profile from Parquet bytes (for MinIO/S3).

        Args:
            data: Raw Parquet bytes

        Returns:
            Loaded RouterProfile

        Raises:
            ValueError: If Parquet format is invalid or profile validation fails
        """
        logger.debug(f"Reading Parquet profile from bytes ({len(data)} bytes)")

        try:
            # Read Parquet from bytes using polars
            import io

            df = pl.read_parquet(io.BytesIO(data))

            if df.is_empty():
                raise ValueError("Parquet data is empty")

            if df.height > 1:
                logger.warning("Parquet data has multiple rows, using first row")

            row = df.row(0, named=True)

            # Check format column
            if "format" not in row:
                raise ValueError("Parquet data missing 'format' column")
            if row["format"] != "json":
                raise ValueError(
                    f"Unsupported format '{row['format']}' in Parquet data"
                )

            # Check data column
            if "data" not in row:
                raise ValueError("Parquet data missing 'data' column")

            # Parse JSON data
            profile_dict = json.loads(row["data"])
            profile = RouterProfile(**profile_dict)
            logger.debug(
                f"Successfully read Parquet profile with {profile.metadata.n_clusters} clusters"
            )
            return profile

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON data in Parquet: {e}"
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
        return [".parquet", ".pq"]
