"""CSV profile reader implementation."""

import csv
import json
import logging
from pathlib import Path

from adaptive_router.loaders.readers.base import ProfileReader
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class CSVProfileReader(ProfileReader):
    """Reader for CSV format RouterProfiles.

    CSV format stores the profile as JSON data in a single row.
    Expected format: CSV with columns 'format' and 'data', where
    'format' is 'json' and 'data' contains the JSON-serialized profile.
    """

    def read_from_path(self, path: Path) -> RouterProfile:
        """Read profile from CSV file path.

        Args:
            path: Path to CSV profile file

        Returns:
            Loaded RouterProfile

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid or profile validation fails
        """
        logger.debug(f"Reading CSV profile from: {path}")

        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    raise ValueError(f"CSV file {path} is empty")

                if len(rows) > 1:
                    logger.warning(
                        f"CSV file {path} has multiple rows, using first row"
                    )

                row = rows[0]

                # Check format column
                if "format" not in row:
                    raise ValueError(f"CSV file {path} missing 'format' column")
                if row["format"] != "json":
                    raise ValueError(
                        f"Unsupported format '{row['format']}' in CSV file {path}"
                    )

                # Check data column
                if "data" not in row:
                    raise ValueError(f"CSV file {path} missing 'data' column")

                # Parse JSON data
                profile_dict = json.loads(row["data"])
                profile = RouterProfile(**profile_dict)
                logger.debug(
                    f"Successfully read CSV profile with {profile.metadata.n_clusters} clusters"
                )
                return profile

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON data in CSV file {path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed for {path}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            raise

    def read_from_bytes(self, data: bytes) -> RouterProfile:
        """Read profile from CSV bytes (for MinIO/S3).

        Args:
            data: Raw CSV bytes

        Returns:
            Loaded RouterProfile

        Raises:
            ValueError: If CSV format is invalid or profile validation fails
        """
        logger.debug(f"Reading CSV profile from bytes ({len(data)} bytes)")

        try:
            # Decode bytes to string
            csv_content = data.decode("utf-8")

            # Parse CSV
            reader = csv.DictReader(csv_content.splitlines())
            rows = list(reader)

            if not rows:
                raise ValueError("CSV data is empty")

            if len(rows) > 1:
                logger.warning("CSV data has multiple rows, using first row")

            row = rows[0]

            # Check format column
            if "format" not in row:
                raise ValueError("CSV data missing 'format' column")
            if row["format"] != "json":
                raise ValueError(f"Unsupported format '{row['format']}' in CSV data")

            # Check data column
            if "data" not in row:
                raise ValueError("CSV data missing 'data' column")

            # Parse JSON data
            profile_dict = json.loads(row["data"])
            profile = RouterProfile(**profile_dict)
            logger.debug(
                f"Successfully read CSV profile with {profile.metadata.n_clusters} clusters"
            )
            return profile

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_msg = f"Invalid data in CSV: {e}"
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
        return [".csv"]
