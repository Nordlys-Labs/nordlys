import logging
from pathlib import Path

from adaptive_router.loaders.base import ProfileLoader
from adaptive_router.loaders.readers import get_reader
from adaptive_router.models.storage import RouterProfile

logger = logging.getLogger(__name__)


class LocalFileProfileLoader(ProfileLoader):
    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)

        if not self.profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {self.profile_path}")

        logger.info(f"LocalFileProfileLoader initialized: {self.profile_path}")

    def load_profile(self) -> RouterProfile:
        logger.info(f"Loading profile from local file: {self.profile_path}")

        try:
            # Auto-detect format from file extension and get appropriate reader
            reader = get_reader(self.profile_path)
            profile = reader.read_from_path(self.profile_path)

            logger.info(
                f"Successfully loaded profile from local file "
                f"(format: {self.profile_path.suffix}, n_clusters: {profile.metadata.n_clusters})"
            )

            return profile

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            raise

    def health_check(self) -> bool:
        return self.profile_path.exists() and self.profile_path.is_file()
