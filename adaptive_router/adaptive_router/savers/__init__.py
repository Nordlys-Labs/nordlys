"""Savers registry and factory functions."""

from urllib.parse import urlparse

from adaptive_router.models.storage import MinIOSettings

from .base import ProfileSaver
from .local import LocalFileProfileSaver
from .minio import MinIOProfileSaver


def get_saver(
    path: str,
    minio_settings: MinIOSettings | None = None,
    s3_settings: MinIOSettings | None = None,
    **kwargs,
) -> ProfileSaver:
    """Auto-detect and return appropriate saver based on path.

    Args:
        path: Destination path (s3://, minio://, or local path)
        minio_settings: MinIO configuration for minio:// URLs
        s3_settings: S3 configuration for s3:// URLs
        **kwargs: Additional config for MinIO/S3 saver

    Returns:
        ProfileSaver instance (LocalFileProfileSaver or MinIOProfileSaver)

    Raises:
        ValueError: If MinIO/S3 settings required but not provided

    Examples:
        >>> get_saver("profile.json")  # LocalFileProfileSaver
        >>> get_saver("s3://bucket/profile.json", s3_settings=settings)  # MinIOProfileSaver
        >>> get_saver("minio://bucket/profile.json", minio_settings=settings)  # MinIOProfileSaver
    """
    if path.startswith("s3://"):
        if s3_settings is None:
            raise ValueError(
                f"S3 settings required for S3 path: {path}. "
                "Provide s3_settings parameter."
            )
        return MinIOProfileSaver.from_settings(s3_settings, **kwargs)
    elif urlparse(path).scheme == "minio":
        if minio_settings is None:
            raise ValueError(
                f"MinIO settings required for MinIO path: {path}. "
                "Provide minio_settings parameter."
            )
        return MinIOProfileSaver.from_settings(minio_settings, **kwargs)
    else:
        return LocalFileProfileSaver()
