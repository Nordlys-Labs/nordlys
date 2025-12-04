"""MinIO/S3 profile saver implementation."""

import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from adaptive_router.savers.base import ProfileSaver
from adaptive_router.savers.writers import get_writer
from adaptive_router.models.storage import RouterProfile, MinIOSettings

logger = logging.getLogger(__name__)


class MinIOProfileSaver(ProfileSaver):
    """Saver for RouterProfiles to MinIO/S3 storage.

    Auto-detects format from file extension and uses appropriate writer
    from savers.writers module. Uploads serialized profile to S3-compatible storage.
    """

    def __init__(
        self,
        bucket_name: str,
        region: str,
        profile_key: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        connect_timeout: int = 5,
        read_timeout: int = 30,
    ):
        """Initialize MinIO profile saver.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            profile_key: Key for profile in bucket
            endpoint_url: MinIO/S3 endpoint URL
            access_key_id: Access key ID
            secret_access_key: Secret access key
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
        """
        self.bucket_name = bucket_name
        self.profile_key = profile_key
        self.endpoint_url = endpoint_url

        config = Config(
            region_name=region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            config=config,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        logger.info(
            f"MinIOProfileSaver initialized: bucket={bucket_name}, "
            f"key={profile_key}, endpoint={endpoint_url}, "
            f"timeouts=(connect:{connect_timeout}s, read:{read_timeout}s)"
        )

    @classmethod
    def from_settings(cls, settings: MinIOSettings, **kwargs) -> "MinIOProfileSaver":
        """Create saver from MinIOSettings.

        Args:
            settings: MinIO configuration settings
            **kwargs: Additional arguments to override settings

        Returns:
            Configured MinIOProfileSaver instance
        """
        return cls(
            bucket_name=settings.bucket_name,
            region=settings.region,
            profile_key=settings.profile_key,
            endpoint_url=settings.endpoint_url,
            access_key_id=settings.root_user,
            secret_access_key=settings.root_password,
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
            **kwargs,
        )

    def save_profile(self, profile: RouterProfile, path: str) -> str:
        """Save profile to MinIO/S3.

        Path format: s3://bucket/key or minio://bucket/key
        Auto-detects format from file extension

        Args:
            profile: RouterProfile to save
            path: S3 path (s3://bucket/key or minio://bucket/key)

        Returns:
            Path where profile was saved

        Raises:
            ValueError: If path format is invalid
            IOError: If upload fails
        """
        # Parse S3 path: s3://bucket/key or minio://bucket/key
        if path.startswith("s3://"):
            # Extract bucket and key from s3://bucket/key
            path_parts = path[5:].split("/", 1)  # Remove 's3://'
            if len(path_parts) != 2:
                raise ValueError(
                    f"Invalid S3 path format: {path}. Expected: s3://bucket/key"
                )
            bucket, key = path_parts
        elif path.startswith("minio://"):
            # Extract bucket and key from minio://bucket/key
            path_parts = path[8:].split("/", 1)  # Remove 'minio://'
            if len(path_parts) != 2:
                raise ValueError(
                    f"Invalid MinIO path format: {path}. Expected: minio://bucket/key"
                )
            bucket, key = path_parts
        else:
            raise ValueError(
                f"Invalid path format: {path}. Expected s3://bucket/key or minio://bucket/key"
            )

        logger.info(f"Saving profile to MinIO: s3://{bucket}/{key}")

        try:
            # Auto-detect format from file extension and get appropriate writer
            writer = get_writer(key)
            profile_bytes = writer.write_to_bytes(profile)

            # Upload to S3
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=profile_bytes,
                ContentType="application/json",  # All formats store JSON data
            )

            logger.info(
                f"Successfully saved profile to MinIO "
                f"(format: {key.split('.')[-1]}, n_clusters: {profile.metadata.n_clusters}, "
                f"size: {len(profile_bytes)} bytes)"
            )

            return path

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = f"MinIO upload failed: {error_code} - {e}"
            logger.error(error_msg)
            raise IOError(error_msg) from e

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            raise

    def health_check(self) -> bool:
        """Check if MinIO/S3 connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"MinIO bucket {self.bucket_name} is accessible")
            return True
        except ClientError as e:
            logger.warning(f"MinIO bucket {self.bucket_name} not accessible: {e}")
            return False
