import json
import logging

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .base import ProfileLoader
from ..models.storage import RouterProfile, MinIOSettings

logger = logging.getLogger(__name__)


class MinIOProfileLoader(ProfileLoader):
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
            f"MinIOProfileLoader initialized: bucket={bucket_name}, "
            f"key={profile_key}, endpoint={endpoint_url}, "
            f"timeouts=(connect:{connect_timeout}s, read:{read_timeout}s)"
        )

    @classmethod
    def from_settings(cls, settings: MinIOSettings) -> "MinIOProfileLoader":
        return cls(
            bucket_name=settings.bucket_name,
            region=settings.region,
            profile_key=settings.profile_key,
            endpoint_url=settings.endpoint_url,
            access_key_id=settings.root_user,
            secret_access_key=settings.root_password,
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
        )

    def load_profile(self) -> RouterProfile:
        try:
            logger.info(
                f"Loading profile from MinIO: s3://{self.bucket_name}/{self.profile_key}"
            )

            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.profile_key)

            data = response["Body"].read()
            profile_dict = json.loads(data)

            profile = RouterProfile(**profile_dict)

            size_kb = len(data) / 1024
            logger.info(
                f"Successfully loaded and validated profile from MinIO "
                f"(size: {size_kb:.1f}KB, n_clusters: {profile.metadata.n_clusters})"
            )

            return profile

        except json.JSONDecodeError as e:
            error_msg = f"Corrupted JSON in MinIO profile: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                error_msg = f"Profile not found in MinIO: s3://{self.bucket_name}/{self.profile_key}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                error_msg = f"MinIO error loading profile: {error_code} - {e}"
                logger.error(error_msg)
                raise

        except Exception as e:
            if "validation error" in str(e).lower():
                error_msg = f"Profile validation failed: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            raise

    def health_check(self) -> bool:
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"MinIO bucket {self.bucket_name} is accessible")
            return True
        except ClientError as e:
            logger.warning(f"MinIO bucket {self.bucket_name} not accessible: {e}")
            return False
