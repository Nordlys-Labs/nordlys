"""Application configuration settings."""

from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants
DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
FUZZY_MATCH_SIMILARITY_THRESHOLD = 0.8


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.PRODUCTION,
        description="Deployment environment (development or production)",
    )

    # MinIO/S3 settings
    minio_private_endpoint: str | None = Field(
        default=None,
        description="Private MinIO endpoint URL",
    )
    minio_public_endpoint: str | None = Field(
        default=None,
        description="Public MinIO endpoint URL",
    )
    minio_root_user: str = Field(
        default=...,
        description="MinIO root user (required)",
    )
    minio_root_password: str = Field(
        default=...,
        description="MinIO root password (required)",
    )
    s3_bucket_name: str = Field(
        default="adaptive-router-profiles",
        description="S3 bucket name",
    )
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_profile_key: str = Field(
        default="global/profile.json",
        description="S3 profile key path",
    )
    s3_connect_timeout: str = Field(default="5", description="S3 connect timeout")
    s3_read_timeout: str = Field(default="30", description="S3 read timeout")

    # CORS settings
    allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins (dev mode allows all)",
    )

    @property
    def origins_list(self) -> list[str]:
        """Parse allowed origins into a list.

        In development: Allows all origins (["*"]) for easier testing.
        In production: Requires explicit ALLOWED_ORIGINS configuration.

        Example: ALLOWED_ORIGINS="https://example.com,https://app.example.com"
        """
        # Dev mode: allow all
        if self.environment == Environment.DEVELOPMENT:
            return ["*"]

        # Prod mode: require explicit configuration
        if not self.allowed_origins:
            return []
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def minio_endpoint(self) -> str:
        """Return the configured MinIO endpoint.

        Security: Requires explicit endpoint configuration - no localhost fallback.
        Use minio_private_endpoint for internal networking, minio_public_endpoint for external.

        Raises:
            ValueError: If no MinIO endpoint is configured
        """
        if self.minio_private_endpoint and self.minio_private_endpoint.strip():
            return self.minio_private_endpoint.strip()
        if self.minio_public_endpoint and self.minio_public_endpoint.strip():
            return self.minio_public_endpoint.strip()

        raise ValueError(
            "MinIO endpoint not configured. Set either MINIO_PRIVATE_ENDPOINT "
            "or MINIO_PUBLIC_ENDPOINT environment variable."
        )

    @field_validator("minio_root_user", "minio_root_password", mode="after")
    @classmethod
    def validate_minio_credentials(cls, v: str) -> str:
        """Validate that MinIO credentials are provided and non-empty.

        Raises:
            ValueError: If credentials are not set or are empty
        """
        if not v or not v.strip():
            raise ValueError(
                "MinIO credentials required. Set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD "
                "environment variables."
            )
        return v
