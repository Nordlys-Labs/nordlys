"""Application configuration settings."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Profile settings
    profile_path: str = Field(
        default="/data/profile.json",
        description="Path to the RouterProfile JSON file",
    )

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
