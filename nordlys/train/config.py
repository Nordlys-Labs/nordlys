"""TOML Configuration Schema for Adaptive Router Training.

This module defines Pydantic models for parsing and validating TOML configuration
files used by the training script. It supports hybrid model loading (API fetch or
full TOML definition) and multiple output formats (local, S3, MinIO).

Example TOML:
    [api]
    adaptive_api_key = "${ADAPTIVE_API_KEY}"  # or direct value

    [dataset]
    path = "data/train.csv"
    type = "csv"
    input_column = "input"
    expected_column = "expected_output"

    [training]
    n_clusters = 20
    max_parallel = 10

    [[models]]
    provider = "openai"
    model_name = "gpt-4"
    # Pricing optional - will fetch from API if missing

    [[models]]
    provider = "openai"
    model_name = "gpt-3.5-turbo"
    cost_per_1m_input_tokens = 0.50
    cost_per_1m_output_tokens = 1.50

    [providers.openai]
    api_key = "${OPENAI_API_KEY}"
    timeout = 60.0

    [output]
    path = "profile.json"
    storage_type = "local"
"""

import os
import sys
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Import TOML library based on Python version
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class APIConfig(BaseModel):
    """API configuration for Adaptive Models API.

    Attributes:
        adaptive_api_key: API key for Adaptive Models API (supports ${ENV_VAR} syntax)
        base_url: Optional custom API base URL
    """

    adaptive_api_key: str
    base_url: str = "https://api.llmadaptive.uk/v1"

    @field_validator("adaptive_api_key")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable syntax ${VAR_NAME}."""
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        path: Path to dataset file
        type: Dataset format (csv, json, or parquet)
        input_column: Column name for input prompts
        expected_column: Column name for expected outputs
    """

    path: str
    type: Literal["csv", "json", "parquet"]
    input_column: str = "input"
    expected_column: str = "expected_output"

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that dataset file exists."""
        if not os.path.exists(v):
            raise ValueError(f"Dataset file not found: {v}")
        return v


class ModelConfigToml(BaseModel):
    """Model configuration in TOML.

    Supports hybrid mode: pricing can be provided in TOML or fetched from API.

    Attributes:
        provider: Model provider (e.g., "openai", "anthropic")
        model_name: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022")
        cost_per_1m_input_tokens: Optional cost per 1M input tokens
        cost_per_1m_output_tokens: Optional cost per 1M output tokens
    """

    provider: str
    model_name: str
    cost_per_1m_input_tokens: Optional[float] = None
    cost_per_1m_output_tokens: Optional[float] = None

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Normalize provider to lowercase."""
        return v.strip().lower()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_pricing(self) -> "ModelConfigToml":
        """Validate that both pricing fields are provided together if any is set."""
        has_input = self.cost_per_1m_input_tokens is not None
        has_output = self.cost_per_1m_output_tokens is not None

        if has_input != has_output:
            raise ValueError(
                f"Model {self.provider}/{self.model_name}: "
                "Both cost_per_1m_input_tokens and cost_per_1m_output_tokens "
                "must be provided together, or both omitted (to fetch from API)"
            )

        if (
            self.cost_per_1m_input_tokens is not None
            and self.cost_per_1m_input_tokens < 0
        ):
            raise ValueError("cost_per_1m_input_tokens must be >= 0")
        if (
            self.cost_per_1m_output_tokens is not None
            and self.cost_per_1m_output_tokens < 0
        ):
            raise ValueError("cost_per_1m_output_tokens must be >= 0")

        return self

    @property
    def has_pricing(self) -> bool:
        """Check if pricing is defined in TOML."""
        return self.cost_per_1m_input_tokens is not None

    def model_spec(self) -> str:
        """Return model specification in provider/model_name format."""
        return f"{self.provider}/{self.model_name}"


class ProviderConfigToml(BaseModel):
    """Provider API configuration.

    Attributes:
        api_key: API key for provider (supports ${ENV_VAR} syntax)
        base_url: Optional custom API endpoint
        organization: Optional organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """

    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3

    @field_validator("api_key")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable syntax ${VAR_NAME}."""
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        """Validate max_retries is non-negative."""
        if v < 0:
            raise ValueError("max_retries must be >= 0")
        return v


class TrainingParams(BaseModel):
    """Training hyperparameters.

    Attributes:
        n_clusters: Number of clusters for K-means
        max_parallel: Maximum parallel model inference requests
        embedding_model: Sentence transformer model for embeddings
        tfidf_max_features: Maximum TF-IDF features
        tfidf_ngram_range: TF-IDF n-gram range [min, max]
        random_seed: Random seed for reproducibility
    """

    n_clusters: int = 20
    max_parallel: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_max_features: int = 5000
    tfidf_ngram_range: List[int] = Field(default=[1, 2])
    random_seed: int = 42

    @field_validator("n_clusters")
    @classmethod
    def validate_n_clusters(cls, v: int) -> int:
        """Validate n_clusters is positive."""
        if v <= 0:
            raise ValueError("n_clusters must be positive")
        return v

    @field_validator("max_parallel")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        """Validate max_parallel is positive."""
        if v <= 0:
            raise ValueError("max_parallel must be positive")
        return v

    @field_validator("tfidf_max_features")
    @classmethod
    def validate_tfidf_max_features(cls, v: int) -> int:
        """Validate tfidf_max_features is positive."""
        if v <= 0:
            raise ValueError("tfidf_max_features must be positive")
        return v

    @field_validator("tfidf_ngram_range")
    @classmethod
    def validate_tfidf_ngram_range(cls, v: List[int]) -> List[int]:
        """Validate tfidf_ngram_range is [min, max] with min <= max."""
        if len(v) != 2:
            raise ValueError("tfidf_ngram_range must be [min, max]")
        if v[0] < 1 or v[1] < 1:
            raise ValueError("tfidf_ngram_range values must be >= 1")
        if v[0] > v[1]:
            raise ValueError("tfidf_ngram_range min must be <= max")
        return v


class S3Config(BaseModel):
    """S3/MinIO storage configuration.

    Attributes:
        endpoint_url: S3 endpoint URL (for MinIO or custom S3)
        access_key_id: AWS access key or MinIO root user
        secret_access_key: AWS secret key or MinIO root password
        bucket_name: S3 bucket name
        region: AWS region
        profile_key: Object key (path) in bucket
    """

    endpoint_url: Optional[str] = None
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    region: str = "us-east-1"
    profile_key: str = "profile.json"

    @field_validator("access_key_id", "secret_access_key")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable syntax ${VAR_NAME}."""
        if v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Environment variable {env_var} not set")
            return value
        return v


class OutputConfig(BaseModel):
    """Output configuration.

    Attributes:
        path: Output path (local file path for local storage)
        storage_type: Storage backend (local, s3, or minio)
        s3: Optional S3/MinIO configuration (required if storage_type is s3/minio)
    """

    path: str
    storage_type: Literal["local", "s3", "minio"] = "local"
    s3: Optional[S3Config] = None

    @model_validator(mode="after")
    def validate_s3_config(self) -> "OutputConfig":
        """Validate S3 config is provided if storage_type is s3/minio."""
        if self.storage_type in ["s3", "minio"] and not self.s3:
            raise ValueError(
                f"storage_type={self.storage_type} requires [output.s3] configuration"
            )
        return self


class TrainingConfig(BaseModel):
    """Root TOML configuration for training.

    Attributes:
        api: Adaptive API configuration
        dataset: Dataset configuration
        models: List of models to train
        providers: Provider API configurations (keyed by provider name)
        training: Training hyperparameters
        output: Output configuration
    """

    api: APIConfig
    dataset: DatasetConfig
    models: List[ModelConfigToml]
    providers: Dict[str, ProviderConfigToml] = Field(default_factory=dict)
    training: TrainingParams = Field(default_factory=TrainingParams)
    output: OutputConfig

    @field_validator("models")
    @classmethod
    def validate_models_not_empty(
        cls, v: List[ModelConfigToml]
    ) -> List[ModelConfigToml]:
        """Validate at least one model is defined."""
        if not v:
            raise ValueError("At least one model must be defined in [[models]]")
        return v

    @model_validator(mode="after")
    def validate_providers_match_models(self) -> "TrainingConfig":
        """Validate all model providers have corresponding provider configs."""
        required_providers = {m.provider for m in self.models}

        # Skip 'local' provider - doesn't need API key
        required_providers.discard("local")

        missing_providers = required_providers - set(self.providers.keys())

        if missing_providers:
            raise ValueError(
                f"Missing provider configurations: {', '.join(missing_providers)}\n"
                f"Add [providers.{next(iter(missing_providers))}] section to TOML"
            )

        return self


def load_config(toml_path: str) -> TrainingConfig:
    """Load and validate TOML configuration.

    Args:
        toml_path: Path to TOML configuration file

    Returns:
        Validated TrainingConfig object

    Raises:
        FileNotFoundError: If TOML file not found
        ValueError: If TOML is invalid or validation fails
    """
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"Configuration file not found: {toml_path}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    return TrainingConfig(**data)
