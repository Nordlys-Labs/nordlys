"""MinIO storage models for profile data structures.

This module provides Pydantic models for validating profile data loaded from MinIO S3 storage.
All profile components (cluster centers, scaler parameters, etc.) are
strongly typed to catch data corruption early and provide better IDE support.
"""

import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from adaptive_router.models.api import Model


class ClusterCentersData(BaseModel):
    """Cluster centers from K-means clustering.

    Attributes:
        n_clusters: Number of clusters (K)
        feature_dim: Dimensionality of feature space
        cluster_centers: K x D matrix of cluster centroids
    """

    n_clusters: int = Field(..., gt=0, description="Number of clusters")
    feature_dim: int = Field(..., gt=0, description="Feature dimensionality")
    cluster_centers: List[List[float]] = Field(
        ..., description="K x D cluster centroids"
    )


class ScalerParametersData(BaseModel):
    """StandardScaler parameters for feature normalization.

    Attributes:
        mean: Mean values for each feature dimension
        scale: Scale (standard deviation) for each feature dimension
    """

    mean: List[float] = Field(..., description="Feature means")
    scale: List[float] = Field(..., description="Feature scales")


class ScalerParameters(BaseModel):
    """All scaler parameters for feature normalization.

    Attributes:
        embedding_scaler: Scaler for semantic embedding features
    """

    embedding_scaler: ScalerParametersData = Field(
        ..., description="Embedding scaler parameters"
    )


class FeatureExtractionConfig(BaseModel):
    """Configuration for feature extraction.

    Attributes:
        batch_size_cuda: Batch size for CUDA devices
        batch_size_cpu: Batch size for CPU devices
        normalize_embeddings: Whether to L2-normalize embeddings
        embedding_cache_size: LRU cache size for embeddings
        allow_trust_remote_code: Allow remote code execution in embedding models
    """

    batch_size_cuda: int = Field(default=128, gt=0, description="Batch size for CUDA")
    batch_size_cpu: int = Field(default=32, gt=0, description="Batch size for CPU")
    normalize_embeddings: bool = Field(
        default=True, description="L2-normalize embeddings"
    )
    embedding_cache_size: int = Field(default=50000, gt=0, description="LRU cache size")
    allow_trust_remote_code: bool = Field(
        default=False, description="Allow remote code execution"
    )


class ClusteringConfig(BaseModel):
    """Configuration for K-means clustering.

    Attributes:
        max_iter: K-means max iterations
        random_state: Random seed for reproducibility
        n_init: Number of K-means initializations
        algorithm: K-means algorithm ('lloyd' or 'elkan')
        normalization_strategy: Feature normalization ('l2', 'l1', or 'max')
    """

    max_iter: int = Field(default=300, gt=0, description="K-means max iterations")
    random_state: int = Field(default=42, description="Random seed")
    n_init: int = Field(default=10, gt=0, description="K-means initializations")
    algorithm: str = Field(default="lloyd", description="K-means algorithm")
    normalization_strategy: str = Field(
        default="l2", description="Feature normalization"
    )


class RoutingConfig(BaseModel):
    """Configuration for routing algorithm.

    Attributes:
        lambda_min: Minimum lambda for cost-quality tradeoff
        lambda_max: Maximum lambda for cost-quality tradeoff
        default_cost_preference: Default cost preference (0.0=cheap, 1.0=quality)
    """

    lambda_min: float = Field(default=0.0, ge=0.0, description="Minimum lambda")
    lambda_max: float = Field(default=2.0, ge=0.0, description="Maximum lambda")
    default_cost_preference: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Default cost preference"
    )


class ProfileMetadata(BaseModel):
    """Metadata about the clustering profile.

    Attributes:
        n_clusters: Number of clusters
        embedding_model: HuggingFace embedding model name
        silhouette_score: Cluster quality metric
        feature_extraction: Feature extraction configuration
        clustering: Clustering configuration
        routing: Routing algorithm configuration
    """

    n_clusters: int = Field(..., gt=0, description="Number of clusters")
    embedding_model: str = Field(..., description="Embedding model name")
    silhouette_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

    # Configuration sections with default factories for backward compatibility
    feature_extraction: FeatureExtractionConfig = Field(
        default_factory=FeatureExtractionConfig,
        description="Feature extraction config",
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="Clustering config"
    )
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig, description="Routing config"
    )


class RouterProfile(BaseModel):
    """Complete router profile structure with validation.

    This is the top-level model for profile data loaded from storage.
    All nested components are validated to catch data corruption early.

    Attributes:
        cluster_centers: K-means cluster centroids
        models: List of models included in this profile
        llm_profiles: Model error rates per cluster (model_id -> K error rates)
        scaler_parameters: StandardScaler parameters for feature normalization
        metadata: Profile metadata (clustering config, silhouette score, etc.)
    """

    cluster_centers: ClusterCentersData = Field(..., description="Cluster centroids")
    models: List[Model] = Field(..., description="Models included in this profile")
    llm_profiles: Dict[str, List[float]] = Field(
        ..., description="Model error rates per cluster"
    )
    scaler_parameters: ScalerParameters = Field(..., description="Scaler parameters")
    metadata: ProfileMetadata = Field(..., description="Profile metadata")

    @field_validator("llm_profiles", mode="before")
    @classmethod
    def normalize_llm_profile_keys(
        cls, llm_profiles: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Normalize llm_profiles keys to lowercase for consistency.

        This ensures that model IDs in llm_profiles match the lowercased
        format produced by Model.unique_id(), preventing validation errors
        when users provide capitalized model names.

        Args:
            llm_profiles: Dictionary of model_id -> error_rates

        Returns:
            Dictionary with all keys normalized to lowercase
        """
        return {k.lower(): v for k, v in llm_profiles.items()}

    @field_validator("llm_profiles", mode="after")
    @classmethod
    def validate_error_rates(
        cls, llm_profiles: Dict[str, List[float]], info: ValidationInfo
    ) -> Dict[str, List[float]]:
        """Validate error rates for all models.

        Ensures:
        1. Each model has error_rates with length matching n_clusters
        2. All error rates are finite numbers within [0.0, 1.0]

        Args:
            llm_profiles: Dictionary of model_id -> error_rates
            info: ValidationInfo containing other field values

        Returns:
            Validated llm_profiles dictionary

        Raises:
            ValueError: If validation fails for any model
        """
        # Get n_clusters from metadata (if available)
        metadata = info.data.get("metadata")
        if metadata is None:
            # metadata hasn't been validated yet, skip cluster count validation
            # (will be caught if metadata is missing/invalid)
            return llm_profiles

        expected_clusters = metadata.n_clusters

        # Track invalid models for comprehensive error reporting
        validation_errors = []

        for model_id, error_rates in llm_profiles.items():
            # Check 1: Verify error_rates is a list
            if not isinstance(error_rates, list):
                validation_errors.append(
                    f"Model '{model_id}': error_rates must be a list, got {type(error_rates).__name__}"
                )
                continue

            # Check 2: Verify length matches n_clusters
            if len(error_rates) != expected_clusters:
                validation_errors.append(
                    f"Model '{model_id}': error_rates length mismatch - "
                    f"expected {expected_clusters} clusters, got {len(error_rates)}"
                )
                continue

            # Check 3: Validate each error rate value
            for i, rate in enumerate(error_rates):
                # Check if it's a number
                if not isinstance(rate, (int, float)):
                    validation_errors.append(
                        f"Model '{model_id}': error_rates[{i}] is not a number - "
                        f"got {type(rate).__name__}"
                    )
                    break

                # Check if finite (not NaN or Inf)
                if not math.isfinite(rate):
                    validation_errors.append(
                        f"Model '{model_id}': error_rates[{i}] is not finite - "
                        f"got {rate}"
                    )
                    break

                # Check range [0.0, 1.0]
                if not (0.0 <= rate <= 1.0):
                    validation_errors.append(
                        f"Model '{model_id}': error_rates[{i}] out of range [0.0, 1.0] - "
                        f"got {rate}"
                    )
                    break

        if validation_errors:
            error_msg = "LLM profiles validation failed:\n" + "\n".join(
                f"  - {err}" for err in validation_errors
            )
            raise ValueError(error_msg)

        return llm_profiles

    @model_validator(mode="after")
    def validate_models_consistency(self) -> "RouterProfile":
        """Validate consistency between models and llm_profiles.

        Ensures that model IDs in the models list match the keys in llm_profiles.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If model IDs are inconsistent
        """
        model_ids_in_models = {m.unique_id() for m in self.models}
        model_ids_in_llm = set(self.llm_profiles.keys())

        if model_ids_in_models != model_ids_in_llm:
            missing_in_llm = model_ids_in_models - model_ids_in_llm
            extra_in_llm = model_ids_in_llm - model_ids_in_models

            errors = []
            if missing_in_llm:
                errors.append(
                    f"Models present in 'models' but missing in 'llm_profiles': {missing_in_llm}"
                )
            if extra_in_llm:
                errors.append(
                    f"Models present in 'llm_profiles' but missing in 'models': {extra_in_llm}"
                )

            error_msg = "Model IDs inconsistency:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        return self


class MinIOSettings(BaseModel):
    """MinIO storage configuration for library usage.

    This class requires explicit constructor arguments for all parameters.
    It does not automatically read from environment variables.

    Args:
        endpoint_url: MinIO endpoint URL (must start with http:// or https://)
        root_user: MinIO root username
        root_password: MinIO root password
        bucket_name: S3 bucket name
        region: AWS region (default: us-east-1, ignored by MinIO but required by boto3)
        profile_key: Key for profile in bucket (default: global/profile.json)
        connect_timeout: Connection timeout in seconds (default: 5)
        read_timeout: Read timeout in seconds (default: 30)

    Example:
        >>> from adaptive_router.models.storage import MinIOSettings
        >>> settings = MinIOSettings(
        ...     endpoint_url="https://minio.example.com",
        ...     root_user="admin",
        ...     root_password="password123",
        ...     bucket_name="adaptive-router-profiles"
        ... )
    """

    endpoint_url: str = Field(..., description="MinIO endpoint URL")
    root_user: str = Field(..., description="MinIO root username")
    root_password: str = Field(..., description="MinIO root password")

    bucket_name: str = Field(..., description="S3 bucket name")
    region: str = Field(default="us-east-1", description="AWS region")
    profile_key: str = Field(
        default="global/profile.json", description="Profile key in bucket"
    )

    # Timeout configuration (configurable for different network conditions)
    connect_timeout: int = Field(default=5, description="Connection timeout in seconds")
    read_timeout: int = Field(default=30, description="Read timeout in seconds")

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate that endpoint_url is a valid URL.

        Args:
            v: The endpoint URL to validate

        Returns:
            The validated URL

        Raises:
            ValueError: If URL doesn't start with http:// or https://
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"endpoint_url must start with http:// or https://, got: {v}"
            )
        return v

    @field_validator("bucket_name")
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate that bucket_name is not empty.

        Args:
            v: The bucket name to validate

        Returns:
            The validated bucket name

        Raises:
            ValueError: If bucket name is empty
        """
        if not v or not v.strip():
            raise ValueError("bucket_name cannot be empty")
        return v.strip()
