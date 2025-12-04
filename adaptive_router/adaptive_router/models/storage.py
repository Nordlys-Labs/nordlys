"""MinIO storage models for profile data structures.

This module provides Pydantic models for validating profile data loaded from MinIO S3 storage.
All profile components (cluster centers, scaler parameters, etc.) are
strongly typed to catch data corruption early and provide better IDE support.
"""

from pydantic import BaseModel, Field, field_validator, model_validator

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
    cluster_centers: list[list[float]] = Field(
        ..., description="K x D cluster centroids"
    )


class ClusteringConfig(BaseModel):
    """Configuration for K-means clustering.

    Attributes:
        max_iter: K-means max iterations
        random_state: Random seed for reproducibility
        n_init: Number of K-means initializations
        algorithm: K-means algorithm ('lloyd' or 'elkan')
        normalization_strategy: Feature normalization ('l2', 'l1', or 'max')
        n_iter: Actual K-means iterations from training
    """

    max_iter: int = Field(default=300, gt=0, description="K-means max iterations")
    random_state: int = Field(default=42, description="Random seed")
    n_init: int = Field(default=10, gt=0, description="K-means initializations")
    algorithm: str = Field(default="lloyd", description="K-means algorithm")
    normalization_strategy: str = Field(
        default="l2", description="Feature normalization"
    )
    n_iter: int = Field(
        default=0, ge=0, description="Actual K-means iterations from training"
    )


class RoutingConfig(BaseModel):
    """Configuration for routing algorithm.

    Attributes:
        lambda_min: Minimum lambda for cost-quality tradeoff
        lambda_max: Maximum lambda for cost-quality tradeoff
        default_cost_preference: Default cost preference (0.0=cheap, 1.0=quality)
    """

    lambda_min: float = Field(
        default=0.0, ge=0.0, description="Minimum lambda for cost-quality tradeoff"
    )
    lambda_max: float = Field(
        default=2.0, ge=0.0, description="Maximum lambda for cost-quality tradeoff"
    )
    default_cost_preference: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default cost preference (0.0=cheap, 1.0=quality)",
    )


class ClusterStats(BaseModel):
    """Statistics about clustering results.

    Attributes:
        n_clusters: Number of clusters
        n_samples: Total number of samples
        silhouette_score: Clustering quality metric (-1 to 1)
        cluster_sizes: Dictionary mapping cluster ID to number of samples
        min_cluster_size: Size of smallest cluster
        max_cluster_size: Size of largest cluster
        avg_cluster_size: Average cluster size
    """

    n_clusters: int
    n_samples: int
    silhouette_score: float
    cluster_sizes: dict[int, int]
    min_cluster_size: int
    max_cluster_size: int
    avg_cluster_size: float


class ProfileMetadata(BaseModel):
    """Metadata about the clustering profile.

    Attributes:
        n_clusters: Number of clusters (K in K-means)
        embedding_model: HuggingFace embedding model name
        silhouette_score: Cluster quality metric (-1 to 1, higher is better)
        allow_trust_remote_code: Allow remote code execution for embedding models
        clustering: K-means clustering configuration
        routing: Routing algorithm configuration
    """

    # Core clustering parameters
    n_clusters: int = Field(..., gt=0, description="Number of clusters")
    embedding_model: str = Field(..., description="Embedding model name")
    silhouette_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Cluster quality metric"
    )

    # Configuration
    allow_trust_remote_code: bool = Field(
        default=False, description="Allow remote code execution for embedding models"
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="K-means clustering configuration"
    )
    routing: RoutingConfig = Field(
        default_factory=RoutingConfig, description="Routing algorithm configuration"
    )


class RouterProfile(BaseModel):
    """Complete router profile structure with validation.

    This is the top-level model for profile data loaded from storage.
    All nested components are validated to catch data corruption early.

    Attributes:
        cluster_centers: K-means cluster centroids
        models: List of models with integrated error rates per cluster
        metadata: Profile metadata (clustering config, silhouette score, etc.)
    """

    cluster_centers: ClusterCentersData = Field(..., description="Cluster centroids")
    models: list[Model] = Field(..., description="Models with error rates")
    metadata: ProfileMetadata = Field(..., description="Profile metadata")

    @model_validator(mode="after")
    def validate_error_rates_consistency(self) -> "RouterProfile":
        """Validate all models have correct number of error rates.

        Ensures each model's error_rates list matches the n_clusters in metadata.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If error rates count doesn't match n_clusters
        """
        expected = self.metadata.n_clusters
        errors = []

        for model in self.models:
            if len(model.error_rates) != expected:
                errors.append(
                    f"Model '{model.unique_id()}': expected {expected} "
                    f"error rates, got {len(model.error_rates)}"
                )

        if errors:
            raise ValueError("Error rates validation failed:\n" + "\n".join(errors))

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
