"""MinIO storage models for profile data structures.

This module provides Pydantic models for validating profile data loaded from MinIO S3 storage.
All profile components (cluster centers, scaler parameters, etc.) are
strongly typed to catch data corruption early and provide better IDE support.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from nordlys.models.api import Model


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
        dtype: Numeric dtype for cluster centers ("float32" or "float64")
        silhouette_score: Cluster quality metric (-1 to 1, higher is better)
        allow_trust_remote_code: Allow remote code execution for embedding models
        clustering: K-means clustering configuration
        routing: Routing algorithm configuration
    """

    # Core clustering parameters
    n_clusters: int = Field(..., gt=0, description="Number of clusters")
    embedding_model: str = Field(..., description="Embedding model name")
    dtype: Literal["float32", "float64"] = Field(
        default="float32", description="Numeric dtype for cluster centers"
    )
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
