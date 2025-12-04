"""Training-related data structures."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for LLM provider during training inference.

    Attributes:
        api_key: API key for the provider
        base_url: Optional custom base URL for API endpoint
        organization: Optional organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts for failed requests
    """

    api_key: str = Field(..., description="API key for provider")
    base_url: Optional[str] = Field(
        None, description="Custom base URL for API endpoint"
    )
    organization: Optional[str] = Field(
        None, description="Organization ID (if applicable)"
    )
    timeout: Optional[float] = Field(60.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts", ge=0)


class TrainingResult(BaseModel):
    """Result from the training process.

    Attributes:
        n_clusters: Number of clusters created
        silhouette_score: Cluster quality metric (higher is better)
        n_models: Number of models included in the profile
        model_ids: List of model identifiers
        error_rates: Error rate per model per cluster
        total_samples: Total number of training samples
        samples_per_cluster: Number of samples assigned to each cluster
        inference_time: Total time spent on model inference (seconds)
        training_time: Total training time (seconds)
    """

    n_clusters: int = Field(..., description="Number of clusters", gt=0)
    silhouette_score: float = Field(..., description="Cluster quality score")
    n_models: int = Field(..., description="Number of models trained", gt=0)
    model_ids: List[str] = Field(..., description="List of model IDs")
    error_rates: Dict[str, List[float]] = Field(
        ..., description="Error rates per model per cluster (model_id -> [error rates])"
    )
    total_samples: int = Field(..., description="Total training samples", ge=0)
    samples_per_cluster: List[int] = Field(..., description="Samples per cluster")
    inference_time: Optional[float] = Field(
        None, description="Total inference time in seconds", ge=0.0
    )
    training_time: float = Field(
        ..., description="Total training time in seconds", ge=0.0
    )


__all__ = [
    "ProviderConfig",
    "TrainingResult",
]
