# __init__.py
"""
Pydantic data models for Adaptive Router.

This module provides type-safe data models for API requests/responses,
profile storage, routing decisions, and configuration.

API Models (Request/Response)
-----------------------------
ModelSelectionRequest
    Input for model selection with prompt, cost_bias, and optional model filter.

ModelSelectionResponse
    Output with selected_model and ranked alternatives.

Model
    LLM model definition with provider, name, and pricing.

Alternative
    Alternative model recommendation with cost ratio.

Storage Models (Profiles)
-------------------------
RouterProfile
    Complete router configuration with cluster centers, models, and metadata.

ProfileMetadata
    Profile metadata including n_clusters, embedding_model, and scores.

ClusterCentersData
    K-means cluster centers matrix storage.

ClusteringConfig
    Configuration for K-means clustering algorithm.

RoutingConfig
    Configuration for cost-accuracy trade-off parameters.

MinIOSettings
    Connection settings for MinIO/S3 profile storage.

Routing Models (Internal)
-------------------------
RoutingDecision
    Internal routing decision state.

ModelInfo
    Model metadata for routing calculations.

ModelPricing
    Model pricing information.

ModelFeatures
    Feature vector for model scoring.

Training Models
---------------
ProviderConfig
    Provider configuration for training.

TrainingResult
    Training output with metrics.

Example
-------
Creating a model selection request:

    >>> from adaptive_router.models import ModelSelectionRequest, Model
    >>>
    >>> request = ModelSelectionRequest(
    ...     prompt="Write a sorting algorithm",
    ...     cost_bias=0.5,
    ...     models=[
    ...         Model(
    ...             provider="openai",
    ...             model_name="gpt-4",
    ...             cost_per_1m_input_tokens=30.0,
    ...             cost_per_1m_output_tokens=60.0,
    ...         )
    ...     ]
    ... )

Working with profiles:

    >>> from adaptive_router.models import RouterProfile, ProfileMetadata
    >>>
    >>> profile = RouterProfile.from_file("profile.json")
    >>> print(f"Profile has {profile.metadata.n_clusters} clusters")

See Also
--------
adaptive_router.core : Core routing components
adaptive_router.loaders : Profile loading implementations
"""

from .api import (
    Alternative,
    Model,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from .health import HealthResponse
from .routing import (
    ModelFeatureVector,
    ModelFeatures,
    ModelInfo,
    ModelPricing,
    RoutingDecision,
)
from .train import ProviderConfig, TrainingResult
from .storage import (
    ClusterCentersData,
    ClusteringConfig,
    ClusterStats,
    MinIOSettings,
    ProfileMetadata,
    RouterProfile,
    RoutingConfig,
)

from .config import ModelConfig

__all__ = [
    "Alternative",
    "ClusterCentersData",
    "ClusteringConfig",
    "ClusterStats",
    "HealthResponse",
    "MinIOSettings",
    "Model",
    "ModelConfig",
    "ModelFeatureVector",
    "ModelFeatures",
    "ModelInfo",
    "ModelPricing",
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "ProfileMetadata",
    "ProviderConfig",
    "RouterProfile",
    "RoutingConfig",
    "RoutingDecision",
    "TrainingResult",
]
