"""
Core ML routing components for Adaptive Router.

This module provides the main routing logic for intelligent LLM model selection
using cluster-based routing with per-cluster error rates.

Classes
-------
ModelRouter
    Main routing class with ``select_model()`` and ``route()`` methods.
    Supports local file and MinIO profile loading.

ClusterEngine
    K-means cluster assignment engine.
    Assigns prompts to clusters using pre-trained centroids.

ProviderRegistry
    Registry for LLM provider configurations.
    Manages model metadata and provider information.

Exceptions
----------
AdaptiveRouterError
    Base exception for all adaptive router errors.

ClusterNotFittedError
    Raised when cluster assignment is attempted before fitting.

FeatureExtractionError
    Raised when feature extraction fails (embedding or TF-IDF).

InvalidModelFormatError
    Raised when model ID format is invalid (expected 'provider/model_name').

ModelNotFoundError
    Raised when requested model is not found in profile.

ProfileLoadError
    Raised when profile loading fails (file not found, invalid format).

Example
-------
Basic routing with ModelRouter:

    >>> from adaptive_router.core import ModelRouter
    >>> from adaptive_router.models import ModelSelectionRequest
    >>>
    >>> router = ModelRouter.from_local_file("profile.json")
    >>> response = router.select_model(
    ...     ModelSelectionRequest(prompt="Explain quantum computing", cost_bias=0.5)
    ... )
    >>> print(response.model_id)
    'openai/gpt-4'

Quick routing (returns model ID string only):

    >>> model_id = router.route("Write a sorting algorithm", cost_bias=0.3)
    >>> print(model_id)
    'openai/gpt-3.5-turbo'

See Also
--------
adaptive_router.loaders : Profile loading implementations
adaptive_router.models : Pydantic data models
adaptive_router_core : High-performance C++ inference core (optional)
"""

from .router import ModelRouter
from .cluster_engine import ClusterEngine
from .provider_registry import ProviderRegistry, default_registry
from ..exceptions.core import (
    AdaptiveRouterError,
    ClusterNotFittedError,
    FeatureExtractionError,
    InvalidModelFormatError,
    ModelNotFoundError,
    ProfileLoadError,
)

__all__ = [
    "ModelRouter",
    "ClusterEngine",
    "ProviderRegistry",
    "default_registry",
    "AdaptiveRouterError",
    "ClusterNotFittedError",
    "FeatureExtractionError",
    "InvalidModelFormatError",
    "ModelNotFoundError",
    "ProfileLoadError",
]
