"""Core routing and clustering engine.

This module provides the main components for intelligent model routing:
- ClusterEngine: Semantic clustering using K-means on embeddings
- ModelRouter: Main routing service for selecting optimal models
- Trainer: Training service for router profiles
- ProviderRegistry: Registry of model providers
"""

from .cluster_engine import ClusterEngine
from .provider_registry import ProviderRegistry
from .router import ModelRouter
from .trainer import Trainer

__all__ = [
    "ClusterEngine",
    "ModelRouter",
    "Trainer",
    "ProviderRegistry",
]
