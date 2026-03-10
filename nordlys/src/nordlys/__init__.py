"""Router - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster error rates, cost optimization, and model capability matching.

Usage:
    >>> from nordlys import Dataset, Trainer, Router, ModelConfig
    >>>
    >>> # Define models
    >>> models = [
    ...     ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ...     ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ... ]
    >>>
    >>> dataset = Dataset.from_list([
    ...     {"id": "1", "input": "What is ML?", "targets": {"openai/gpt-4": 1, "anthropic/claude-3-sonnet": 0}},
    ... ])
    >>> checkpoint = Trainer(models=models).fit(dataset)
    >>> router = Router(checkpoint=checkpoint)
    >>> result = router.route("Explain quantum computing")
    >>> print(f"Selected: {result.model_id}")
"""

# ============================================================================
# Main API
# ============================================================================

from nordlys.dataset import Dataset
from nordlys.router import Router, ModelConfig, RouteResult
from nordlys.trainer import Trainer

# Reduction components
from nordlys import reduction

# Clustering components
from nordlys import clustering

# Embedding components
from nordlys import embeddings

# C++ Core types
from nordlys_core import (
    NordlysCheckpoint,
    TrainingMetrics,
    EmbeddingConfig,
    ClusteringConfig,
    ModelFeatures,
)

# ============================================================================
# Package metadata
# ============================================================================

__version__ = "0.2.4"

__all__ = [
    # Main API
    "Dataset",
    "Trainer",
    "Router",
    "ModelConfig",
    "RouteResult",
    # C++ Core types
    "NordlysCheckpoint",
    "TrainingMetrics",
    "EmbeddingConfig",
    "ClusteringConfig",
    "ModelFeatures",
    # Modules
    "reduction",
    "clustering",
    "embeddings",
]
