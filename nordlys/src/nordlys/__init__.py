"""Router - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster scores.

Usage:
    >>> from nordlys import Dataset, Trainer, Router
    >>>
    >>> # Define models (just model IDs)
    >>> models = ["openai/gpt-4", "anthropic/claude-3-sonnet"]
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
from nordlys.router import Router, RouteResult
from nordlys.trainer import (
    EvaluationReport,
    FittedStructure,
    RoutingPolicy,
    Trainer,
)

# Reduction components
from nordlys import reduction

# Clustering components
from nordlys import clustering

# Search components
from nordlys import search

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

__version__ = "0.7.0"

__all__ = [
    # Main API
    "Dataset",
    "Trainer",
    "Router",
    "RouteResult",
    # Trainer types
    "FittedStructure",
    "RoutingPolicy",
    "EvaluationReport",
    # C++ Core types
    "NordlysCheckpoint",
    "TrainingMetrics",
    "EmbeddingConfig",
    "ClusteringConfig",
    "ModelFeatures",
    # Modules
    "reduction",
    "clustering",
    "search",
    "embeddings",
]
