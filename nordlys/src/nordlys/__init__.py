"""Nordlys - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster error rates, cost optimization, and model capability matching.

Usage:
    >>> from nordlys import Nordlys, ModelConfig
    >>> import pandas as pd
    >>>
    >>> # Define models
    >>> models = [
    ...     ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ...     ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ... ]
    >>>
    >>> # Training data
    >>> df = pd.DataFrame({
    ...     "questions": ["What is ML?", "Write code", ...],
    ...     "openai/gpt-4": [0.92, 0.85, ...],
    ...     "anthropic/claude-3-sonnet": [0.88, 0.91, ...],
    ... })
    >>>
    >>> # Fit and route
    >>> model = Nordlys(models=models)
    >>> model.fit(df)
    >>> result = model.route("Explain quantum computing", cost_bias=0.5)
    >>> print(f"Selected: {result.model_id}")
"""

# ============================================================================
# Main API
# ============================================================================

from nordlys.nordlys import Nordlys, ModelConfig, RouteResult

# Reduction components
from nordlys import reduction

# Clustering components
from nordlys import clustering

# C++ Core types
from nordlys_core_ext import (
    NordlysCheckpoint,
    TrainingMetrics,
    EmbeddingConfig,
    ClusteringConfig,
    RoutingConfig,
    ModelFeatures,
)

# ============================================================================
# Package metadata
# ============================================================================

__version__ = "0.1.4"

__all__ = [
    # Main API
    "Nordlys",
    "ModelConfig",
    "RouteResult",
    # C++ Core types
    "NordlysCheckpoint",
    "TrainingMetrics",
    "EmbeddingConfig",
    "ClusteringConfig",
    "RoutingConfig",
    "ModelFeatures",
    # Modules
    "reduction",
    "clustering",
]
