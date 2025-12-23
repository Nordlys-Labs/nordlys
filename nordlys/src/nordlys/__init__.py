"""Nordlys - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster error rates, cost optimization, and model capability matching.

Basic Usage:
    >>> from nordlys import ModelRouter, ModelSelectionRequest
    >>>
    >>> # Load from profile (auto-detects JSON or MessagePack)
    >>> router = ModelRouter.from_file("router_profile.json")
    >>> request = ModelSelectionRequest(prompt="Write a Python function", cost_bias=0.5)
    >>> response = router.select_model(request)
    >>> print(f"Selected: {response.model_id}")
    >>> print(f"Available models: {[m.unique_id() for m in router.profile.models]}")

Advanced Usage:
    >>> from nordlys import Trainer, RouterProfile
    >>>
    >>> # Train a new profile
    >>> trainer = Trainer()
    >>> trainer.train_from_polars(df, models)
    >>> trainer.save_profile("my_profile.json")
"""

# ============================================================================
# TIER 1: Essential API (90% of users)
# ============================================================================

# Core services
from .core.router import ModelRouter
from .core.trainer import Trainer

# Request/Response models
from .models.api import (
    Alternative,
    ModelSelectionRequest,
    ModelSelectionResponse,
)

# Training models
from .models.train import ProviderConfig, TrainingResult

# Storage configuration (needed for initialization)

# ============================================================================
# TIER 2: Configuration & Integration
# ============================================================================

# Model types for routing
from .models.api import Model

# Storage types (profile structure)
from .models.storage import (
    ClusterCentersData,
    ProfileMetadata,
    RouterProfile,
)

# Configuration types (YAML and routing config)
from .models.config import (
    ModelConfig,
)

# ============================================================================
# TIER 3: Advanced API (Scripts, testing, custom implementations)
# ============================================================================

# Core ML components
from .core import (
    ClusterEngine,
)

# Routing internals and public types
from .models.routing import (
    ModelFeatureVector,
    ModelFeatures,
    ModelInfo,
    ModelPricing,
    RoutingDecision,
)

# Health check
from .models.health import HealthResponse

# ============================================================================
# Package metadata
# ============================================================================

__version__ = "0.1.0"

__all__ = [
    # ========================================================================
    # Tier 1: Essential API
    # ========================================================================
    "ModelRouter",
    "Trainer",
    "ModelSelectionRequest",
    "ModelSelectionResponse",
    "Alternative",
    "ProviderConfig",
    "TrainingResult",
    # ========================================================================
    # Tier 2: Configuration & Integration
    # ========================================================================
    # Model types
    "Model",
    # Storage types
    "RouterProfile",
    "ProfileMetadata",
    "ClusterCentersData",
    # Configuration
    "ModelConfig",
    # ========================================================================
    # Tier 2.5: Public Routing Types (clean API)
    # ========================================================================
    "ModelInfo",
    "ModelPricing",
    # ========================================================================
    # Tier 3: Advanced API
    # ========================================================================
    # Core components
    "ClusterEngine",
    # Routing internals
    "ModelFeatureVector",
    "ModelFeatures",
    "RoutingDecision",
    # Health
    "HealthResponse",
]
