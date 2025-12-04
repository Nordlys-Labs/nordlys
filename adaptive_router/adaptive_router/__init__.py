"""Adaptive Router - Intelligent LLM model selection library.

This package provides intelligent model routing using cluster-based selection
with per-cluster error rates, cost optimization, and model capability matching.

Basic Usage:
    >>> from adaptive_router import ModelRouter, ModelSelectionRequest, MinIOSettings
    >>>
    >>> settings = MinIOSettings(
    ...     endpoint_url="https://minio.example.com",
    ...     root_user="admin",
    ...     root_password="password",
    ...     bucket_name="profiles"
    ... )
    >>> router = ModelRouter.from_minio(settings, model_costs)
    >>> request = ModelSelectionRequest(prompt="Write a Python function", cost_bias=0.5)
    >>> response = router.select_model(request)
    >>> print(f"Selected: {response.provider}/{response.model}")

Advanced Usage:
    >>> from adaptive_router import (
    ...     ClusterEngine,
    ...     LocalFileProfileLoader,
    ...     RouterProfile,
    ... )
    >>>
    >>> # Custom profile loading
    >>> loader = LocalFileProfileLoader(profile_path="custom_profile.json")
    >>> profile = loader.load_profile()
    >>> router = ModelRouter.from_profile(profile, model_costs)
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
from .models.storage import MinIOSettings

# ============================================================================
# TIER 2: Configuration & Integration
# ============================================================================

# Model types for routing
from .models.api import Model

# Profile loaders (for custom profile loading)
from .loaders import (
    LocalFileProfileLoader,
    MinIOProfileLoader,
    ProfileLoader,
)

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
    "MinIOSettings",
    # ========================================================================
    # Tier 2: Configuration & Integration
    # ========================================================================
    # Model types
    "Model",
    # Loaders
    "ProfileLoader",
    "LocalFileProfileLoader",
    "MinIOProfileLoader",
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
