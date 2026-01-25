"""Type stubs for the _core native module."""

from typing import Optional
from collections.abc import Sequence

# Type definitions
class TrainingMetrics:
    """Training metrics (optional fields)."""
    
    @property
    def n_samples(self) -> Optional[int]: ...
    """Number of training samples (None if not available)."""
    
    @property
    def cluster_sizes(self) -> Optional[list[int]]: ...
    """Cluster sizes (None if not available)."""
    
    @property
    def silhouette_score(self) -> Optional[float]: ...
    """Silhouette score (None if not available)."""
    
    @property
    def inertia(self) -> Optional[float]: ...
    """Inertia (None if not available)."""


class EmbeddingConfig:
    """Embedding configuration."""
    
    model: str
    """Embedding model ID."""
    
    dtype: str
    """Data type ('float32' or 'float64')."""
    
    trust_remote_code: bool
    """Whether to trust remote code."""


class ClusteringConfig:
    """Clustering configuration parameters."""
    
    n_clusters: int
    """Number of clusters."""
    
    random_state: int
    """Random state for reproducibility."""
    
    max_iter: int
    """Maximum iterations."""
    
    n_init: int
    """Number of initializations."""
    
    algorithm: str
    """Clustering algorithm."""
    
    normalization: str
    """Normalization strategy."""


class ModelFeatures:
    """Model configuration with error rates."""
    
    model_id: str
    """Full model ID (e.g., 'openai/gpt-4')."""
    
    error_rates: list[float]
    """Per-cluster error rates."""
    
    cost_per_1m_input_tokens: float
    """Cost per 1M input tokens."""
    
    cost_per_1m_output_tokens: float
    """Cost per 1M output tokens."""
    
    def provider(self) -> str: ...
    """Extract provider from model_id."""
    
    def model_name(self) -> str: ...
    """Extract model name from model_id."""
    
    def cost_per_1m_tokens(self) -> float: ...
    """Average cost per 1M tokens."""


class RouteResult32:
    """Routing result for float32 precision."""
    
    selected_model: str
    """Selected model ID."""
    
    alternatives: list[str]
    """List of alternative model IDs."""
    
    cluster_id: int
    """Assigned cluster ID."""
    
    cluster_distance: float
    """Distance to cluster center."""
    
    def __repr__(self) -> str: ...


class RouteResult64:
    """Routing result for float64 precision."""
    
    selected_model: str
    """Selected model ID."""
    
    alternatives: list[str]
    """List of alternative model IDs."""
    
    cluster_id: int
    """Assigned cluster ID."""
    
    cluster_distance: float
    """Distance to cluster center."""
    
    def __repr__(self) -> str: ...


class NordlysCheckpoint:
    """Serialized Nordlys model checkpoint containing cluster centers and model metadata."""
    
    version: str
    """Checkpoint format version."""
    
    models: list[ModelFeatures]
    """List of model configurations."""
    
    embedding: EmbeddingConfig
    """Embedding configuration."""
    
    clustering: ClusteringConfig
    """Clustering configuration."""
    
    metrics: TrainingMetrics
    """Training metrics (optional fields)."""
    
    @staticmethod
    def from_json_file(path: str) -> "NordlysCheckpoint": ...
    """Load checkpoint from JSON file."""
    
    @staticmethod
    def from_json_string(json_str: str) -> "NordlysCheckpoint": ...
    """Load checkpoint from JSON string."""
    
    @staticmethod
    def from_msgpack_file(path: str) -> "NordlysCheckpoint": ...
    """Load checkpoint from MessagePack file."""
    
    @staticmethod
    def from_msgpack_bytes(data: bytes) -> "NordlysCheckpoint": ...
    """Load checkpoint from MessagePack bytes."""
    
    def to_json_string(self) -> str: ...
    """Serialize checkpoint to JSON string."""
    
    def to_json_file(self, path: str) -> None: ...
    """Write checkpoint to JSON file."""
    
    def to_msgpack_bytes(self) -> bytes: ...
    """Serialize checkpoint to MessagePack bytes."""
    
    def to_msgpack_file(self, path: str) -> None: ...
    """Write checkpoint to MessagePack file."""
    
    def validate(self) -> None: ...
    """Validate checkpoint data integrity."""
    
    @property
    def cluster_centers(self): ...
    """Cluster centers as numpy array."""
    
    @property
    def n_clusters(self) -> int: ...
    """Number of clusters (computed)."""
    
    @property
    def feature_dim(self) -> int: ...
    """Feature dimensionality (computed)."""
    
    @property
    def dtype(self) -> str: ...
    """Data type ('float32' or 'float64')."""
    
    @property
    def embedding_model(self) -> str: ...
    """Embedding model ID."""
    
    @property
    def random_state(self) -> int: ...
    """Random state."""
    
    @property
    def allow_trust_remote_code(self) -> bool: ...
    """Trust remote code flag."""
    
    @property
    def silhouette_score(self) -> float: ...
    """Silhouette score (-1.0 if not available)."""


class Nordlys32:
    """High-performance routing engine with float32 precision."""
    
    @staticmethod
    def from_checkpoint(checkpoint: NordlysCheckpoint, device: str = "cpu") -> "Nordlys32": ...
    """Load engine from checkpoint."""
    
    def route(
        self,
        embedding: "numpy.ndarray[numpy.float32]",
        models: Optional[list[str]] = None
    ) -> RouteResult32: ...
    """Route an embedding to the best model."""
    
    def route_batch(
        self,
        embeddings: "numpy.ndarray[numpy.float32]",
        models: Optional[list[str]] = None
    ) -> list[RouteResult32]: ...
    """Batch route multiple embeddings."""
    
    def get_supported_models(self) -> list[str]: ...
    """Get list of all supported model IDs."""
    
    @property
    def n_clusters(self) -> int: ...
    """Number of clusters in the model."""
    
    @property
    def embedding_dim(self) -> int: ...
    """Expected embedding dimensionality."""
    
    @property
    def dtype(self) -> str: ...
    """Data type of the engine."""


class Nordlys64:
    """High-performance routing engine with float64 precision."""
    
    @staticmethod
    def from_checkpoint(checkpoint: NordlysCheckpoint, device: str = "cpu") -> "Nordlys64": ...
    """Load engine from checkpoint."""
    
    def route(
        self,
        embedding: "numpy.ndarray[numpy.float64]",
        models: Optional[list[str]] = None
    ) -> RouteResult64: ...
    """Route an embedding to the best model."""
    
    def route_batch(
        self,
        embeddings: "numpy.ndarray[numpy.float64]",
        models: Optional[list[str]] = None
    ) -> list[RouteResult64]: ...
    """Batch route multiple embeddings."""
    
    def get_supported_models(self) -> list[str]: ...
    """Get list of all supported model IDs."""
    
    @property
    def n_clusters(self) -> int: ...
    """Number of clusters in the model."""
    
    @property
    def embedding_dim(self) -> int: ...
    """Expected embedding dimensionality."""
    
    @property
    def dtype(self) -> str: ...
    """Data type of the engine."""


def load_checkpoint(path: str) -> NordlysCheckpoint: ...
"""Load checkpoint from file (auto-detects format)."""

__version__: str
