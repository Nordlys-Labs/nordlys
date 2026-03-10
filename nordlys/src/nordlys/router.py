"""Runtime Router class.

This module provides runtime-only routing from a precompiled checkpoint.
Training is handled by ``Dataset`` + ``Trainer``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
from cachetools import LRUCache
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nordlys.clustering import (
    ClusterInfo,
    ClusterMetrics,
)
from nordlys_core import Nordlys as NordlysCore, NordlysCheckpoint


logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 10


class ModelConfig(BaseModel):
    """Model configuration with costs for Router router.

    This is a simplified version for the new Router API.
    For the legacy API, use nordlys.models.config.ModelConfig.

    Attributes:
        id: Model identifier in "provider/model_name" format (e.g., "openai/gpt-4")
        cost_input: Cost per 1M input tokens
        cost_output: Cost per 1M output tokens
    """

    id: str = Field(..., min_length=1, description="Model ID (e.g., 'openai/gpt-4')")
    cost_input: float = Field(..., ge=0, description="Cost per 1M input tokens")
    cost_output: float = Field(..., ge=0, description="Cost per 1M output tokens")

    @property
    def cost_average(self) -> float:
        """Average cost per 1M tokens."""
        return (self.cost_input + self.cost_output) / 2

    @property
    def provider(self) -> str:
        """Extract provider from model ID."""
        provider, separator, _ = self.id.partition("/")
        # If no slash found, separator is empty, so return empty string
        return provider if separator else ""

    @property
    def model_name(self) -> str:
        """Extract model name from model ID."""
        _, _, model_name = self.id.partition("/")
        return model_name if model_name else self.id

    model_config = {"frozen": True}


@dataclass
class RouteResult:
    """Result of routing a prompt to a model.

    Attributes:
        model_id: Selected model identifier
        cluster_id: Assigned cluster ID
        cluster_distance: Distance to cluster centroid
        alternatives: Ranked list of alternative model IDs (best to worst)
    """

    model_id: str
    cluster_id: int
    cluster_distance: float
    alternatives: list[str] = field(default_factory=list)


class Router:
    """Unified runtime model routing from checkpoints.

    Router selects models using checkpoint centroids and per-cluster error rates.
    Use ``Trainer.fit(...)`` to create checkpoints; this class does not train.

    Example:
        >>> from nordlys import Router, ModelConfig
        >>> import pandas as pd
        >>>
        >>> # Define models with costs
        >>> models = [
        ...     ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
        ...     ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
        ... ]
        >>>
        >>> # Training data: DataFrame with "questions" + model accuracy columns
        >>> df = pd.DataFrame({
        ...     "questions": ["What is ML?", "Write code", "Explain databases"],
        ...     "openai/gpt-4": [0.92, 0.85, 0.88],
        ...     "anthropic/claude-3-sonnet": [0.88, 0.91, 0.85],
        ... })
        >>>
        >>> # Load from checkpoint file
        >>> model = Router(checkpoint="router.msgpack")
        >>>
         >>> # Route prompts
         >>> result = model.route("Explain quantum computing")
         >>> print(f"Selected: {result.model_id}")
    """

    def __init__(
        self,
        checkpoint: NordlysCheckpoint | str | Path,
        embedding_cache_size: int = 1000,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """Initialize Router router.

        Args:
            checkpoint: Checkpoint input (in-memory object or file path)
            embedding_cache_size: Maximum number of embeddings to cache (must be > 0)
            device: Device for C++ core clustering operations ("cpu" or "cuda")
        """
        # C++ core (initialized on load or after fit) - set early to avoid __del__ errors
        self._core_engine: NordlysCore | None = None
        resolved_checkpoint: NordlysCheckpoint
        if isinstance(checkpoint, NordlysCheckpoint):
            resolved_checkpoint = checkpoint
        else:
            path = Path(checkpoint)
            if path.suffix.lower() == ".msgpack":
                resolved_checkpoint = NordlysCheckpoint.from_msgpack_file(str(path))
            else:
                resolved_checkpoint = NordlysCheckpoint.from_json_file(str(path))
        models = [
            ModelConfig(
                id=m.model_id,
                cost_input=m.cost_per_1m_input_tokens,
                cost_output=m.cost_per_1m_output_tokens,
            )
            for m in resolved_checkpoint.models
        ]
        embedding_model = resolved_checkpoint.embedding.model
        allow_trust_remote_code = resolved_checkpoint.embedding.trust_remote_code

        if embedding_cache_size <= 0:
            raise ValueError("embedding_cache_size must be greater than 0")

        # Validate and store device
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")
        self._device = device

        self._models = models
        self._model_ids = [m.id for m in models]

        # Embedding model - loaded at initialization
        self._embedding_model_name = embedding_model
        self._allow_trust_remote_code = allow_trust_remote_code
        self._embedding_model: SentenceTransformer
        logger.info(
            f"Loading embedding model '{embedding_model}' on device: {self._device}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            self._embedding_model = SentenceTransformer(
                embedding_model,
                device=self._device,
                trust_remote_code=allow_trust_remote_code,
            )
        self._embedding_model.tokenizer.clean_up_tokenization_spaces = False

        # Embedding cache - LRU cache for computed embeddings
        self._embedding_cache_size = embedding_cache_size
        self._embedding_cache: LRUCache[str, np.ndarray] = LRUCache(
            maxsize=embedding_cache_size
        )

        # Checkpoint-derived metadata (populated on load)
        self._nr_clusters = 0
        self._random_state = 0

        # Runtime state (populated on load/from-checkpoint)
        self._centroids: np.ndarray | None = None
        self._metrics: ClusterMetrics | None = None
        self._model_accuracies: dict[int, dict[str, float]] | None = None

        self._checkpoint = resolved_checkpoint
        self._load_checkpoint_state(resolved_checkpoint)

    def _compute_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        """Compute embeddings for texts in batch with caching support.

        Checks cache first for each text, then computes only cache misses in batch.
        This combines the efficiency of batch processing with cache benefits.
        """
        if not texts:
            return np.array([])

        # Fast path: check if all texts are cache misses (common case during fit)
        # Quick check without building full structures
        if not any(text in self._embedding_cache for text in texts):
            # All cache misses - fast path
            # Convert Sequence to list for encode() which expects list[str]
            texts_list = list(texts)
            embeddings = self._embedding_model.encode(
                texts_list,
                convert_to_numpy=True,
                show_progress_bar=False,  # Disable progress bar for internal calls
            )
            # Batch update cache
            self._embedding_cache.update(zip(texts, embeddings))
            return embeddings

        # Mixed cache hits/misses - optimized single-pass approach
        cached_indices_set = set()
        cached_data = {}  # index -> embedding mapping
        texts_to_compute = []

        # Single pass to separate cache hits and misses
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                cached_indices_set.add(i)
                cached_data[i] = self._embedding_cache[text]
            else:
                texts_to_compute.append(text)

        # Compute embeddings for cache misses in batch
        if texts_to_compute:
            new_embeddings = self._embedding_model.encode(
                texts_to_compute,
                convert_to_numpy=True,
                show_progress_bar=False,  # Disable progress bar for internal calls
            )

            # Batch update cache
            self._embedding_cache.update(zip(texts_to_compute, new_embeddings))
        else:
            new_embeddings = np.array([])

        # Pre-allocate result array for better performance
        # Get embedding dimension from first available embedding
        sample_embedding = (
            cached_data[next(iter(cached_data))] if cached_data else new_embeddings[0]
        )
        embedding_dim = sample_embedding.shape[0]
        result = np.empty((len(texts), embedding_dim), dtype=sample_embedding.dtype)

        # Fill result array in correct order
        compute_idx = 0
        for i in range(len(texts)):
            if i in cached_indices_set:
                result[i] = cached_data[i]
            else:
                result[i] = new_embeddings[compute_idx]
                compute_idx += 1

        return result

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text with LRU caching.

        Caches embeddings to avoid recomputation for repeated prompts.

        Note: This method is NOT thread-safe. For multi-threaded use,
        add external synchronization.

        Args:
            text: The text to compute embedding for.

        Returns:
            The embedding vector as a numpy array.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Cache miss: compute embedding
        embedding: np.ndarray = self._embedding_model.encode(
            [text], convert_to_numpy=True
        )[0]

        self._embedding_cache[text] = embedding

        return embedding

    def route(
        self,
        prompt: str,
        models: list[str] | None = None,
    ) -> RouteResult:
        """Route a prompt to the best model using C++ core engine.

        Args:
            prompt: The text prompt to route
            models: Optional list of model IDs to filter

        Returns:
            RouteResult with selected model and alternatives
        """
        core_engine = cast(NordlysCore, self._core_engine)

        # Compute embedding (with caching for repeated prompts)
        embedding = self.compute_embedding(prompt)

        # Ensure float32 and C-contiguous
        if embedding.dtype != np.float32 or not embedding.flags["C_CONTIGUOUS"]:
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        # Route using C++ core
        response = core_engine.route(embedding, [] if models is None else models)

        return RouteResult(
            model_id=response.selected_model,
            cluster_id=response.cluster_id,
            cluster_distance=float(response.cluster_distance),
            alternatives=list(response.alternatives),
        )

    def route_batch(
        self,
        prompts: Sequence[str],
        models: list[str] | None = None,
    ) -> list[RouteResult]:
        """Route multiple prompts in batch using core engine's route_batch.

        Args:
            prompts: List of text prompts
            models: Optional list of model IDs to filter

        Returns:
            List of RouteResults
        """
        core_engine = cast(NordlysCore, self._core_engine)

        if not prompts:
            return []

        # Compute embeddings in batch (more efficient for unique texts)
        embeddings = self._compute_embeddings(prompts)

        # Ensure float32 and C-contiguous
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Route using C++ core engine's route_batch
        responses = core_engine.route_batch(
            embeddings, [] if models is None else models
        )

        # Convert responses to RouteResult objects
        return [
            RouteResult(
                model_id=response.selected_model,
                cluster_id=response.cluster_id,
                cluster_distance=float(response.cluster_distance),
                alternatives=list(response.alternatives),
            )
            for response in responses
        ]

    # =========================================================================
    # Introspection methods
    # =========================================================================

    def get_cluster_info(self, cluster_id: int) -> ClusterInfo:
        """Get information about a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            ClusterInfo with cluster details
        """
        centroids = cast(np.ndarray, self._centroids)
        model_accuracies = cast(dict[int, dict[str, float]], self._model_accuracies)
        metrics = cast(ClusterMetrics, self._metrics)

        if cluster_id < 0 or cluster_id >= len(centroids):
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        size = 0
        if metrics.cluster_sizes is not None and cluster_id < len(
            metrics.cluster_sizes
        ):
            size = int(metrics.cluster_sizes[cluster_id])

        return ClusterInfo(
            cluster_id=cluster_id,
            size=size,
            centroid=centroids[cluster_id],
            model_accuracies=model_accuracies.get(cluster_id, {}),
        )

    def get_clusters(self) -> list[ClusterInfo]:
        """Get information about all clusters.

        Returns:
            List of ClusterInfo objects
        """
        centroids = cast(np.ndarray, self._centroids)

        return [
            self.get_cluster_info(cluster_id) for cluster_id in range(len(centroids))
        ]

    def get_metrics(self) -> ClusterMetrics:
        """Get clustering metrics.

        Returns:
            ClusterMetrics object
        """
        return cast(ClusterMetrics, self._metrics)

    # =========================================================================
    # Embedding cache management
    # =========================================================================

    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def embedding_cache_info(self) -> dict[str, int]:
        """Get embedding cache info.

        Returns:
            Dictionary with size and maxsize.
        """
        return {
            "size": len(self._embedding_cache),
            "maxsize": self._embedding_cache_size,
        }

    # =========================================================================
    # Fitted attributes (sklearn convention: trailing underscore)
    # =========================================================================

    @property
    def centroids_(self) -> np.ndarray:
        """Cluster centroids of shape (n_clusters, n_features)."""
        return cast(np.ndarray, self._centroids)

    @property
    def metrics_(self) -> ClusterMetrics:
        """Clustering metrics restored from checkpoint."""
        return cast(ClusterMetrics, self._metrics)

    @property
    def model_accuracies_(self) -> dict[int, dict[str, float]]:
        """Per-cluster per-model accuracy scores.

        Returns:
            Dict mapping cluster_id -> {model_id: accuracy}
        """
        return cast(dict[int, dict[str, float]], self._model_accuracies)

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        return len(cast(np.ndarray, self._centroids))

    def _load_checkpoint_state(self, checkpoint: NordlysCheckpoint) -> None:
        """Populate runtime fields from checkpoint data."""
        self._nr_clusters = checkpoint.clustering.n_clusters
        self._random_state = checkpoint.clustering.random_state

        self._core_engine = NordlysCore.from_checkpoint(checkpoint, device=self._device)
        self._centroids = np.asarray(checkpoint.cluster_centers, dtype=np.float32)

        self._model_accuracies = {
            cluster_id: {
                model.model_id: 1.0 - model.error_rates[cluster_id]
                for model in checkpoint.models
            }
            for cluster_id in range(checkpoint.clustering.n_clusters)
        }

        self._metrics = ClusterMetrics(
            silhouette_score=checkpoint.metrics.silhouette_score,
            n_clusters=checkpoint.clustering.n_clusters,
            n_samples=checkpoint.metrics.n_samples,
            cluster_sizes=checkpoint.metrics.cluster_sizes,
            inertia=checkpoint.metrics.inertia,
        )

        logger.info("Loaded checkpoint with C++ core (float32)")

    def __repr__(self) -> str:
        return (
            f"Router(models={len(self._models)}, "
            f"nr_clusters={self._nr_clusters}, "
            "status=ready)"
        )
