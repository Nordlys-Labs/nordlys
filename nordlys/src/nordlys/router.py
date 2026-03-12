"""Runtime Router class.

This module provides runtime-only routing from a precompiled checkpoint.
Training is handled by ``Dataset`` + ``Trainer``.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from nordlys.clustering import (
    ClusterInfo,
    ClusterMetrics,
)
from nordlys.reduction import restore_reducer
from nordlys.reduction.base import ReductionPayload
from nordlys_core import Nordlys as NordlysCore, NordlysCheckpoint


logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 10


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
        >>> from nordlys import Router
        >>>
        >>> # Load from checkpoint file
        >>> router = Router(checkpoint="router.msgpack")
        >>>
        >>> # Route prompts
        >>> result = router.route("Explain quantum computing")
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
        resolved_checkpoint = self._resolve_checkpoint(checkpoint)
        models = [m.model_id for m in resolved_checkpoint.models]
        embedding_model = resolved_checkpoint.embedding.model
        allow_trust_remote_code = resolved_checkpoint.embedding.trust_remote_code

        if embedding_cache_size <= 0:
            raise ValueError("embedding_cache_size must be greater than 0")

        # Validate and store device
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")
        self._device = device

        self._models = models
        self._model_ids = models

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
        core_engine = self._core_engine

        # Compute embedding (with caching for repeated prompts)
        embedding = self.compute_embedding(prompt)

        # Ensure float32 and C-contiguous
        if embedding.dtype != np.float32 or not embedding.flags["C_CONTIGUOUS"]:
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        embedding = np.ascontiguousarray(
            self._reduce_for_routing(embedding.reshape(1, -1))[0], dtype=np.float32
        )

        # Route using C++ core
        response = core_engine.route(embedding, None if models is None else models)

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
        core_engine = self._core_engine

        if not prompts:
            return []

        # Compute embeddings in batch (more efficient for unique texts)
        embeddings = self._compute_embeddings(prompts)

        # Ensure float32 and C-contiguous
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        embeddings = self._reduce_for_routing(embeddings)

        # Route using C++ core engine's route_batch
        responses = core_engine.route_batch(
            embeddings, None if models is None else models
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
        centroids = self._centroids
        model_accuracies = self._model_accuracies
        metrics = self._metrics

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
        centroids = self._centroids

        return [
            self.get_cluster_info(cluster_id) for cluster_id in range(len(centroids))
        ]

    def get_metrics(self) -> ClusterMetrics:
        """Get clustering metrics.

        Returns:
            ClusterMetrics object
        """
        return self._metrics

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
        return self._centroids

    @property
    def metrics_(self) -> ClusterMetrics:
        """Clustering metrics restored from checkpoint."""
        return self._metrics

    @property
    def model_accuracies_(self) -> dict[int, dict[str, float]]:
        """Per-cluster per-model accuracy scores.

        Returns:
            Dict mapping cluster_id -> {model_id: accuracy}
        """
        return self._model_accuracies

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        return len(self._centroids)

    @staticmethod
    def _resolve_checkpoint(
        checkpoint: NordlysCheckpoint | str | Path,
    ) -> NordlysCheckpoint:
        """Resolve checkpoint inputs into an in-memory checkpoint object."""
        if isinstance(checkpoint, NordlysCheckpoint):
            return checkpoint

        path = Path(checkpoint)
        try:
            if path.suffix.lower() == ".msgpack":
                return NordlysCheckpoint.from_msgpack_file(str(path))
            return NordlysCheckpoint.from_json_file(str(path))
        except ValueError as exc:
            logger.error(
                "Failed to load checkpoint",
                extra={"checkpoint_path": str(path), "error": str(exc)},
            )
            raise

    @staticmethod
    def _reduction_payload_from_checkpoint(
        checkpoint: NordlysCheckpoint,
    ) -> ReductionPayload | None:
        """Convert core reduction metadata into the validated Python payload type."""
        reduction = checkpoint.reduction
        if reduction is None:
            return None

        return ReductionPayload(
            kind=reduction.kind,
            config=json.loads(reduction.config_json),
            state=json.loads(reduction.state_json),
        )

    def _load_checkpoint_state(self, checkpoint: NordlysCheckpoint) -> None:
        """Populate runtime fields from checkpoint data."""
        self._nr_clusters = checkpoint.clustering.n_clusters
        self._random_state = checkpoint.clustering.random_state
        self._reducer = restore_reducer(
            self._reduction_payload_from_checkpoint(checkpoint)
        )
        self._core_engine = NordlysCore.from_checkpoint(checkpoint, device=self._device)
        self._centroids = np.asarray(checkpoint.cluster_centers, dtype=np.float32)
        self._model_accuracies = {
            cluster_id: {
                model.model_id: model.scores[cluster_id]  # type: ignore[attr-defined]
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

    def _reduce_for_routing(self, embeddings: np.ndarray) -> np.ndarray:
        reducer = self._reducer
        if reducer is None:
            return embeddings

        reduced = reducer.transform(embeddings)
        if reduced.dtype != np.float32 or not reduced.flags["C_CONTIGUOUS"]:
            reduced = np.ascontiguousarray(reduced, dtype=np.float32)

        expected_width = self._centroids.shape[1]
        actual_width = reduced.shape[1]
        if actual_width != expected_width:
            raise ValueError(
                "_reduce_for_routing produced embeddings with width "
                f"{actual_width}, expected {expected_width} from centroids "
                f"{self._centroids.shape}; reducer={type(reducer).__name__}"
            )
        return reduced

    def __repr__(self) -> str:
        return (
            f"Router(models={len(self._models)}, "
            f"nr_clusters={self._nr_clusters}, "
            "status=ready)"
        )
