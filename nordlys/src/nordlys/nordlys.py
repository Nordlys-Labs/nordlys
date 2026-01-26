"""Main Nordlys orchestrator class with sklearn-like API.

This module provides the unified Nordlys class that orchestrates
clustering, routing, and model selection with a simple BERTopic-style API.
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
import pandas as pd
from cachetools import LRUCache
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from nordlys.clustering import (
    ClusterInfo,
    ClusterMetrics,
    Clusterer,
    KMeansClusterer,
    compute_cluster_metrics,
)
from nordlys.reduction import Reducer
from nordlys_core import Nordlys as NordlysCore, NordlysCheckpoint


logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ITER = 300
DEFAULT_N_INIT = 10


class ModelConfig(BaseModel):
    """Model configuration with costs for Nordlys router.

    This is a simplified version for the new Nordlys API.
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


def _get_device() -> str:
    """Get the best available device for embedding computation."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Nordlys:
    """Unified model routing with sklearn-like API.

    Nordlys provides intelligent model selection based on prompt clustering.
    It follows BERTopic's design patterns: dependency injection, sensible defaults,
    and sklearn-compatible fit/transform/predict methods.

    Example:
        >>> from nordlys import Nordlys, ModelConfig
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
        >>> # Create and fit
        >>> model = Nordlys(models=models, nr_clusters=10)
        >>> model.fit(df)
        >>>
         >>> # Route prompts
         >>> result = model.route("Explain quantum computing")
         >>> print(f"Selected: {result.model_id}")
    """

    def __init__(
        self,
        models: list[ModelConfig],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        umap_model: Reducer | None = None,
        cluster_model: Clusterer | None = None,
        nr_clusters: int = 20,
        random_state: int = 42,
        allow_trust_remote_code: bool = False,
        embedding_cache_size: int = 1000,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """Initialize Nordlys router.

        Args:
            models: List of model configurations with costs
            embedding_model: Hugging Face model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            umap_model: Optional dimensionality reducer (e.g., UMAPReducer, PCAReducer)
            cluster_model: Clustering algorithm (default: KMeansClusterer)
            nr_clusters: Number of clusters (used if cluster_model is None)
            random_state: Random seed for reproducibility
            allow_trust_remote_code: Allow remote code execution for embedding model
            embedding_cache_size: Maximum number of embeddings to cache (must be > 0)
            device: Device for C++ core clustering operations ("cpu" or "cuda")
        """
        # C++ core (initialized on load or after fit) - set early to avoid __del__ errors
        self._core_engine: NordlysCore | None = None

        if not models:
            raise ValueError("At least one model configuration is required")

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

        # Reducer (optional)
        self._reducer = umap_model

        # Clusterer
        self._nr_clusters = nr_clusters
        self._random_state = random_state
        if cluster_model is not None:
            self._clusterer = cluster_model
        else:
            self._clusterer = KMeansClusterer(
                n_clusters=nr_clusters,
                random_state=random_state,
            )

        # Fitted state (None until fit() is called)
        self._embeddings: np.ndarray | None = None
        self._reduced_embeddings: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._metrics: ClusterMetrics | None = None
        self._model_accuracies: dict[int, dict[str, float]] | None = None
        self._is_fitted = False

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

    def fit(self, df: pd.DataFrame, questions_col: str = "questions") -> "Nordlys":
        """Fit the router on training data.

        Args:
            df: DataFrame with questions column and model accuracy columns.
                Model accuracy columns should be named with model IDs (e.g., "openai/gpt-4")
                and contain values in [0, 1] where 1 = correct, 0 = incorrect.
            questions_col: Name of the column containing questions/prompts

        Returns:
            Self

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Fitting Nordlys on {len(df)} samples")

        # Validate DataFrame
        if questions_col not in df.columns:
            raise ValueError(f"DataFrame must have a '{questions_col}' column")

        # Check for model accuracy columns
        missing_models = set(self._model_ids) - set(df.columns)
        if missing_models:
            raise ValueError(
                f"DataFrame missing accuracy columns for models: {missing_models}. "
                f"Expected columns: {self._model_ids}"
            )

        # Extract questions
        questions = df[questions_col].tolist()

        # Step 1: Compute embeddings
        logger.info("Computing embeddings...")
        self._embeddings = self._compute_embeddings(questions)

        # Step 2: Apply reducer if provided
        if self._reducer is not None:
            logger.info("Applying dimensionality reduction...")
            self._reduced_embeddings = self._reducer.fit_transform(self._embeddings)
            clustering_input = self._reduced_embeddings
        else:
            self._reduced_embeddings = None
            clustering_input = self._embeddings

        # Step 3: Cluster
        logger.info(f"Clustering with {self._clusterer}...")
        self._clusterer.fit(clustering_input)
        self._labels = self._clusterer.labels_
        self._centroids = self._clusterer.cluster_centers_

        # Step 4: Compute metrics
        inertia = getattr(self._clusterer, "inertia_", None)
        self._metrics = compute_cluster_metrics(clustering_input, self._labels, inertia)
        logger.info(f"Clustering complete: {self._metrics}")

        # Step 5: Compute per-cluster accuracy for each model
        self._model_accuracies = {
            cluster_id: {
                model_id: float(df.loc[mask, model_id].mean())
                for model_id in self._model_ids
            }
            for cluster_id in range(self._clusterer.n_clusters_)
            if (mask := self._labels == cluster_id).any()
        }

        self._is_fitted = True

        # Step 6: Initialize C++ core engine - required for routing
        logger.info("Initializing C++ core engine...")
        checkpoint = self._to_checkpoint()
        try:
            self._core_engine = NordlysCore.from_checkpoint(
                checkpoint, device=self._device
            )
        except (ValueError, RuntimeError, AttributeError, OSError) as e:
            raise RuntimeError(
                f"Failed to initialize C++ core engine: {e}. "
                "This may indicate invalid checkpoint data or a compatibility issue."
            ) from e

        logger.info("Nordlys fitting complete")
        return self

    def fit_transform(
        self, df: pd.DataFrame, questions_col: str = "questions"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the router and return embeddings and labels.

        Args:
            df: DataFrame with questions and model accuracy columns
            questions_col: Name of the questions column

        Returns:
            Tuple of (embeddings, labels)
        """
        self.fit(df, questions_col)
        # After fit(), these are guaranteed to be set
        return self._ensure_embeddings(), self._ensure_labels()

    def transform(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Transform texts to embeddings and cluster assignments.

        Args:
            texts: List of text prompts

        Returns:
            Tuple of (embeddings, cluster_labels)
        """
        self._check_is_fitted()

        # Compute embeddings
        embeddings = self._compute_embeddings(texts)

        # Apply reducer if fitted
        if self._reducer is not None:
            reduced = self._reducer.transform(embeddings)
        else:
            reduced = embeddings

        # Predict clusters
        labels = self._clusterer.predict(reduced)

        return embeddings, labels

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
        self._check_is_fitted()
        core_engine = self._ensure_core_engine()

        # Compute embedding (with caching for repeated prompts)
        embedding = self.compute_embedding(prompt)

        # Ensure float32 and C-contiguous
        if embedding.dtype != np.float32 or not embedding.flags["C_CONTIGUOUS"]:
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        # Route using C++ core
        if models is None:
            models = []
        response = core_engine.route(embedding, models)

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
        self._check_is_fitted()
        core_engine = self._ensure_core_engine()

        if not prompts:
            return []

        if models is None:
            models = []

        # Compute embeddings in batch (more efficient for unique texts)
        embeddings = self._compute_embeddings(prompts)

        # Ensure float32 and C-contiguous
        if embeddings.dtype != np.float32 or not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Route using C++ core engine's route_batch
        responses = core_engine.route_batch(embeddings, models)

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

    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Nordlys must be fitted before use. Call fit() first.")

    def _ensure_embeddings(self) -> np.ndarray:
        """Ensure embeddings are available and return them."""
        self._check_is_fitted()
        if self._embeddings is None:
            raise RuntimeError(
                "Embeddings are not available. This should not happen after fit()."
            )
        return self._embeddings

    def _ensure_labels(self) -> np.ndarray:
        """Ensure labels are available and return them."""
        self._check_is_fitted()
        if self._labels is None:
            raise RuntimeError(
                "Labels are not available. This should not happen after fit()."
            )
        return self._labels

    def _ensure_centroids(self) -> np.ndarray:
        """Ensure centroids are available and return them."""
        self._check_is_fitted()
        if self._centroids is None:
            raise RuntimeError(
                "Centroids are not available. This should not happen after fit()."
            )
        return self._centroids

    def _ensure_model_accuracies(self) -> dict[int, dict[str, float]]:
        """Ensure model accuracies are available and return them."""
        self._check_is_fitted()
        if self._model_accuracies is None:
            raise RuntimeError(
                "Model accuracies are not available. This should not happen after fit()."
            )
        return self._model_accuracies

    def _ensure_metrics(self) -> ClusterMetrics:
        """Ensure metrics are available and return them."""
        self._check_is_fitted()
        if self._metrics is None:
            raise RuntimeError(
                "Metrics are not available. This should not happen after fit()."
            )
        return self._metrics

    def _ensure_core_engine(self) -> NordlysCore:
        """Ensure core engine is available and return it."""
        self._check_is_fitted()
        if self._core_engine is None:
            raise RuntimeError(
                "Core engine is not initialized. This should not happen after fit()."
            )
        return self._core_engine

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
        self._check_is_fitted()
        centroids = self._ensure_centroids()
        labels = self._ensure_labels()
        model_accuracies = self._ensure_model_accuracies()

        if cluster_id < 0 or cluster_id >= len(centroids):
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        mask = labels == cluster_id
        size = int(mask.sum())

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
        self._check_is_fitted()
        centroids = self._ensure_centroids()

        return [
            self.get_cluster_info(cluster_id) for cluster_id in range(len(centroids))
        ]

    def get_metrics(self) -> ClusterMetrics:
        """Get clustering metrics.

        Returns:
            ClusterMetrics object
        """
        return self._ensure_metrics()

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
        return self._ensure_centroids()

    @property
    def labels_(self) -> np.ndarray:
        """Training sample cluster labels of shape (n_samples,)."""
        return self._ensure_labels()

    @property
    def embeddings_(self) -> np.ndarray:
        """Training sample embeddings of shape (n_samples, embedding_dim)."""
        return self._ensure_embeddings()

    @property
    def reduced_embeddings_(self) -> np.ndarray | None:
        """Reduced embeddings if reducer was used, else None."""
        self._check_is_fitted()
        return self._reduced_embeddings

    @property
    def metrics_(self) -> ClusterMetrics:
        """Clustering metrics computed during fit."""
        return self._ensure_metrics()

    @property
    def model_accuracies_(self) -> dict[int, dict[str, float]]:
        """Per-cluster per-model accuracy scores.

        Returns:
            Dict mapping cluster_id -> {model_id: accuracy}
        """
        return self._ensure_model_accuracies()

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        return len(self._ensure_centroids())

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """Save the fitted model to a file.

        Supports both JSON (.json) and MessagePack (.msgpack) formats.

        Args:
            path: Output path (extension determines format)
        """
        self._check_is_fitted()

        path = Path(path)
        checkpoint = self._to_checkpoint()

        if path.suffix.lower() == ".msgpack":
            checkpoint.to_msgpack_file(str(path))
        else:
            checkpoint.to_json_file(str(path))

        logger.info(f"Saved Nordlys model to {path}")

    def _to_checkpoint(self) -> NordlysCheckpoint:
        """Convert fitted state to NordlysCheckpoint."""
        centroids = self._ensure_centroids()
        model_accuracies = self._ensure_model_accuracies()
        metrics = self._ensure_metrics()
        n_clusters = len(centroids)

        # Build models list with error rates (only model_id, no provider/model_name split)
        models = [
            {
                "model_id": model_config.id,
                "cost_per_1m_input_tokens": model_config.cost_input,
                "cost_per_1m_output_tokens": model_config.cost_output,
                "error_rates": [
                    1.0 - model_accuracies.get(cluster_id, {}).get(model_config.id, 0.5)
                    for cluster_id in range(n_clusters)
                ],
            }
            for model_config in self._models
        ]

        # New optimized checkpoint format (v2.0)
        checkpoint_dict = {
            "version": "2.0",
            "cluster_centers": centroids.tolist(),
            "models": models,
            "embedding": {
                "model": self._embedding_model_name,
                "trust_remote_code": self._allow_trust_remote_code,
            },
            "clustering": {
                "n_clusters": n_clusters,
                "random_state": self._random_state,
                "max_iter": DEFAULT_MAX_ITER,
                "n_init": DEFAULT_N_INIT,
                "algorithm": "lloyd",
                "normalization": "l2",
            },
            "metrics": {
                "n_samples": metrics.n_samples,
                "cluster_sizes": metrics.cluster_sizes,
                "silhouette_score": metrics.silhouette_score,
                "inertia": metrics.inertia,
            },
        }

        return NordlysCheckpoint.from_json_string(json.dumps(checkpoint_dict))

    @classmethod
    def load(
        cls,
        path: str | Path,
        models: list[ModelConfig] | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> "Nordlys":
        """Load a fitted model from a file.

        Args:
            path: Path to saved model file
            models: Optional list of model configs (overrides saved costs)
            device: Device for C++ core clustering operations ("cpu" or "cuda")

        Returns:
            Loaded Nordlys instance
        """
        path = Path(path)

        if path.suffix.lower() == ".msgpack":
            checkpoint = NordlysCheckpoint.from_msgpack_file(str(path))
        else:
            checkpoint = NordlysCheckpoint.from_json_file(str(path))

        return cls._from_checkpoint(checkpoint, models, device)

    @classmethod
    def _from_checkpoint(
        cls,
        checkpoint: NordlysCheckpoint,
        models: list[ModelConfig] | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> "Nordlys":
        """Create Nordlys instance from NordlysCheckpoint."""
        # Extract model configs from checkpoint if not provided
        if models is None:
            models = [
                ModelConfig(
                    id=m.model_id,
                    cost_input=m.cost_per_1m_input_tokens,
                    cost_output=m.cost_per_1m_output_tokens,
                )
                for m in checkpoint.models
            ]

        # Create instance using new checkpoint structure
        instance = cls(
            models=models,
            embedding_model=checkpoint.embedding.model,
            nr_clusters=checkpoint.clustering.n_clusters,
            random_state=checkpoint.clustering.random_state,
            allow_trust_remote_code=checkpoint.embedding.trust_remote_code,
            device=device,
        )

        # Initialize C++ core - this is the source of truth for all routing
        instance._core_engine = NordlysCore.from_checkpoint(checkpoint, device=device)

        # Populate Python-side fitted state from checkpoint (always float32)
        instance._centroids = np.asarray(
            checkpoint.cluster_centers,
            dtype=np.float32,
        )

        # Build model_accuracies from checkpoint.models error_rates
        # Structure: {cluster_id: {model_id: accuracy}}
        instance._model_accuracies = {
            cluster_id: {
                model.model_id: 1.0 - model.error_rates[cluster_id]
                for model in checkpoint.models
            }
            for cluster_id in range(checkpoint.clustering.n_clusters)
        }

        # Restore metrics from checkpoint (all fields may be None)
        instance._metrics = ClusterMetrics(
            silhouette_score=checkpoint.metrics.silhouette_score,
            n_clusters=checkpoint.clustering.n_clusters,
            n_samples=checkpoint.metrics.n_samples,
            cluster_sizes=checkpoint.metrics.cluster_sizes,
            inertia=checkpoint.metrics.inertia,
        )

        # These cannot be restored from checkpoint (require original training data)
        instance._labels = None
        instance._embeddings = None
        instance._reduced_embeddings = None

        instance._is_fitted = True

        logger.info("Loaded checkpoint with C++ core (float32)")
        return instance

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"Nordlys(models={len(self._models)}, "
            f"nr_clusters={self._nr_clusters}, "
            f"reducer={self._reducer}, "
            f"clusterer={self._clusterer}, "
            f"status={status})"
        )
