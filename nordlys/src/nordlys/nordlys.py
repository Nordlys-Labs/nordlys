"""Main Nordlys orchestrator class with sklearn-like API.

This module provides the unified Nordlys class that orchestrates
clustering, routing, and model selection with a simple BERTopic-style API.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
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
from nordlys_core_ext import Nordlys32, Nordlys64, NordlysCheckpoint


logger = logging.getLogger(__name__)


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
        if "/" in self.id:
            return self.id.split("/", 1)[0]
        return ""

    @property
    def model_name(self) -> str:
        """Extract model name from model ID."""
        if "/" in self.id:
            return self.id.split("/", 1)[1]
        return self.id

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
        >>> result = model.route("Explain quantum computing", cost_bias=0.5)
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
        """
        # C++ core (initialized on load or after fit) - set early to avoid __del__ errors
        self._core_engine: Nordlys32 | Nordlys64 | None = None

        if not models:
            raise ValueError("At least one model configuration is required")

        self._models = models
        self._model_ids = [m.id for m in models]

        # Embedding model - lazy loaded on first use
        self._embedding_model_name = embedding_model
        self._embedding_model: SentenceTransformer | None = None
        self._allow_trust_remote_code = allow_trust_remote_code

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
        self._dtype = "float32"

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model lazily."""
        if self._embedding_model is not None:
            return self._embedding_model

        device = _get_device()
        logger.info(
            f"Loading embedding model '{self._embedding_model_name}' on device: {device}"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name,
                device=device,
                trust_remote_code=self._allow_trust_remote_code,
            )
        self._embedding_model.tokenizer.clean_up_tokenization_spaces = False

        return self._embedding_model

    def _compute_embeddings(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        model = self._load_embedding_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

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
            if self._dtype == "float32":
                self._core_engine = Nordlys32.from_checkpoint(checkpoint)
            else:
                self._core_engine = Nordlys64.from_checkpoint(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize C++ core engine: {e}") from e

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
        assert self._embeddings is not None
        assert self._labels is not None
        return self._embeddings, self._labels

    def transform(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        cost_bias: float = 0.5,
    ) -> RouteResult:
        """Route a prompt to the best model using C++ core engine.

        Args:
            prompt: The text prompt to route
            cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)

        Returns:
            RouteResult with selected model and alternatives
        """
        self._check_is_fitted()
        assert self._core_engine is not None

        # Compute embedding
        embedding = self._compute_embeddings([prompt])[0]

        # Ensure correct dtype and C-contiguous
        target_dtype = np.float64 if self._dtype == "float64" else np.float32
        if embedding.dtype != target_dtype or not embedding.flags["C_CONTIGUOUS"]:
            embedding = np.ascontiguousarray(embedding, dtype=target_dtype)

        # Route using C++ core
        response = self._core_engine.route(embedding, cost_bias, [])

        return RouteResult(
            model_id=response.selected_model,
            cluster_id=response.cluster_id,
            cluster_distance=float(response.cluster_distance),
            alternatives=list(response.alternatives),
        )

    def route_batch(
        self,
        prompts: list[str],
        cost_bias: float = 0.5,
    ) -> list[RouteResult]:
        """Route multiple prompts in batch.

        Args:
            prompts: List of text prompts
            cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)

        Returns:
            List of RouteResults
        """
        return [self.route(p, cost_bias) for p in prompts]

    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Nordlys must be fitted before use. Call fit() first.")

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
        assert self._centroids is not None
        assert self._labels is not None
        assert self._model_accuracies is not None

        if cluster_id < 0 or cluster_id >= len(self._centroids):
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        mask = self._labels == cluster_id
        size = int(mask.sum())

        return ClusterInfo(
            cluster_id=cluster_id,
            size=size,
            centroid=self._centroids[cluster_id],
            model_accuracies=self._model_accuracies.get(cluster_id, {}),
        )

    def get_clusters(self) -> list[ClusterInfo]:
        """Get information about all clusters.

        Returns:
            List of ClusterInfo objects
        """
        self._check_is_fitted()
        assert self._centroids is not None

        return [
            self.get_cluster_info(cluster_id)
            for cluster_id in range(len(self._centroids))
        ]

    def get_metrics(self) -> ClusterMetrics:
        """Get clustering metrics.

        Returns:
            ClusterMetrics object
        """
        self._check_is_fitted()
        assert self._metrics is not None
        return self._metrics

    # =========================================================================
    # Fitted attributes (sklearn convention: trailing underscore)
    # =========================================================================

    @property
    def centroids_(self) -> np.ndarray:
        """Cluster centroids of shape (n_clusters, n_features)."""
        self._check_is_fitted()
        assert self._centroids is not None
        return self._centroids

    @property
    def labels_(self) -> np.ndarray:
        """Training sample cluster labels of shape (n_samples,)."""
        self._check_is_fitted()
        assert self._labels is not None
        return self._labels

    @property
    def embeddings_(self) -> np.ndarray:
        """Training sample embeddings of shape (n_samples, embedding_dim)."""
        self._check_is_fitted()
        assert self._embeddings is not None
        return self._embeddings

    @property
    def reduced_embeddings_(self) -> np.ndarray | None:
        """Reduced embeddings if reducer was used, else None."""
        self._check_is_fitted()
        return self._reduced_embeddings

    @property
    def metrics_(self) -> ClusterMetrics:
        """Clustering metrics computed during fit."""
        self._check_is_fitted()
        assert self._metrics is not None
        return self._metrics

    @property
    def model_accuracies_(self) -> dict[int, dict[str, float]]:
        """Per-cluster per-model accuracy scores.

        Returns:
            Dict mapping cluster_id -> {model_id: accuracy}
        """
        self._check_is_fitted()
        assert self._model_accuracies is not None
        return self._model_accuracies

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        self._check_is_fitted()
        assert self._centroids is not None
        return len(self._centroids)

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
        assert self._centroids is not None
        assert self._model_accuracies is not None
        assert self._metrics is not None

        centroids = self._centroids
        model_accuracies = self._model_accuracies
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
                "dtype": self._dtype,
                "trust_remote_code": self._allow_trust_remote_code,
            },
            "clustering": {
                "n_clusters": n_clusters,
                "random_state": self._random_state,
                "max_iter": 300,
                "n_init": 10,
                "algorithm": "lloyd",
                "normalization": "l2",
            },
            "routing": {
                "cost_bias_min": 0.0,
                "cost_bias_max": 1.0,
                "default_cost_bias": 0.5,
                "max_alternatives": 5,
            },
            "metrics": {
                "n_samples": self._metrics.n_samples,
                "cluster_sizes": self._metrics.cluster_sizes,
                "silhouette_score": self._metrics.silhouette_score,
                "inertia": self._metrics.inertia,
            },
        }

        return NordlysCheckpoint.from_json_string(json.dumps(checkpoint_dict))

    @classmethod
    def load(
        cls, path: str | Path, models: list[ModelConfig] | None = None
    ) -> "Nordlys":
        """Load a fitted model from a file.

        Args:
            path: Path to saved model file
            models: Optional list of model configs (overrides saved costs)

        Returns:
            Loaded Nordlys instance
        """
        path = Path(path)

        if path.suffix.lower() == ".msgpack":
            checkpoint = NordlysCheckpoint.from_msgpack_file(str(path))
        else:
            checkpoint = NordlysCheckpoint.from_json_file(str(path))

        return cls._from_checkpoint(checkpoint, models)

    @classmethod
    def _from_checkpoint(
        cls,
        checkpoint: NordlysCheckpoint,
        models: list[ModelConfig] | None = None,
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
        )

        # Initialize C++ core - this is the source of truth for all routing
        if checkpoint.embedding.dtype == "float32":
            instance._core_engine = Nordlys32.from_checkpoint(checkpoint)
        else:
            instance._core_engine = Nordlys64.from_checkpoint(checkpoint)

        # Populate Python-side fitted state from checkpoint
        instance._dtype = checkpoint.embedding.dtype
        instance._centroids = np.asarray(
            checkpoint.cluster_centers,
            dtype=np.float32 if checkpoint.embedding.dtype == "float32" else np.float64,
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

        logger.info(f"Loaded checkpoint with C++ core ({checkpoint.embedding.dtype})")
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
