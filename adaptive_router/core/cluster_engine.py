"""Clustering engine for semantic clustering of text inputs.

Simplified for better DX with Trainer and ModelRouter - accepts only strings.
"""

import logging

import numpy as np
from pydantic import BaseModel
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from adaptive_router.core.feature_extractor import FeatureExtractor
from adaptive_router.exceptions.core import (
    ClusterNotConfiguredError,
    ClusterNotFittedError,
)
from adaptive_router.models.storage import ClusteringConfig, FeatureExtractionConfig

logger = logging.getLogger(__name__)


class ClusterStats(BaseModel):
    n_clusters: int
    n_samples: int
    silhouette_score: float
    cluster_sizes: dict[int, int]
    min_cluster_size: int
    max_cluster_size: int
    avg_cluster_size: float


class ClusterEngine(BaseEstimator):
    """Engine for clustering text inputs using K-means on semantic embeddings.

    This class handles the clustering workflow for the adaptive router:
    1. Extracts semantic embeddings from text
    2. Performs spherical K-means clustering (cosine similarity)
    3. Assigns new texts to clusters

    Example:
        >>> engine = ClusterEngine().configure(n_clusters=5)
        >>> engine.fit(["question 1", "question 2", ...])
        >>> cluster_ids = engine.predict(["new question"])
    """

    def __init__(self) -> None:
        """Initialize empty ClusterEngine.

        Call configure() before training, or set attributes directly for restoration.
        This lightweight initialization allows router restoration without loading models.
        """
        # Configuration
        self.n_clusters: int | None = None
        self.max_iter: int | None = None
        self.random_state: int | None = None
        self.n_init: int | None = None
        self.algorithm: str | None = None
        self.normalization_strategy: str | None = None
        self.embedding_model: str | None = None
        self.allow_trust_remote_code: bool = False

        # Configuration objects
        self.clustering_config: ClusteringConfig | None = None
        self.feature_config: FeatureExtractionConfig | None = None

        # Components
        self.feature_extractor: FeatureExtractor | None = None
        self.kmeans: KMeans | None = None

        # Fitted state
        self.cluster_assignments: npt.NDArray[np.int32] = np.array([], dtype=np.int32)
        self.silhouette: float = 0.0
        self.is_fitted_flag: bool = False

    def configure(
        self,
        n_clusters: int = 20,
        max_iter: int = 300,
        random_state: int = 42,
        n_init: int = 10,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        allow_trust_remote_code: bool = False,
    ) -> "ClusterEngine":
        """Configure cluster engine for training.

        Must be called before fit(). Not needed when restoring from saved state.

        Args:
            n_clusters: Number of clusters (K)
            max_iter: Maximum iterations for K-means
            random_state: Random seed for reproducibility
            n_init: Number of K-means runs with different centroid seeds
            embedding_model: HuggingFace model for semantic embeddings
            allow_trust_remote_code: Allow remote code execution in embedding models

        Returns:
            Self for method chaining
        """
        # Create configs from individual parameters
        clustering_config = ClusteringConfig(
            max_iter=max_iter,
            random_state=random_state,
            n_init=n_init,
        )
        feature_config = FeatureExtractionConfig()

        self.clustering_config = clustering_config
        self.feature_config = feature_config

        self.n_clusters = n_clusters
        self.max_iter = clustering_config.max_iter
        self.random_state = clustering_config.random_state
        self.n_init = clustering_config.n_init
        self.algorithm = clustering_config.algorithm
        self.normalization_strategy = clustering_config.normalization_strategy
        self.embedding_model = embedding_model
        self.allow_trust_remote_code = allow_trust_remote_code

        # Initialize heavyweight components
        self.initialize_components()
        return self

    def initialize_components(self) -> None:
        """Initialize FeatureExtractor and KMeans.

        Raises:
            ClusterNotConfiguredError: If configuration parameters not set
        """
        if (
            self.n_clusters is None
            or self.embedding_model is None
            or self.max_iter is None
            or self.random_state is None
            or self.n_init is None
            or self.algorithm is None
        ):
            raise ClusterNotConfiguredError(
                "Configuration incomplete. Call configure() before initializing components."
            )

        logger.info(f"Initializing ClusterEngine with {self.n_clusters} clusters")

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            embedding_model=self.embedding_model,
            allow_trust_remote_code=self.allow_trust_remote_code,
        )

        # Set config on feature_extractor after creation (must be set by configure())
        if self.feature_config is not None:
            self.feature_extractor.config = self.feature_config
            self.feature_extractor.normalize_embeddings = (
                self.feature_config.normalize_embeddings
            )
            self.feature_extractor.embedding_cache_size = (
                self.feature_config.embedding_cache_size
            )

        # K-means clusterer with config
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=self.n_init,
            verbose=0,
            algorithm=self.algorithm,
        )

    def _check_configured(self) -> None:
        """Check if engine is configured with required components.

        Raises:
            ClusterNotConfiguredError: If components not initialized
        """
        if self.feature_extractor is None or self.kmeans is None:
            raise ClusterNotConfiguredError(
                "ClusterEngine not configured. Call configure() before use, "
                "or set components manually during restoration."
            )

    @property
    def is_fitted(self) -> bool:
        """Check if the engine has been fitted."""
        return self.is_fitted_flag

    def fit(self, inputs: list[str]) -> "ClusterEngine":
        """Fit clustering model on text inputs.

        Args:
            inputs: List of text strings to cluster

        Returns:
            Self for method chaining

        Raises:
            ClusterNotConfiguredError: If configure() not called
            ValueError: If inputs list is empty
        """
        self._check_configured()
        assert self.feature_extractor is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not inputs:
            raise ValueError("inputs cannot be empty")

        logger.info(f"Fitting clustering model on {len(inputs)} inputs")

        # Extract semantic embedding features
        features = self.feature_extractor.fit_transform(inputs)

        # Normalize using configured strategy
        norm_strategy = self.normalization_strategy or "l2"
        features_normalized = normalize(features, norm=norm_strategy, copy=False)

        # Perform K-means clustering
        self.kmeans.fit(features_normalized)
        self.cluster_assignments = self.kmeans.labels_.astype(np.int32)

        # Compute silhouette score
        unique_labels = np.unique(self.cluster_assignments)

        if len(unique_labels) > 1:
            self.silhouette = float(
                silhouette_score(features_normalized, self.cluster_assignments)
            )
            logger.info(f"Clustering complete. Silhouette score: {self.silhouette:.3f}")
        else:
            self.silhouette = 0.0
            logger.warning(
                "All points assigned to single cluster - silhouette score undefined"
            )

        self.is_fitted_flag = True
        return self

    def predict(self, inputs: list[str]) -> npt.NDArray[np.int32]:
        """Predict cluster assignments for new text inputs.

        Args:
            inputs: List of text strings to assign to clusters

        Returns:
            Array of cluster IDs (integers)

        Raises:
            ClusterNotConfiguredError: If configure() not called
            ClusterNotFittedError: If predict is called before fit
        """
        self._check_configured()
        assert self.feature_extractor is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not self.is_fitted_flag:
            raise ClusterNotFittedError("Must call fit() before predict()")

        check_is_fitted(self, ["kmeans"])

        # Extract features
        features = self.feature_extractor.transform(inputs)

        # Normalize using configured strategy
        norm_strategy = self.normalization_strategy or "l2"
        features_normalized = normalize(features, norm=norm_strategy, copy=False)

        # Predict clusters
        return self.kmeans.predict(features_normalized).astype(np.int32)

    def assign_single(self, text: str) -> tuple[int, float]:
        """Assign a single text to the nearest cluster.

        Args:
            text: Input text string

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ClusterNotConfiguredError: If configure() not called
            ClusterNotFittedError: If called before fit
        """
        self._check_configured()
        assert self.feature_extractor is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not self.is_fitted_flag:
            raise ClusterNotFittedError("Must call fit() before assign_single()")

        check_is_fitted(self, ["kmeans"])

        # Extract features
        features = self.feature_extractor.transform([text])

        # Normalize
        features_normalized = normalize(features, norm="l2", copy=False)

        # Predict cluster and compute distance
        cluster_id = int(self.kmeans.predict(features_normalized)[0])
        distances = self.kmeans.transform(features_normalized)[0]
        distance = float(distances[cluster_id])

        return cluster_id, distance

    @property
    def cluster_stats(self) -> ClusterStats:
        """Get statistics about clustering results.

        Returns:
            Dictionary with clustering statistics

        Raises:
            ClusterNotFittedError: If called before fit
        """
        if not self.is_fitted_flag:
            raise ClusterNotFittedError(
                "Must call fit() before accessing cluster_stats"
            )

        check_is_fitted(self, ["kmeans"])

        if len(self.cluster_assignments) == 0:
            raise ClusterNotFittedError("No cluster assignments available")

        unique, counts = np.unique(self.cluster_assignments, return_counts=True)

        return ClusterStats(
            n_clusters=self.n_clusters,
            n_samples=len(self.cluster_assignments),
            silhouette_score=self.silhouette,
            cluster_sizes={
                int(cluster_id): int(count) for cluster_id, count in zip(unique, counts)
            },
            min_cluster_size=int(counts.min()),
            max_cluster_size=int(counts.max()),
            avg_cluster_size=float(counts.mean()),
        )
