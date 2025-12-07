"""Clustering engine for semantic clustering of text inputs.

Simplified for better DX with Trainer and ModelRouter - accepts only strings.
"""

import logging
import platform

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted

from adaptive_router.exceptions.core import (
    ClusterNotConfiguredError,
    ClusterNotFittedError,
)
from adaptive_router.models.storage import ClusteringConfig, ClusterStats

logger = logging.getLogger(__name__)


class ClusterEngine(BaseEstimator):
    """Engine for clustering text inputs using K-means on semantic embeddings.

    This class handles the clustering workflow for the adaptive router:
    1. Extracts semantic embeddings from text using SentenceTransformers
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
        self.embedding_model_name: str | None = None
        self.allow_trust_remote_code: bool = False
        self.batch_size: int = 32

        # Configuration objects
        self.clustering_config: ClusteringConfig | None = None

        # Components
        self.embedding_model: SentenceTransformer | None = None
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
        batch_size: int = 32,
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
            batch_size: Batch size for embedding generation

        Returns:
            Self for method chaining
        """
        # Create config from parameters
        clustering_config = ClusteringConfig(
            max_iter=max_iter,
            random_state=random_state,
            n_init=n_init,
        )

        self.clustering_config = clustering_config
        self.n_clusters = n_clusters
        self.max_iter = clustering_config.max_iter
        self.random_state = clustering_config.random_state
        self.n_init = clustering_config.n_init
        self.algorithm = clustering_config.algorithm
        self.normalization_strategy = clustering_config.normalization_strategy
        self.embedding_model_name = embedding_model
        self.allow_trust_remote_code = allow_trust_remote_code
        self.batch_size = batch_size

        # Initialize heavyweight components
        self.initialize_components()
        return self

    @staticmethod
    def get_device() -> str:
        """Determine the appropriate device for model loading.

        Returns:
            Device string: 'cpu' for macOS, 'cuda' if available, otherwise 'cpu'
        """
        if platform.system() == "Darwin":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_components(self) -> None:
        """Initialize SentenceTransformer and KMeans.

        Raises:
            ClusterNotConfiguredError: If configuration parameters not set
        """
        if (
            self.n_clusters is None
            or self.embedding_model_name is None
            or self.max_iter is None
            or self.random_state is None
            or self.n_init is None
            or self.algorithm is None
        ):
            raise ClusterNotConfiguredError(
                "Configuration incomplete. Call configure() before initializing components."
            )

        logger.info(f"Initializing ClusterEngine with {self.n_clusters} clusters")

        # Determine device
        device = self.get_device()
        logger.info(
            f"Loading embedding model '{self.embedding_model_name}' on device: {device}"
        )

        # Load SentenceTransformer
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=device,
            trust_remote_code=self.allow_trust_remote_code,
        )

        # Set tokenizer cleanup for future compatibility
        try:
            self.embedding_model.tokenizer.clean_up_tokenization_spaces = False
        except AttributeError:
            pass

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
        if self.embedding_model is None or self.kmeans is None:
            raise ClusterNotConfiguredError(
                "ClusterEngine not configured. Call configure() before use, "
                "or set components manually during restoration."
            )

    def _extract_embeddings(
        self, texts: list[str], show_progress: bool = False
    ) -> npt.NDArray[np.float32]:
        """Extract embeddings from texts using SentenceTransformer.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array (n_samples × embedding_dim) as float32
        """
        assert self.embedding_model is not None  # For mypy

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=self.batch_size,
            normalize_embeddings=False,  # We normalize separately with L2
            convert_to_numpy=True,
        )

        # Ensure float32 dtype for sklearn compatibility
        return embeddings.astype(np.float32, copy=False)

    def _normalize_features(
        self, features: npt.NDArray[np.float32], norm: str = "l2"
    ) -> npt.NDArray[np.float32]:
        """Normalize features for spherical k-means.

        Args:
            features: Feature array (float32)
            norm: Normalization strategy ('l1', 'l2', or 'max')

        Returns:
            Normalized features as float32 array
        """
        features_normalized = normalize(features, norm=norm, copy=False)
        # Always ensure float32 for sklearn KMeans compatibility
        return np.asarray(features_normalized, dtype=np.float32)

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
        assert self.embedding_model is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not inputs:
            raise ValueError("inputs cannot be empty")

        logger.info(f"Fitting clustering model on {len(inputs)} inputs")

        # Extract semantic embeddings
        logger.info("Generating embeddings...")
        embeddings = self._extract_embeddings(inputs, show_progress=True)

        # Normalize for spherical k-means (cosine similarity)
        logger.info("Normalizing embeddings...")
        norm_strategy = self.normalization_strategy or "l2"
        embeddings_normalized = self._normalize_features(embeddings, norm=norm_strategy)

        # Perform K-means clustering
        logger.info("Performing k-means clustering...")
        self.kmeans.fit(embeddings_normalized)
        self.cluster_assignments = self.kmeans.labels_.astype(np.int32)

        # Compute silhouette score
        unique_labels = np.unique(self.cluster_assignments)

        if len(unique_labels) > 1:
            self.silhouette = float(
                silhouette_score(embeddings_normalized, self.cluster_assignments)
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
        assert self.embedding_model is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not self.is_fitted_flag:
            raise ClusterNotFittedError("Must call fit() before predict()")

        check_is_fitted(self, ["kmeans"])

        # Extract embeddings
        embeddings = self._extract_embeddings(inputs, show_progress=False)

        # Normalize
        norm_strategy = self.normalization_strategy or "l2"
        embeddings_normalized = self._normalize_features(
            embeddings, norm=norm_strategy
        ).astype(np.float32)

        # Predict clusters
        return self.kmeans.predict(embeddings_normalized).astype(np.int32)

    def assign_single(self, text: str) -> tuple[int, float]:
        """Assign a single text to the nearest cluster.

        Performance: <5ms per call (feature extraction dominates at ~1-3ms)

        Args:
            text: Input text string

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ClusterNotConfiguredError: If configure() not called
            ClusterNotFittedError: If called before fit
        """
        self._check_configured()
        assert self.embedding_model is not None  # For mypy
        assert self.kmeans is not None  # For mypy

        if not self.is_fitted_flag:
            raise ClusterNotFittedError("Must call fit() before assign_single()")

        check_is_fitted(self, ["kmeans"])

        # Extract embedding
        embeddings = self._extract_embeddings([text], show_progress=False)

        # Normalize
        norm_strategy = self.normalization_strategy or "l2"
        embeddings_normalized = self._normalize_features(
            embeddings, norm=norm_strategy
        ).astype(np.float32)

        # Predict cluster and compute distance
        cluster_id = int(self.kmeans.predict(embeddings_normalized)[0])
        distances = self.kmeans.transform(embeddings_normalized)[0]
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

        if self.n_clusters is None:
            raise ClusterNotConfiguredError("n_clusters is not configured")

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

    @classmethod
    def from_fitted_state(
        cls,
        cluster_centers: np.ndarray,
        n_clusters: int,
        embedding_model: SentenceTransformer,
        embedding_model_name: str,
        silhouette_score: float = 0.0,
        clustering_config: ClusteringConfig | None = None,
    ) -> "ClusterEngine":
        """Restore a ClusterEngine from fitted state (for router loading).

        Args:
            cluster_centers: K-means cluster centers array
            n_clusters: Number of clusters
            embedding_model: Loaded SentenceTransformer model
            embedding_model_name: Name of the embedding model
            silhouette_score: Clustering quality score
            clustering_config: Clustering configuration

        Returns:
            Configured ClusterEngine ready for prediction
        """
        # Create empty instance
        engine = cls()

        # Set configuration from clustering_config if provided
        if clustering_config:
            engine.clustering_config = clustering_config
            engine.max_iter = clustering_config.max_iter
            engine.random_state = clustering_config.random_state
            engine.n_init = clustering_config.n_init
            engine.algorithm = clustering_config.algorithm
            engine.normalization_strategy = clustering_config.normalization_strategy

        # Set core attributes
        engine.n_clusters = n_clusters
        engine.embedding_model_name = embedding_model_name
        engine.embedding_model = embedding_model
        engine.batch_size = 32  # Default batch size

        # Create KMeans with fitted state
        engine.kmeans = KMeans(n_clusters=n_clusters)
        engine.kmeans.cluster_centers_ = cluster_centers.astype(np.float32)
        engine.kmeans.n_iter_ = clustering_config.n_iter if clustering_config else 0
        engine.kmeans.n_features_in_ = cluster_centers.shape[1]
        engine.kmeans._n_threads = 1  # Runtime default for sklearn 1.4+

        # Set fitted state
        engine.silhouette = silhouette_score
        engine.is_fitted_flag = True

        return engine
