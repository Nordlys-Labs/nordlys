"""Spectral clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering


class SpectralClusterer:
    """Spectral clustering wrapper.

    Thin wrapper over sklearn.cluster.SpectralClustering.

    Example:
        >>> clusterer = SpectralClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        affinity: str = "nearest_neighbors",
        n_neighbors: int = 10,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Initialize Spectral clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            affinity: Affinity type: "nearest_neighbors", "rbf", "precomputed" (default: "nearest_neighbors")
            n_neighbors: Number of neighbors for affinity (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional arguments passed to SpectralClustering
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._kwargs = kwargs
        self._model: SpectralClustering | None = None
        self._cluster_centers: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

    def _create_model(self) -> SpectralClustering:
        """Create the underlying SpectralClustering model."""
        return SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "SpectralClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        # Validate input embeddings
        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got {embeddings.ndim}D array"
            )

        n_samples, n_features = embeddings.shape

        if n_samples == 0 or n_features == 0:
            raise ValueError(
                f"Embeddings cannot be empty: got shape ({n_samples}, {n_features})"
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain NaN or Inf values. All values must be finite."
            )

        if n_samples < self.n_clusters:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        self._model = self._create_model()
        self._model.fit(embeddings)
        self._embeddings = embeddings
        self._compute_cluster_centers()
        return self

    def _compute_cluster_centers(self) -> None:
        """Compute cluster centers as mean of cluster members."""
        if self._model is None or self._embeddings is None:
            return

        labels = self._model.labels_
        unique_labels = np.unique(labels)

        centers = []
        for label in sorted(unique_labels):
            mask = labels == label
            center = self._embeddings[mask].mean(axis=0)
            centers.append(center)

        self._cluster_centers = np.array(centers)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments by assigning to nearest centroid.

        Note: Spectral clustering doesn't natively support predict.
        This assigns new samples to their nearest cluster center.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self._cluster_centers is None:
            raise RuntimeError(
                "Clusterer must be fitted before predict. Call fit() first."
            )

        # Validate input embeddings
        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got {embeddings.ndim}D array"
            )

        n_samples, n_features = embeddings.shape

        if n_samples == 0:
            raise ValueError("Embeddings cannot be empty: got 0 samples")

        expected_features = self._cluster_centers.shape[1]
        if n_features != expected_features:
            raise ValueError(
                f"Feature dimension mismatch: embeddings have {n_features} features, "
                f"but clusterer was fitted with {expected_features} features"
            )

        # Ensure numeric dtype
        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError(
                f"Embeddings must have numeric dtype, got {embeddings.dtype}"
            )

        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self._cluster_centers, axis=2
        )
        return distances.argmin(axis=1)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and return cluster assignments.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        self.fit(embeddings)
        assert self._model is not None
        labels = self._model.labels_
        assert labels is not None
        return labels

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features).

        Computed as the mean of cluster members.
        """
        if self._cluster_centers is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        labels = self._model.labels_
        assert labels is not None
        return labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        return self.n_clusters

    def __repr__(self) -> str:
        return f"SpectralClusterer(n_clusters={self.n_clusters}, affinity='{self.affinity}')"
