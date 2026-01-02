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
            raise RuntimeError("Clusterer must be fitted before predict. Call fit() first.")

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
        return self._model.labels_

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
        return self._model.labels_

    @property
    def n_clusters_(self) -> int:
        """Number of clusters."""
        return self.n_clusters

    def __repr__(self) -> str:
        return f"SpectralClusterer(n_clusters={self.n_clusters}, affinity='{self.affinity}')"
