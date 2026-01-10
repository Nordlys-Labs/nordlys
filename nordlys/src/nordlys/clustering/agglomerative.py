"""Agglomerative clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class AgglomerativeClusterer:
    """Agglomerative (hierarchical) clustering wrapper.

    Thin wrapper over sklearn.cluster.AgglomerativeClustering.

    Example:
        >>> clusterer = AgglomerativeClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        linkage: str = "ward",
        metric: str = "euclidean",
        **kwargs,
    ) -> None:
        """Initialize Agglomerative clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            linkage: Linkage criterion: "ward", "complete", "average", "single" (default: "ward")
            metric: Distance metric (default: "euclidean"). Note: ward requires euclidean.
            **kwargs: Additional arguments passed to AgglomerativeClustering
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self._kwargs = kwargs
        self._model: AgglomerativeClustering | None = None
        self._cluster_centers: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

    def _create_model(self) -> AgglomerativeClustering:
        """Create the underlying AgglomerativeClustering model."""
        # Ward linkage requires euclidean metric
        metric = "euclidean" if self.linkage == "ward" else self.metric
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=metric,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "AgglomerativeClusterer":
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

        Note: Agglomerative clustering doesn't natively support predict.
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
        return f"AgglomerativeClusterer(n_clusters={self.n_clusters}, linkage='{self.linkage}')"
