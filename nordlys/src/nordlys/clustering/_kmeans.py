"""K-Means clusterer."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class KMeansClusterer:
    """K-Means clustering wrapper.

    Thin wrapper over sklearn.cluster.KMeans with sensible defaults.

    Example:
        >>> clusterer = KMeansClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
        >>> labels = clusterer.predict(new_embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42,
        algorithm: str = "lloyd",
        **kwargs,
    ) -> None:
        """Initialize K-Means clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            max_iter: Maximum iterations per run (default: 300)
            n_init: Number of initializations (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            algorithm: K-means algorithm variant (default: "lloyd")
            **kwargs: Additional arguments passed to KMeans
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.algorithm = algorithm
        self._kwargs = kwargs
        self._model: KMeans | None = None

    def _create_model(self) -> KMeans:
        """Create the underlying KMeans model."""
        return KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
            algorithm=self.algorithm,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "KMeansClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        self._model = self._create_model()
        self._model.fit(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self._model is None:
            raise RuntimeError(
                "Clusterer must be fitted before predict. Call fit() first."
            )
        return self._model.predict(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and predict cluster assignments.

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
        """Cluster centers of shape (n_clusters, n_features)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model.cluster_centers_

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
        """Number of clusters found."""
        return self.n_clusters

    @property
    def inertia_(self) -> float:
        """Sum of squared distances to closest centroid."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        inertia = self._model.inertia_
        assert inertia is not None
        return inertia

    @property
    def n_iter_(self) -> int:
        """Number of iterations run."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model.n_iter_

    def __repr__(self) -> str:
        return f"KMeansClusterer(n_clusters={self.n_clusters})"
