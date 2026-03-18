"""Bisecting K-Means clusterer."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.base import Clusterer
from nordlys.clustering.bisecting_kmeans.cpu import fit as fit_cpu
from nordlys.clustering.bisecting_kmeans.cuda import fit as fit_cuda
from nordlys.clustering.bisecting_kmeans.protocol import BisectingKMeansModel
from nordlys.device import DeviceType, get_device, require_cuda


class BisectingKMeansClusterer(Clusterer):
    """Bisecting K-Means clustering wrapper.

    Bisecting K-Means works by repeatedly splitting the largest cluster
    into two until the desired number of clusters is reached.

    Supports both CPU and CUDA execution.

    Example:
        >>> clusterer = BisectingKMeansClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
        >>> labels = clusterer.predict(new_embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        max_iter: int = 100,
        batch_size: int = 1024,
        random_state: int = 42,
        init_size: int | None = None,
        max_no_improvement: int = 10,
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize BisectingKMeans clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            max_iter: Maximum iterations per split (default: 100)
            batch_size: Size of mini-batches (default: 1024)
            random_state: Random seed for reproducibility (default: 42)
            init_size: Number of samples for initialization (default: 3 * batch_size)
            max_no_improvement: Stop if no improvement for N iterations (default: 10)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to MiniBatchKMeans
        """
        match device:
            case "cuda":
                require_cuda()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.init_size = init_size
        self.max_no_improvement = max_no_improvement
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: BisectingKMeansModel | None = None

    def _require_model(self) -> BisectingKMeansModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "BisectingKMeansClusterer":
        """Fit the clusterer on embeddings using bisecting K-Means.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        match self.device:
            case "cuda":
                self._model = fit_cuda(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
            case _:
                self._model = fit_cpu(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    init_size=self.init_size,
                    max_no_improvement=self.max_no_improvement,
                    embeddings=embeddings,
                )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings using nearest centroid.

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
        return self.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""
        return self._require_model().cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        return self._require_model().labels_

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found."""
        return self.n_clusters

    @property
    def inertia_(self) -> float:
        """Sum of squared distances to closest centroid."""
        return self._require_model().inertia_

    def __repr__(self) -> str:
        return f"BisectingKMeansClusterer(n_clusters={self.n_clusters}, device={self.device!r})"
