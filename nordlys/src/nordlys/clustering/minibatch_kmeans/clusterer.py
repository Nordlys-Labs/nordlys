"""MiniBatch K-Means clusterer."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.base import Clusterer
from nordlys.clustering.minibatch_kmeans.cpu import create_sklearn_model, fit as fit_cpu
from nordlys.clustering.minibatch_kmeans.cuda import fit as fit_cuda
from nordlys.clustering.minibatch_kmeans.protocol import MiniBatchKMeansModel
from nordlys.device import DeviceType, get_device, require_cuda


class MiniBatchKMeansClusterer(Clusterer):
    """Mini-Batch K-Means clustering wrapper.

    Thin wrapper over sklearn.cluster.MiniBatchKMeans.
    Useful for large datasets where full K-Means is too slow.
    Supports both CPU (sklearn) and CUDA (custom implementation) execution.

    Example:
        >>> clusterer = MiniBatchKMeansClusterer(n_clusters=20)
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
        reassignment_ratio: float = 0.01,
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize MiniBatchKMeans clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            max_iter: Maximum iterations per run (default: 100)
            batch_size: Size of mini-batches (default: 1024)
            random_state: Random seed for reproducibility (default: 42)
            init_size: Number of samples to use for initialization (default: 3 * batch_size)
            max_no_improvement: Stop if no improvement for N iterations (default: 10)
            reassignment_ratio: Control random cluster reassignments (default: 0.01)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to MiniBatchKMeans
        """
        match device:
            case "cuda":
                require_cuda()
                if init_size is not None:
                    msg = "MiniBatchKMeansCUDA does not support init_size"
                    raise ValueError(msg)
                if max_no_improvement != 10:
                    msg = "MiniBatchKMeansCUDA does not support max_no_improvement"
                    raise ValueError(msg)
                if reassignment_ratio != 0.01:
                    msg = "MiniBatchKMeansCUDA does not support reassignment_ratio"
                    raise ValueError(msg)
                if kwargs:
                    msg = f"MiniBatchKMeansCUDA does not support kwargs: {list(kwargs.keys())}"
                    raise ValueError(msg)

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.init_size = init_size
        self.max_no_improvement = max_no_improvement
        self.reassignment_ratio = reassignment_ratio
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: MiniBatchKMeansModel | None = None

    def _require_model(self) -> MiniBatchKMeansModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "MiniBatchKMeansClusterer":
        """Fit the clusterer on embeddings.

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
                model = create_sklearn_model(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    init_size=self.init_size,
                    max_no_improvement=self.max_no_improvement,
                    reassignment_ratio=self.reassignment_ratio,
                    **self._kwargs,
                )
                self._model = fit_cpu(model, embeddings)
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
        return f"MiniBatchKMeansClusterer(n_clusters={self.n_clusters}, device={self.device!r})"
