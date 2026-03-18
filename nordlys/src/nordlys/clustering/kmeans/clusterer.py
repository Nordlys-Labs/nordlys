"""K-Means clusterer."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.base import Clusterer
from nordlys.clustering.kmeans.cpu import create_sklearn_model, fit as fit_cpu
from nordlys.clustering.kmeans.cuda import fit as fit_cuda
from nordlys.clustering.kmeans.protocol import KMeansModel
from nordlys.device import DeviceType, get_device, require_cuda


class KMeansClusterer(Clusterer):
    """K-Means clustering wrapper.

    Thin wrapper over sklearn.cluster.KMeans with sensible defaults.
    Supports both CPU (sklearn) and CUDA (cuML) execution.

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
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize K-Means clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            max_iter: Maximum iterations per run (default: 300)
            n_init: Number of initializations (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            algorithm: K-means algorithm variant (default: "lloyd")
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to KMeans
        """
        match device:
            case "cuda":
                require_cuda()
                if algorithm != "lloyd":
                    msg = (
                        f"KMeansCUDA supports only algorithm='lloyd', got '{algorithm}'"
                    )
                    raise ValueError(msg)
                if kwargs:
                    msg = f"KMeansCUDA does not support kwargs: {list(kwargs.keys())}"
                    raise ValueError(msg)

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.algorithm = algorithm
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: KMeansModel | None = None

    def _require_model(self) -> KMeansModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "KMeansClusterer":
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
                    n_init=self.n_init,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
            case _:
                model = create_sklearn_model(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=self.random_state,
                    algorithm=self.algorithm,
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

    @property
    def n_iter_(self) -> int:
        """Number of iterations run."""
        return self._require_model().n_iter_

    def __repr__(self) -> str:
        return f"KMeansClusterer(n_clusters={self.n_clusters}, device={self.device!r})"
