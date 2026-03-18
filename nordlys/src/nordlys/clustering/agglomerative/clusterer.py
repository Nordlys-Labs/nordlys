"""Agglomerative clusterer."""

from __future__ import annotations

from nordlys.clustering.base import Clusterer
from nordlys.clustering.agglomerative.cpu import (
    create_sklearn_model,
    fit as fit_cpu,
)
from nordlys.clustering.agglomerative.cuda import fit as fit_cuda
from nordlys.clustering.agglomerative.protocol import (
    AgglomerativeModel,
    AgglomerativeMetric,
)
from nordlys.device import DeviceType, get_device, require_cuda

import numpy as np


class AgglomerativeClusterer(Clusterer):
    """Agglomerative (hierarchical) clustering wrapper.

    Thin wrapper over sklearn.cluster.AgglomerativeClustering with sensible defaults.
    Supports both CPU (sklearn) and CUDA (scipy) execution.

    Example:
        >>> clusterer = AgglomerativeClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        linkage: str = "ward",
        metric: AgglomerativeMetric = "euclidean",
        random_state: int | None = None,
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize Agglomerative clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            linkage: Linkage criterion: "ward", "complete", "average", "single" (default: "ward")
            metric: Distance metric (default: "euclidean"). Note: ward requires euclidean.
            random_state: Random seed for reproducibility (default: None)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to AgglomerativeClustering
        """
        match device:
            case "cuda":
                require_cuda()
                if linkage == "ward" and metric != "euclidean":
                    msg = "Ward linkage requires euclidean metric"
                    raise ValueError(msg)
                if kwargs:
                    msg = f"Agglomerative CUDA does not support kwargs: {list(kwargs.keys())}"
                    raise ValueError(msg)

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.random_state = random_state
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: AgglomerativeModel | None = None
        self._embeddings: np.ndarray | None = None

    def _require_model(self, context: str = "") -> AgglomerativeModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            if context == "predict":
                raise RuntimeError("Clusterer must be fitted before predict")
            raise RuntimeError("Clusterer must be fitted first")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "AgglomerativeClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
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

        match self.device:
            case "cuda":
                self._model = fit_cuda(
                    n_clusters=self.n_clusters,
                    linkage=self.linkage,
                    metric=self.metric,
                    random_state=self.random_state or 42,
                    embeddings=embeddings,
                )
            case _:
                model = create_sklearn_model(
                    n_clusters=self.n_clusters,
                    linkage=self.linkage,
                    metric=self.metric,
                    **self._kwargs,
                )
                self._model = fit_cpu(model, embeddings)
        self._embeddings = embeddings
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments by assigning to nearest centroid.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        return self._require_model("predict").predict(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and return cluster assignments.

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
        """Number of clusters."""
        return self.n_clusters

    def __repr__(self) -> str:
        return f"AgglomerativeClusterer(n_clusters={self.n_clusters}, linkage='{self.linkage}', device={self.device!r})"
