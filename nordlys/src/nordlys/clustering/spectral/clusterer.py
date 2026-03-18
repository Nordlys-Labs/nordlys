"""Spectral clusterer."""

from __future__ import annotations

from nordlys.clustering.base import Clusterer
from nordlys.clustering.spectral.cpu import create_sklearn_model, fit as fit_cpu
from nordlys.clustering.spectral.cuda import fit as fit_cuda
from nordlys.clustering.spectral.protocol import SpectralModel
from nordlys.device import DeviceType, get_device, require_cuda

import numpy as np


class SpectralClusterer(Clusterer):
    """Spectral clustering wrapper.

    Thin wrapper over sklearn.cluster.SpectralClustering with sensible defaults.
    Supports both CPU (sklearn) and CUDA (custom cupy) execution.

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
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize Spectral clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            affinity: Affinity type: "nearest_neighbors", "rbf", "precomputed" (default: "nearest_neighbors")
            n_neighbors: Number of neighbors for affinity (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to SpectralClustering
        """
        match device:
            case "cuda":
                require_cuda()
                if affinity != "nearest_neighbors":
                    msg = (
                        f"Spectral CUDA supports only affinity='nearest_neighbors', "
                        f"got '{affinity}'"
                    )
                    raise ValueError(msg)
                if kwargs:
                    msg = (
                        f"Spectral CUDA does not support kwargs: {list(kwargs.keys())}"
                    )
                    raise ValueError(msg)

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: SpectralModel | None = None

    def _require_model(self, context: str = "") -> SpectralModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            if context == "predict":
                raise RuntimeError("Clusterer must be fitted before predict")
            raise RuntimeError("Clusterer must be fitted first")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "SpectralClusterer":
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
                    affinity=self.affinity,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
            case _:
                model = create_sklearn_model(
                    n_clusters=self.n_clusters,
                    affinity=self.affinity,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                    **self._kwargs,
                )
                self._model = fit_cpu(model, embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments by assigning to nearest centroid.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, n_features), got {embeddings.ndim}D array"
            )

        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings cannot be empty")

        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError("Embeddings must have numeric dtype")

        model = self._require_model("predict")
        n_features = model.cluster_centers_.shape[1]
        if embeddings.shape[1] != n_features:
            raise ValueError(
                f"Feature dimension mismatch: embeddings have {embeddings.shape[1]} features, "
                f"but clusterer was fitted with {n_features} features"
            )

        return model.predict(embeddings)

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
        return f"SpectralClusterer(n_clusters={self.n_clusters}, affinity='{self.affinity}', device={self.device!r})"
