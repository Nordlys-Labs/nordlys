"""HDBSCAN clusterer."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.base import Clusterer
from nordlys.clustering.hdbscan_clusterer.cpu import fit as fit_cpu
from nordlys.clustering.hdbscan_clusterer.cuda import fit as fit_cuda
from nordlys.clustering.hdbscan_clusterer.protocol import HDBSCANModel
from nordlys.device import DeviceType, get_device, require_cuda


class HDBSCANClusterer(Clusterer):
    """HDBSCAN clustering wrapper.

    Thin wrapper over hdbscan.HDBSCAN with sensible defaults.
    HDBSCAN automatically determines the number of clusters.
    Supports both CPU (hdbscan) and CUDA (cuML) execution.

    Example:
        >>> clusterer = HDBSCANClusterer(min_cluster_size=100)
        >>> clusterer.fit(embeddings)
        >>> # Note: HDBSCAN doesn't support predict() for new samples by default
    """

    def __init__(
        self,
        min_cluster_size: int = 100,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        prediction_data: bool = True,
        random_state: int | None = None,
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum size of clusters (default: 100)
            min_samples: Minimum samples for core points (default: None, uses min_cluster_size)
            metric: Distance metric (default: "euclidean")
            cluster_selection_epsilon: Distance threshold for merging (default: 0.0)
            cluster_selection_method: "eom" or "leaf" (default: "eom")
            prediction_data: Generate prediction data for approximate_predict (default: True)
            random_state: Random seed for reproducibility (default: None)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to HDBSCAN
        """
        match device:
            case "cuda":
                require_cuda()
                if metric != "euclidean":
                    msg = "HDBSCANCUDA supports only metric='euclidean'"
                    raise ValueError(msg)

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.prediction_data = prediction_data
        self.random_state = random_state
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: HDBSCANModel | None = None
        self._embeddings: np.ndarray | None = None

    def _require_model(self) -> HDBSCANModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "HDBSCANClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        self._embeddings = embeddings
        match self.device:
            case "cuda":
                self._model = fit_cuda(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    cluster_selection_method=self.cluster_selection_method,
                    prediction_data=self.prediction_data,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
            case _:
                self._model = fit_cpu(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_epsilon=self.cluster_selection_epsilon,
                    cluster_selection_method=self.cluster_selection_method,
                    prediction_data=self.prediction_data,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        return self._require_model().predict(embeddings)

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
        """Cluster centers of shape (n_clusters, n_features).

        Computed as the mean of cluster members.
        """
        return self._require_model().cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,).

        -1 indicates noise points.
        """
        return self._require_model().labels_

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding noise)."""
        return self._require_model().n_clusters_

    @property
    def probabilities_(self) -> np.ndarray:
        """Cluster membership probabilities of shape (n_samples,)."""
        return self._require_model().probabilities_

    def __repr__(self) -> str:
        return f"HDBSCANClusterer(min_cluster_size={self.min_cluster_size}, device={self.device!r})"
