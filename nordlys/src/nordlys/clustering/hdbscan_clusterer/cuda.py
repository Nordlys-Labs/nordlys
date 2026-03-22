"""GPU adapter for HDBSCAN using cuML."""

from __future__ import annotations

from nordlys.clustering.hdbscan_clusterer.protocol import HDBSCANModel

import logging

import numpy as np

logger = logging.getLogger(__name__)


class HDBSCANCUDAModel:
    """Adapter for cuML HDBSCAN (GPU)."""

    def __init__(
        self,
        cluster_centers: np.ndarray,
        labels: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
    ) -> None:
        self._cluster_centers = cluster_centers
        self._labels = labels if labels is not None else np.array([], dtype=np.int32)
        self._probabilities = (
            probabilities
            if probabilities is not None
            else np.array([], dtype=np.float64)
        )

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @labels_.setter
    def labels_(self, value: np.ndarray) -> None:
        self._labels = value

    @property
    def n_clusters_(self) -> int:
        labels = self._labels
        if len(labels) == 0:
            return 0
        return len(set(labels)) - (1 if -1 in labels else 0)

    @property
    def probabilities_(self) -> np.ndarray:
        return self._probabilities

    @probabilities_.setter
    def probabilities_(self, value: np.ndarray) -> None:
        self._probabilities = value

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._cluster_centers is None or len(self._cluster_centers) == 0:
            return np.full(len(embeddings), -1)
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self._cluster_centers, axis=2
        )
        return distances.argmin(axis=1)


def fit(
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
    prediction_data: bool,
    random_state: int | None,
    embeddings: np.ndarray,
) -> HDBSCANModel:
    """Fit using CUDA (cuML)."""
    import cuml

    if metric != "euclidean":
        logger.warning(
            "cuML HDBSCAN only supports 'euclidean' metric, got '%s'. Using euclidean.",
            metric,
        )

    model = cuml.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        prediction_data=prediction_data,
    )
    model.fit(embeddings)

    labels = model.labels_.astype(np.int32)
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]

    if len(valid_labels) == 0:
        cluster_centers = np.empty((0, embeddings.shape[1]))
    else:
        centers = []
        for label in sorted(valid_labels):
            mask = labels == label
            center = embeddings[mask].mean(axis=0)
            centers.append(center)
        cluster_centers = np.array(centers)

    cuda_model = HDBSCANCUDAModel(cluster_centers)
    cuda_model.labels_ = labels
    cuda_model.probabilities_ = model.probabilities_.astype(np.float64)

    return cuda_model
