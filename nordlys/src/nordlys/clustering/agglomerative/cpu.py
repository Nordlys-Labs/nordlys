"""CPU adapter for Agglomerative."""

from __future__ import annotations

from nordlys.clustering.agglomerative.protocol import (
    AgglomerativeModel,
    AgglomerativeMetric,
)

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


class SklearnAgglomerativeModel:
    """Adapter for sklearn AgglomerativeClustering."""

    def __init__(
        self,
        model: AgglomerativeClustering,
        cluster_centers: np.ndarray,
        metric: AgglomerativeMetric,
    ) -> None:
        self._model = model
        self._cluster_centers = cluster_centers
        self._metric = metric

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._model.labels_

    @property
    def n_clusters_(self) -> int:
        return self._model.n_clusters

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._metric not in ("euclidean", "l2", "sqeuclidean"):
            raise ValueError(
                f"predict() is not supported for metric '{self._metric}'. "
                "Only 'euclidean' (or 'l2', 'sqeuclidean') metrics are supported for prediction."
            )
        distances = pairwise_distances(
            embeddings, self._cluster_centers, metric="euclidean"
        )
        return distances.argmin(axis=1)


def create_sklearn_model(
    n_clusters: int,
    linkage: str,
    metric: AgglomerativeMetric,
    **kwargs: object,
) -> "SklearnAgglomerativeModel":
    """Create a sklearn Agglomerative model (does not fit).

    Args:
        n_clusters: Number of clusters
        linkage: Linkage criterion: "ward", "complete", "average", "single"
        metric: Distance metric
        **kwargs: Additional arguments passed to AgglomerativeClustering

    Returns:
        An unfitted SklearnAgglomerativeModel instance
    """
    actual_metric = "euclidean" if linkage == "ward" else metric
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=actual_metric,
        **kwargs,
    )
    return SklearnAgglomerativeModel(model, np.array([]), metric)


def fit(
    model: SklearnAgglomerativeModel,
    embeddings: np.ndarray,
) -> AgglomerativeModel:
    """Fit the sklearn model and return the adapter."""
    model._model.fit(embeddings)
    labels = model._model.labels_
    unique_labels = np.unique(labels)
    centers = []
    for label in sorted(unique_labels):
        mask = labels == label
        center = embeddings[mask].mean(axis=0)
        centers.append(center)
    model._cluster_centers = np.array(centers)
    return model
