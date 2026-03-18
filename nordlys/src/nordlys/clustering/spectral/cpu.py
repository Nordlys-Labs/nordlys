"""CPU adapter for Spectral."""

from __future__ import annotations

from nordlys.clustering.spectral.protocol import SpectralModel

import numpy as np
from sklearn.cluster import SpectralClustering


class SklearnSpectralModel:
    """Adapter for sklearn SpectralClustering."""

    def __init__(
        self,
        model: SpectralClustering,
        cluster_centers: np.ndarray,
    ) -> None:
        self._model = model
        self._cluster_centers = cluster_centers

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
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self._cluster_centers, axis=2
        )
        return distances.argmin(axis=1)


def create_sklearn_model(
    n_clusters: int,
    affinity: str,
    n_neighbors: int,
    random_state: int,
    **kwargs: object,
) -> "SklearnSpectralModel":
    """Create a sklearn Spectral model (does not fit).

    Args:
        n_clusters: Number of clusters
        affinity: Affinity type: "nearest_neighbors", "rbf", "precomputed"
        n_neighbors: Number of neighbors for affinity
        random_state: Random seed
        **kwargs: Additional arguments passed to SpectralClustering

    Returns:
        An unfitted SklearnSpectralModel instance
    """
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        random_state=random_state,
        **kwargs,
    )
    return SklearnSpectralModel(model, np.array([]))


def fit(
    model: SklearnSpectralModel,
    embeddings: np.ndarray,
) -> SpectralModel:
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
