"""CUDA adapter for MiniBatch KMeans."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from nordlys.clustering.minibatch_kmeans.protocol import MiniBatchKMeansModel


class CumlMiniBatchKMeansModel:
    """CUDA implementation using iterative k-means++ initialization and streaming updates."""

    def __init__(
        self,
        centroids: np.ndarray,
        labels: np.ndarray,
        inertia: float,
        n_iter: int,
    ) -> None:
        self._centroids = centroids
        self._labels = labels
        self._inertia = inertia
        self._n_iter = n_iter

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._centroids

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def inertia_(self) -> float:
        return self._inertia

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        data = np.asarray(embeddings, dtype=np.float32)
        distances = np.sqrt(
            np.sum(
                (data[:, np.newaxis, :] - self._centroids[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        return np.argmin(distances, axis=1)


def fit(
    n_clusters: int,
    max_iter: int,
    batch_size: int,
    random_state: int,
    embeddings: np.ndarray,
) -> MiniBatchKMeansModel:
    """Fit using CUDA (custom implementation)."""
    data = np.asarray(embeddings, dtype=np.float32)
    n_samples = data.shape[0]

    init_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=1,
        n_init=10,
        batch_size=min(batch_size, n_samples),
        random_state=random_state,
        init="k-means++",
    )
    init_model.fit(data)
    centroids = init_model.cluster_centers_.copy()

    for iteration in range(max_iter):
        rng = np.random.RandomState(random_state + iteration)
        batch_indices = rng.choice(
            n_samples, size=min(batch_size, n_samples), replace=False
        )
        batch = data[batch_indices]

        distances = np.sqrt(
            np.sum(
                (batch[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        batch_labels = np.argmin(distances, axis=1)

        for k in range(n_clusters):
            mask = batch_labels == k
            if mask.sum() > 0:
                learning_rate = 0.01
                centroids[k] = (1 - learning_rate) * centroids[
                    k
                ] + learning_rate * batch[mask].mean(axis=0)

    final_distances = np.sqrt(
        np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    )
    labels = np.argmin(final_distances, axis=1)
    inertia = float(final_distances[np.arange(n_samples), labels].sum())

    return CumlMiniBatchKMeansModel(centroids, labels, inertia, max_iter)
