"""GPU adapter for MiniBatch KMeans using cupy."""

from __future__ import annotations

from nordlys.clustering.minibatch_kmeans.protocol import MiniBatchKMeansModel

import numpy as np


class CumlMiniBatchKMeansModel:
    """GPU implementation of MiniBatchKMeans using cupy."""

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
        squared_distances = np.sum(
            (data[:, np.newaxis, :] - self._centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared_distances, axis=1)


def fit(
    n_clusters: int,
    max_iter: int,
    batch_size: int,
    random_state: int,
    embeddings: np.ndarray,
) -> MiniBatchKMeansModel:
    """Fit using GPU (cupy)."""
    import cupy as cp

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples = data.shape[0]

    centroids = _kmeans_plus_plus_init(data, n_clusters, random_state)

    for iteration in range(max_iter):
        rng = np.random.RandomState(random_state + iteration)
        batch_indices = rng.choice(
            n_samples, size=min(batch_size, n_samples), replace=False
        )
        batch = data[batch_indices]

        squared_distances = cp.sum(
            (batch[:, cp.newaxis, :] - centroids[cp.newaxis, :, :]) ** 2,
            axis=2,
        )
        batch_labels = cp.asarray(np.argmin(cp.asnumpy(squared_distances), axis=1))

        for k in range(n_clusters):
            mask = batch_labels == k
            if int(cp.sum(mask)) > 0:
                learning_rate = 0.01
                centroids[k] = (1 - learning_rate) * centroids[
                    k
                ] + learning_rate * cp.mean(batch[mask], axis=0)

    final_squared_distances = cp.sum(
        (data[:, cp.newaxis, :] - centroids[cp.newaxis, :, :]) ** 2,
        axis=2,
    )
    final_labels = np.argmin(cp.asnumpy(final_squared_distances), axis=1)
    inertia = float(
        final_squared_distances[cp.arange(n_samples), cp.asarray(final_labels)].sum()
    )

    return CumlMiniBatchKMeansModel(
        cp.asnumpy(centroids), final_labels, inertia, max_iter
    )


def _kmeans_plus_plus_init(data, n_clusters: int, random_state: int):
    """Initialize centroids using k-means++ algorithm on GPU."""
    import cupy as cp

    rng = np.random.RandomState(random_state)
    n_samples = int(data.shape[0])

    first_idx = rng.randint(0, n_samples)
    centroids_list = [data[first_idx]]

    for _ in range(n_clusters - 1):
        distances = cp.full(n_samples, cp.inf, dtype=cp.float32)
        for centroid in centroids_list:
            dist = cp.sum((data - centroid) ** 2, axis=1)
            distances = cp.minimum(distances, dist)

        probabilities = distances / distances.sum()
        cumsum = cp.cumsum(probabilities)
        r = rng.random()
        next_idx = int(cp.searchsorted(cumsum, r)[0])
        next_idx = min(next_idx, n_samples - 1)
        centroids_list.append(data[next_idx])

    return cp.stack(centroids_list)
