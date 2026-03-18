"""GPU adapter for MiniBatch KMeans using cupy."""

from __future__ import annotations

from nordlys.clustering._cuda_minibatch import (
    _fit_minibatch_core,
    greedy_kmeanspp_init,
)
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
    """Fit using GPU (cupy), matching sklearn MiniBatchKMeans semantics."""
    import cupy as cp

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples, n_features = data.shape

    rng = np.random.RandomState(random_state)

    centers = greedy_kmeanspp_init(data, n_clusters, rng, x_squared_norms=None)

    centers, labels, inertia, n_steps = _fit_minibatch_core(
        data=data,
        init_centers=centers,
        batch_size=batch_size,
        max_iter=max_iter,
        max_no_improvement=10,
        reassignment_ratio=0.01,
        random_state=rng,
    )

    n_iter = int(np.ceil((n_steps * min(batch_size, n_samples)) / n_samples))

    return CumlMiniBatchKMeansModel(
        cp.asnumpy(centers),
        cp.asnumpy(labels).astype(np.int64),
        inertia,
        n_iter,
    )
