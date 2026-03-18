"""GPU adapter for Bisecting KMeans using cupy."""

from __future__ import annotations

from nordlys.clustering._cuda_minibatch import (
    _fit_minibatch_core,
    greedy_kmeanspp_init,
)
from nordlys.clustering.bisecting_kmeans.protocol import BisectingKMeansModel

import numpy as np


class CumlBisectingKMeansModel:
    """Result model for BisectingKMeans CUDA."""

    def __init__(
        self,
        cluster_centers: np.ndarray,
        labels: np.ndarray,
        inertia: float,
    ) -> None:
        self._cluster_centers = cluster_centers
        self._labels = labels
        self._inertia = inertia

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def inertia_(self) -> float:
        return self._inertia

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        data = np.asarray(embeddings, dtype=np.float32)
        squared_distances = np.sum(
            (data[:, np.newaxis, :] - self._cluster_centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(squared_distances, axis=1)


def fit(
    n_clusters: int,
    max_iter: int,
    batch_size: int,
    random_state: int,
    embeddings: np.ndarray,
) -> BisectingKMeansModel:
    """Fit using CUDA (cupy), matching CPU bisecting semantics."""
    import cupy as cp

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples = data.shape[0]

    if n_clusters <= 1:
        labels = cp.zeros(n_samples, dtype=cp.int64)
        cluster_centers = cp.mean(data, axis=0, keepdims=True)
        inertia = float(cp.sum((data - cluster_centers) ** 2))
        return CumlBisectingKMeansModel(
            cp.asnumpy(cluster_centers), cp.asnumpy(labels).astype(np.int64), inertia
        )

    cluster_assignments = cp.zeros(n_samples, dtype=cp.int64)
    cluster_sizes: list[int] = [int(n_samples)]
    centroids: list[cp.ndarray] = [cp.mean(data, axis=0)]

    rng = np.random.RandomState(random_state)

    while len(centroids) < n_clusters:
        largest_cluster_idx = int(np.argmax(cluster_sizes))
        largest_cluster_size = cluster_sizes[largest_cluster_idx]

        if largest_cluster_size < 2:
            break

        cluster_indices = cp.where(cluster_assignments == largest_cluster_idx)[0]
        cluster_data = data[cluster_indices]

        init_centers = greedy_kmeanspp_init(
            cluster_data, n_clusters=2, random_state=rng, x_squared_norms=None
        )

        split_centers, split_labels, _, _ = _fit_minibatch_core(
            data=cluster_data,
            init_centers=init_centers,
            batch_size=min(batch_size, 1024),
            max_iter=max_iter,
            max_no_improvement=10,
            reassignment_ratio=0.01,
            random_state=rng,
        )

        new_cluster_idx = len(centroids)

        for i, global_idx in enumerate(cluster_indices):
            if int(split_labels[i]) == 1:
                cluster_assignments[global_idx] = new_cluster_idx

        split_size_0 = int(cp.count_nonzero(split_labels == 0))
        split_size_1 = int(cp.count_nonzero(split_labels == 1))
        cluster_sizes[largest_cluster_idx] = split_size_0
        cluster_sizes.append(split_size_1)

        new_centroids: list[cp.ndarray] = []
        for c_idx in range(len(centroids)):
            if c_idx == largest_cluster_idx:
                mask = cluster_assignments == c_idx
                cnt = int(cp.count_nonzero(mask))
                if cnt > 0:
                    new_centroids.append(cp.mean(data[mask], axis=0))
                else:
                    new_centroids.append(centroids[c_idx])
            else:
                new_centroids.append(centroids[c_idx])

        mask = cluster_assignments == new_cluster_idx
        cnt = int(cp.count_nonzero(mask))
        if cnt > 0:
            new_centroids.append(cp.mean(data[mask], axis=0))
        else:
            new_centroids.append(split_centers[1])

        centroids = new_centroids

    unique_labels = sorted(set(int(v) for v in cp.asnumpy(cluster_assignments).ravel()))
    if len(unique_labels) != len(centroids):
        centroids = [centroids[label] for label in unique_labels]

    label_mapping = {
        old_label: new_label for new_label, old_label in enumerate(unique_labels)
    }
    labels = np.array(
        [label_mapping[int(label)] for label in cp.asnumpy(cluster_assignments)],
        dtype=np.int64,
    )
    cluster_centers = cp.stack(centroids)

    squared_distances = cp.sum(
        (data[:, cp.newaxis, :] - cluster_centers[cp.newaxis, :, :]) ** 2,
        axis=2,
    )
    inertia = float(squared_distances[cp.arange(n_samples), cp.asarray(labels)].sum())

    return CumlBisectingKMeansModel(cp.asnumpy(cluster_centers), labels, inertia)
