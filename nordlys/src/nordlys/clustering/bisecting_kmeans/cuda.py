"""GPU adapter for Bisecting KMeans using cuML and cupy."""

from __future__ import annotations

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
    """Fit using CUDA (cuML + cupy)."""
    import cupy as cp
    import cuml

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples = data.shape[0]

    if n_clusters <= 1:
        labels = cp.zeros(n_samples, dtype=cp.int64)
        cluster_centers = cp.mean(data, axis=0, keepdims=True)
        inertia = float(cp.sum((data - cluster_centers) ** 2))
        return CumlBisectingKMeansModel(
            cp.asnumpy(cluster_centers), cp.asnumpy(labels), inertia
        )

    cluster_assignments = cp.zeros(n_samples, dtype=cp.int64)
    cluster_sizes = [int(n_samples)]
    centroids = [cp.mean(data, axis=0)]

    while len(centroids) < n_clusters:
        largest_cluster_idx = int(np.argmax(cluster_sizes))
        largest_cluster_size = cluster_sizes[largest_cluster_idx]

        if largest_cluster_size < 2:
            break

        cluster_indices = cp.where(cluster_assignments == largest_cluster_idx)[0]
        cluster_data = data[cluster_indices]

        init_model = cuml.KMeans(
            n_clusters=2,
            max_iter=1,
            n_init=10,
            random_state=random_state,
        )
        init_model.fit(cp.asarray(cluster_data))
        split_centroids = cp.asarray(init_model.cluster_centers_)

        for iteration in range(max_iter):
            rng = np.random.RandomState(random_state + iteration)
            n_cluster = int(cluster_data.shape[0])
            batch_indices = rng.choice(
                n_cluster, size=min(batch_size, n_cluster), replace=False
            )
            batch = cluster_data[batch_indices]

            squared_distances = cp.sum(
                (batch[:, cp.newaxis, :] - split_centroids[cp.newaxis, :, :]) ** 2,
                axis=2,
            )
            batch_labels = cp.argmin(squared_distances, axis=1)

            for k in range(2):
                mask = batch_labels == k
                if int(cp.sum(mask)) > 0:
                    learning_rate = 0.01
                    split_centroids[k] = (1 - learning_rate) * split_centroids[
                        k
                    ] + learning_rate * cp.mean(batch[mask], axis=0)

        final_squared_distances = cp.sum(
            (cluster_data[:, cp.newaxis, :] - split_centroids[cp.newaxis, :, :]) ** 2,
            axis=2,
        )
        split_labels = cp.argmin(final_squared_distances, axis=1)

        new_cluster_idx = len(centroids)

        for i, global_idx in enumerate(cluster_indices):
            if int(split_labels[i]) == 1:
                cluster_assignments[global_idx] = new_cluster_idx

        split_size_0 = int(cp.sum(split_labels == 0))
        split_size_1 = int(cp.sum(split_labels == 1))
        cluster_sizes[largest_cluster_idx] = split_size_0
        cluster_sizes.append(split_size_1)

        new_centroids = []
        for c_idx in range(len(centroids)):
            if c_idx == largest_cluster_idx:
                mask = cluster_assignments == c_idx
                if int(cp.sum(mask)) > 0:
                    new_centroids.append(cp.mean(data[mask], axis=0))
                else:
                    new_centroids.append(centroids[c_idx])
            else:
                new_centroids.append(centroids[c_idx])

        mask = cluster_assignments == new_cluster_idx
        if int(cp.sum(mask)) > 0:
            new_centroids.append(cp.mean(data[mask], axis=0))
        else:
            new_centroids.append(split_centroids[1])

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
