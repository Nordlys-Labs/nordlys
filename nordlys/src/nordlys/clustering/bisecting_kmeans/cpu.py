"""CPU adapter for Bisecting KMeans."""

from __future__ import annotations

from nordlys.clustering.bisecting_kmeans.protocol import BisectingKMeansModel

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class SklearnBisectingKMeansModel:
    """Result model for BisectingKMeans."""

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
        distances = np.sqrt(
            np.sum(
                (embeddings[:, np.newaxis, :] - self._cluster_centers[np.newaxis, :, :])
                ** 2,
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
    init_size: int | None = None,
    max_no_improvement: int = 10,
) -> BisectingKMeansModel:
    """Fit using CPU (sklearn)."""
    n_samples = embeddings.shape[0]

    if n_clusters <= 1:
        labels = np.zeros(n_samples, dtype=np.int64)
        cluster_centers = embeddings.mean(axis=0, keepdims=True)
        inertia = float(np.sum((embeddings - cluster_centers) ** 2))
        return SklearnBisectingKMeansModel(cluster_centers, labels, inertia)

    cluster_assignments = np.zeros(n_samples, dtype=np.int64)
    cluster_sizes = [n_samples]
    centroids = [embeddings.mean(axis=0)]

    while len(centroids) < n_clusters:
        largest_cluster_idx = np.argmax(cluster_sizes)
        largest_cluster_size = cluster_sizes[largest_cluster_idx]

        if largest_cluster_size < 2:
            break

        cluster_indices = np.where(cluster_assignments == largest_cluster_idx)[0]
        cluster_data = embeddings[cluster_indices]

        split_model = MiniBatchKMeans(
            n_clusters=2,
            max_iter=max_iter,
            batch_size=min(batch_size, 1024),
            random_state=random_state,
            init="k-means++",
        )
        split_model.fit(cluster_data)
        split_labels = split_model.predict(cluster_data)

        new_cluster_idx = len(centroids)

        for i, global_idx in enumerate(cluster_indices):
            if split_labels[i] == 1:
                cluster_assignments[global_idx] = new_cluster_idx

        split_size_0 = np.sum(split_labels == 0)
        split_size_1 = np.sum(split_labels == 1)
        cluster_sizes[largest_cluster_idx] = split_size_0
        cluster_sizes.append(split_size_1)

        new_centroids = []
        for c_idx in range(len(centroids)):
            if c_idx == largest_cluster_idx:
                mask = cluster_assignments == c_idx
                if np.sum(mask) > 0:
                    new_centroids.append(embeddings[mask].mean(axis=0))
                else:
                    new_centroids.append(centroids[c_idx])
            else:
                new_centroids.append(centroids[c_idx])

        mask = cluster_assignments == new_cluster_idx
        if np.sum(mask) > 0:
            new_centroids.append(embeddings[mask].mean(axis=0))
        else:
            new_centroids.append(split_model.cluster_centers_[1])

        centroids = new_centroids

    unique_labels = sorted(set(cluster_assignments))
    if len(unique_labels) != len(centroids):
        centroids = [centroids[label] for label in unique_labels]

    label_mapping = {
        old_label: new_label for new_label, old_label in enumerate(unique_labels)
    }
    labels = np.array(
        [label_mapping[label] for label in cluster_assignments], dtype=np.int64
    )
    cluster_centers = np.stack(centroids)

    squared_distances = np.sum(
        (embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    inertia = float(squared_distances[np.arange(n_samples), labels].sum())

    return SklearnBisectingKMeansModel(cluster_centers, labels, inertia)
