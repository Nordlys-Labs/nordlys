"""CUDA adapter for Agglomerative using scipy (CPU fallback due to CUDA version mismatch)."""

from __future__ import annotations

from nordlys.clustering.agglomerative.protocol import AgglomerativeModel

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage as linkage_func
from scipy.spatial.distance import pdist


class CupyAgglomerativeModel:
    """Implementation of Agglomerative using scipy (CPU)."""

    def __init__(
        self,
        cluster_centers: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        self._cluster_centers = cluster_centers
        self._labels = labels
        self._n_clusters = n_clusters

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def n_clusters_(self) -> int:
        return self._n_clusters

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        data = embeddings
        centers = self._cluster_centers

        distances = np.linalg.norm(
            data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        return labels


def fit(
    n_clusters: int,
    linkage: str,
    metric: str,
    random_state: int,
    embeddings: np.ndarray,
) -> AgglomerativeModel:
    """Fit using scipy (CPU) due to CUDA version incompatibility.

    Implements agglomerative clustering using:
    1. Compute pairwise distances
    2. Build linkage matrix
    3. Cut dendrogram to get n_clusters
    """
    if linkage not in ("ward", "complete", "average", "single"):
        msg = f"Agglomerative CUDA supports linkage='ward', 'complete', 'average', 'single', got '{linkage}'"
        raise ValueError(msg)

    if linkage == "ward" and metric != "euclidean":
        msg = "Ward linkage requires euclidean metric"
        raise ValueError(msg)

    data = embeddings.astype(np.float32)
    n_samples, n_features = data.shape

    if linkage == "ward":
        pdist_result = pdist(data, metric="euclidean")
        Z = linkage_func(pdist_result, method=linkage)
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    else:
        pdist_result = pdist(data, metric=str(metric))
        Z = linkage_func(pdist_result, method=linkage)
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    centers = []
    for label in range(n_clusters):
        mask = labels == label
        if mask.sum() > 0:
            centers.append(data[mask].mean(axis=0))
        else:
            centers.append(data[0])
    centers = np.array(centers)

    return CupyAgglomerativeModel(centers, labels, n_clusters)
