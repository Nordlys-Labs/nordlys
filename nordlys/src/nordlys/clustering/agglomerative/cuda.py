"""CUDA adapter for Agglomerative using cupy."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.agglomerative.protocol import AgglomerativeModel


class CupyAgglomerativeModel:
    """GPU implementation of Agglomerative using cupy."""

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
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self._cluster_centers, axis=2
        )
        return distances.argmin(axis=1)


def fit(
    n_clusters: int,
    linkage: str,
    metric: str,
    random_state: int,
    embeddings: np.ndarray,
) -> AgglomerativeModel:
    """Fit using GPU (cupy).

    Implements agglomerative clustering using:
    1. Compute pairwise distances
    2. Build linkage matrix using agglomerative clustering
    3. Cut dendrogram to get n_clusters
    """
    import cupy as cp

    if linkage not in ("ward", "complete", "average", "single"):
        msg = f"Agglomerative CUDA supports linkage='ward', 'complete', 'average', 'single', got '{linkage}'"
        raise ValueError(msg)

    if linkage == "ward" and metric != "euclidean":
        msg = "Ward linkage requires euclidean metric"
        raise ValueError(msg)

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples, n_features = data.shape

    if linkage == "ward":
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        from scipy.spatial.distance import pdist

        pdist_result = pdist(cp.asnumpy(data), metric="euclidean")
        Z = scipy_linkage(pdist_result, method=linkage)
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    else:
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        from scipy.spatial.distance import pdist

        pdist_result = pdist(cp.asnumpy(data), metric=metric)
        Z = scipy_linkage(pdist_result, method=linkage)
        from scipy.cluster.hierarchy import fcluster

        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

    labels = cp.asnumpy(labels)

    centers = []
    for label in range(n_clusters):
        mask = labels == label
        if mask.sum() > 0:
            centers.append(embeddings[mask].mean(axis=0))
        else:
            centers.append(embeddings[0])
    centers = np.array(centers)

    return CupyAgglomerativeModel(centers, labels, n_clusters)
