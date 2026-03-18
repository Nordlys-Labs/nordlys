"""CUDA adapter for Spectral using scipy (CPU fallback due to CUDA version mismatch)."""

from __future__ import annotations

from nordlys.clustering.spectral.protocol import SpectralModel

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class CupySpectralModel:
    """Implementation of Spectral using scipy (CPU)."""

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
    affinity: str,
    n_neighbors: int,
    random_state: int,
    embeddings: np.ndarray,
) -> SpectralModel:
    """Fit using scipy (CPU) due to CUDA version incompatibility.

    Implements spectral clustering using:
    1. Build affinity graph (k-nearest neighbors)
    2. Compute normalized graph Laplacian
    3. Compute eigenvectors
    4. Run k-means on eigenvectors
    """
    if affinity != "nearest_neighbors":
        msg = f"Spectral CUDA supports only affinity='nearest_neighbors', got '{affinity}'"
        raise ValueError(msg)

    data = embeddings.astype(np.float32)
    n_samples = data.shape[0]

    if n_samples == 1:
        labels = np.zeros(1, dtype=np.int32)
        centers = data.copy()
        return CupySpectralModel(centers, labels, 1)

    k = min(n_clusters, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(data)
    knn_indices = nn.kneighbors_graph(data, mode="connectivity")
    knn_indices = knn_indices.toarray()

    pairwise_dists = np.linalg.norm(
        data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2
    )

    sigma = float(np.mean(pairwise_dists[knn_indices > 0])) + 1e-10

    weights = np.exp(-(pairwise_dists**2) / (2 * sigma**2))
    knn_mask = knn_indices.astype(np.float32)
    weights = weights * knn_mask
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)

    W = csr_matrix(weights)
    D = csr_matrix(np.eye(n_samples))
    d_sum = np.array(W.sum(axis=1)).flatten() + 1e-10
    D.data = d_sum

    D_inv_sqrt = D.copy()
    D_inv_sqrt.data = 1.0 / (np.sqrt(D_inv_sqrt.data) + 1e-10)

    L = D - W
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    from scipy.sparse.linalg import eigsh

    eigvals, eigvecs = eigsh(L_norm.astype(np.float64), k=k, which="SM")

    idx = np.argsort(np.real(eigvals))
    eigvecs = np.real(eigvecs)[:, idx]
    embedding = eigvecs[:, :n_clusters]

    norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10
    embedding = embedding / norm

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(embedding)
    labels = kmeans.labels_

    centers = []
    for label in range(n_clusters):
        mask = labels == label
        if mask.sum() > 0:
            centers.append(data[mask].mean(axis=0))
        else:
            centers.append(data[0])
    centers = np.array(centers)

    return CupySpectralModel(centers, labels, n_clusters)
