"""CUDA adapter for Spectral using cupy."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.spectral.protocol import SpectralModel


class CupySpectralModel:
    """GPU implementation of Spectral using cupy."""

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
    affinity: str,
    n_neighbors: int,
    random_state: int,
    embeddings: np.ndarray,
) -> SpectralModel:
    """Fit using GPU (cupy).

    Implements spectral clustering using:
    1. Build affinity graph (k-nearest neighbors)
    2. Compute normalized graph Laplacian
    3. Compute eigenvectors
    4. Run k-means on eigenvectors
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import eye as scipy_eye

    if affinity != "nearest_neighbors":
        msg = f"Spectral CUDA supports only affinity='nearest_neighbors', got '{affinity}'"
        raise ValueError(msg)

    n_samples = embeddings.shape[0]

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(embeddings)
    knn_indices = nn.kneighbors_graph(embeddings, mode="connectivity")
    knn_indices = knn_indices.toarray()

    dists = np.linalg.norm(
        embeddings[:, np.newaxis] - embeddings[np.newaxis, :], axis=2
    )

    sigma = np.mean(dists[knn_indices.nonzero()]) + 1e-10

    weights = np.exp(-(dists**2) / (2 * sigma**2))
    weights[knn_indices == 0] = 0
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)

    W = csr_matrix(weights)
    D = scipy_eye(n_samples, format="csr")
    D.data = np.array(W.sum(axis=1)).flatten() + 1e-10

    D_inv_sqrt = D.copy()
    D_inv_sqrt.data = 1.0 / np.sqrt(D_inv_sqrt.data + 1e-10)

    L = D - W
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    k = min(n_clusters, n_samples - 1)
    eigvals, eigvecs = eigsh(L_norm.astype(np.float64), k=k, which="SM")

    idx = np.argsort(eigvals.real)
    eigvecs = eigvecs.real[:, idx]
    embedding = eigvecs[:, :n_clusters]

    norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10
    embedding = embedding / norm

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(embedding)
    labels = kmeans.labels_

    centers = []
    for label in range(n_clusters):
        mask = labels == label
        if mask.sum() > 0:
            centers.append(embeddings[mask].mean(axis=0))
        else:
            centers.append(embeddings[0])
    centers = np.array(centers)

    return CupySpectralModel(centers, labels, n_clusters)
