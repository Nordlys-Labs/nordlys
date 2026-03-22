"""GPU adapter for Spectral using CuPy and cuML."""

from __future__ import annotations

import logging

import numpy as np

import cupy as cp
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.spatial import KDTree
from cupyx.scipy.sparse.linalg import eigsh

from nordlys.clustering.spectral.protocol import SpectralModel

logger = logging.getLogger(__name__)


class CupySpectralModel:
    """GPU spectral clustering model using CuPy arrays.

    Model state is kept on the GPU for efficiency.  Results returned through
    the protocol interface (labels_, cluster_centers_) are converted to numpy
    so callers can use them with sklearn metrics and standard numpy operations.
    """

    def __init__(
        self,
        cluster_centers: cp.ndarray,
        labels: cp.ndarray,
        n_clusters: int,
    ) -> None:
        self._cluster_centers = cluster_centers
        self._labels = labels
        self._n_clusters = n_clusters

    @property
    def cluster_centers_(self) -> np.ndarray:
        return cp.asnumpy(self._cluster_centers)

    @property
    def labels_(self) -> np.ndarray:
        return cp.asnumpy(self._labels)

    @property
    def n_clusters_(self) -> int:
        return self._n_clusters

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        data_gpu = cp.asarray(embeddings)
        centers_gpu = self._cluster_centers
        distances = cp.linalg.norm(
            data_gpu[:, cp.newaxis, :] - centers_gpu[cp.newaxis, :, :],
            axis=2,
        )
        return cp.asnumpy(distances.argmin(axis=1))


def fit(
    n_clusters: int,
    affinity: str,
    n_neighbors: int,
    random_state: int,
    embeddings: cp.ndarray,
) -> SpectralModel:
    """GPU spectral clustering via k-NN affinity graph and normalized Laplacian.

    Algorithm:
        1. Build k-NN graph on GPU using CuPy KDTree.
        2. Weight edges with Gaussian kernel using per-edge sigma from kth-NN distance.
        3. Symmetrize the directed k-NN graph.
        4. Compute normalized graph Laplacian on GPU.
        5. Compute k smallest eigenvectors via CuPy sparse eigensolver.
        6. Cluster eigenvectors with cuML KMeans on GPU.
        7. Return centroids as mean of cluster members in original space.

    Args:
        n_clusters: Number of clusters.
        affinity: Must be "nearest_neighbors" (other affinities not supported on GPU).
        n_neighbors: Number of nearest neighbors per side.
        random_state: Random seed for KMeans.
        embeddings: Input array of shape (n_samples, n_features) on GPU.

    Returns:
        SpectralModel with GPU arrays for labels and centroids.
    """
    if affinity != "nearest_neighbors":
        msg = f"Spectral CUDA supports only affinity='nearest_neighbors', got '{affinity}'"
        raise ValueError(msg)

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples = data.shape[0]

    if n_samples == 1:
        labels = cp.zeros(1, dtype=cp.int32)
        centers = data.copy()
        return CupySpectralModel(centers, labels, 1)

    k = min(n_clusters, n_samples - 1)

    tree = KDTree(data)
    distances, indices = tree.query(data, k=n_neighbors + 1)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    sigma = float(cp.mean(distances)) + 1e-10
    weights = cp.exp(-(distances**2) / (2 * sigma**2))

    rows = cp.repeat(cp.arange(n_samples, dtype=cp.int32), n_neighbors)
    cols = indices.ravel()
    vals = weights.ravel()
    W = cp_sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    W = W + W.T
    W_data = W.data * 0.5
    W = cp_sparse.csr_matrix(
        (W_data, W.indices, W.indptr), shape=(n_samples, n_samples)
    )

    d_sum = cp.asarray(W.sum(axis=1)).flatten() + 1e-10
    D = cp_sparse.diags(d_sum)
    D_inv_sqrt = D.copy()
    D_inv_sqrt.data = 1.0 / (cp.sqrt(D_inv_sqrt.data) + 1e-10)

    L = D - W
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigvals, eigvecs = eigsh(L_norm, k=k, which="SA")

    idx = cp.argsort(cp.real(eigvals))
    eigvecs = cp.real(eigvecs)[:, idx]
    embedding = eigvecs[:, :n_clusters]

    row_norm = cp.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10
    embedding = embedding / row_norm

    embedding_np = embedding.get()
    from cuml.cluster import KMeans as cuml_KMeans

    kmeans = cuml_KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init="auto"
    )
    kmeans.fit(embedding_np)
    kmeans_labels = kmeans.labels_
    labels = cp.asarray(kmeans_labels, dtype=cp.int32)

    centers = []
    for label in range(n_clusters):
        mask = labels == label
        if int(cp.count_nonzero(mask)) > 0:
            centers.append(data[mask].mean(axis=0))
        else:
            centers.append(data[0])
    centers = cp.stack(centers) if centers else cp.empty((0, data.shape[1]))

    return CupySpectralModel(centers, labels, n_clusters)
