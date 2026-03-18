"""CUDA adapter for KMeans."""

from __future__ import annotations

import numpy as np

from nordlys.clustering.kmeans.protocol import KMeansModel


class CumlKMeansModel:
    """Adapter for cuML KMeans."""

    def __init__(
        self,
        cluster_centers_: np.ndarray,
        labels_: np.ndarray,
        inertia: float,
        n_iter: int,
    ) -> None:
        self._cluster_centers = cluster_centers_
        self._labels = labels_
        self._inertia = inertia
        self._n_iter = n_iter

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def inertia_(self) -> float:
        return self._inertia

    @property
    def n_iter_(self) -> int:
        return self._n_iter

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        data = np.asarray(embeddings, dtype=np.float32)
        distances = np.sqrt(
            np.sum(
                (data[:, np.newaxis, :] - self._cluster_centers[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        return np.argmin(distances, axis=1)


def fit(
    n_clusters: int,
    max_iter: int,
    n_init: int,
    random_state: int,
    embeddings: np.ndarray,
) -> KMeansModel:
    """Fit using CUDA (cuML)."""
    import cuml

    data = np.asarray(embeddings, dtype=np.float32)

    cuml_model = cuml.KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        output_type="numpy",
    )
    cuml_model.fit(data)

    return CumlKMeansModel(
        cluster_centers_=cuml_model.cluster_centers_,
        labels_=cuml_model.labels_,
        inertia=float(cuml_model.inertia_),
        n_iter=int(cuml_model.n_iter_),
    )
