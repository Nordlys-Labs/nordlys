"""CPU adapter for KMeans."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from nordlys.clustering.kmeans.protocol import KMeansModel


class SklearnKMeansModel:
    """Adapter for sklearn KMeans."""

    def __init__(self, model: KMeans) -> None:
        self._model = model

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._model.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        assert self._model.labels_ is not None
        return self._model.labels_

    @property
    def inertia_(self) -> float:
        assert self._model.inertia_ is not None
        return float(self._model.inertia_)

    @property
    def n_iter_(self) -> int:
        return int(self._model.n_iter_)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)


def create_sklearn_model(
    n_clusters: int,
    max_iter: int,
    n_init: int,
    random_state: int,
    algorithm: str,
    **kwargs,
) -> SklearnKMeansModel:
    """Create and fit a sklearn KMeans model."""
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm,
        **kwargs,
    )
    return SklearnKMeansModel(model)


def fit(
    model: SklearnKMeansModel,
    embeddings: np.ndarray,
) -> KMeansModel:
    """Fit the sklearn model and return the adapter."""
    model._model.fit(embeddings)
    return model
