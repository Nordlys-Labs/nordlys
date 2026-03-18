"""CPU adapter for MiniBatch KMeans."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from nordlys.clustering.minibatch_kmeans.protocol import MiniBatchKMeansModel


class SklearnMiniBatchKMeansModel:
    """Adapter for sklearn MiniBatchKMeans."""

    def __init__(self, model: MiniBatchKMeans) -> None:
        self._model = model

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._model.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        return self._model.labels_

    @property
    def inertia_(self) -> float:
        return float(self._model.inertia_)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)


def create_sklearn_model(
    n_clusters: int,
    max_iter: int,
    batch_size: int,
    random_state: int,
    init_size: int | None,
    max_no_improvement: int,
    reassignment_ratio: float,
    **kwargs,
) -> SklearnMiniBatchKMeansModel:
    """Create a sklearn MiniBatchKMeans model."""
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=random_state,
        init_size=init_size,
        max_no_improvement=max_no_improvement,
        reassignment_ratio=reassignment_ratio,
        **kwargs,
    )
    return SklearnMiniBatchKMeansModel(model)


def fit(
    model: SklearnMiniBatchKMeansModel,
    embeddings: np.ndarray,
) -> MiniBatchKMeansModel:
    """Fit the sklearn model and return the adapter."""
    model._model.fit(embeddings)
    return model
