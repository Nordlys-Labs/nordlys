"""CPU adapter for GMM."""

from __future__ import annotations

from typing import cast

from nordlys.clustering.gmm.protocol import GMMModel

import numpy as np
from sklearn.mixture import GaussianMixture


class SklearnGMMModel:
    """Adapter for sklearn GaussianMixture (post-fit only)."""

    def __init__(
        self, model: GaussianMixture, embeddings: np.ndarray | None = None
    ) -> None:
        self._model = model
        self._embeddings = embeddings

    @property
    def cluster_centers_(self) -> np.ndarray:
        return cast(np.ndarray, self._model.means_)

    @property
    def labels_(self) -> np.ndarray:
        if self._embeddings is not None:
            return self._model.predict(self._embeddings)
        raise RuntimeError(
            "Embeddings not stored; call fit() to store them or use predict() instead."
        )

    @property
    def n_components_(self) -> int:
        return self._model.n_components

    @property
    def weights_(self) -> np.ndarray:
        return cast(np.ndarray, self._model.weights_)

    @property
    def covariances_(self) -> np.ndarray:
        return cast(np.ndarray, self._model.covariances_)

    @property
    def converged_(self) -> bool:
        return self._model.converged_

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(embeddings)


def create_sklearn_model(
    n_components: int,
    covariance_type: str,
    max_iter: int,
    n_init: int,
    random_state: int,
    **kwargs: object,
) -> GaussianMixture:
    """Create an unfitted sklearn GMM model."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        **kwargs,
    )
    return model


def fit(
    model: GaussianMixture,
    embeddings: np.ndarray,
) -> GMMModel:
    """Fit the sklearn model and return the adapter."""
    model.fit(embeddings)
    return SklearnGMMModel(model, embeddings)
