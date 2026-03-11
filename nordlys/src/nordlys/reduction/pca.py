"""PCA dimensionality reducer."""

from __future__ import annotations

import base64
import pickle

import numpy as np
from pydantic import Field, JsonValue
from sklearn.decomposition import PCA

from nordlys.reduction.base import (
    Reducer,
    ReducerConfigModel,
    ReducerStateModel,
    register_reducer,
)


class PCAConfig(ReducerConfigModel):
    """Strict checkpoint schema for PCA constructor config."""

    n_components: int | float = 50
    random_state: int = 42
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class PCAState(ReducerStateModel):
    """Strict checkpoint schema for PCA fitted state."""

    pickle_b64: str = Field(min_length=1)


@register_reducer
class PCAReducer(Reducer):
    """PCA dimensionality reduction wrapper.

    Thin wrapper over sklearn.decomposition.PCA.

    Example:
        >>> reducer = PCAReducer(n_components=50)
        >>> reduced = reducer.fit_transform(embeddings)
    """

    kind = "pca"
    config_model = PCAConfig
    state_model = PCAState

    def __init__(
        self,
        n_components: int | float = 50,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Initialize PCA reducer.

        Args:
            n_components: Number of components to keep (default: 50).
                If float in (0, 1), selects components to explain that variance.
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional arguments passed to PCA
        """
        self.n_components = n_components
        self.random_state = random_state
        self._kwargs = kwargs
        self._model: PCA | None = None

    def _create_model(self) -> PCA:
        """Create the underlying PCA model."""
        return PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "PCAReducer":
        """Fit the reducer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        self._model = self._create_model()
        self._model.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to reduced dimensions.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        if self._model is None:
            raise RuntimeError(
                "Reducer must be fitted before transform. Call fit() first."
            )
        return self._model.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        self._model = self._create_model()
        return self._model.fit_transform(embeddings)

    def checkpoint_config(self) -> PCAConfig:
        return PCAConfig(
            n_components=self.n_components,
            random_state=self.random_state,
            kwargs=self._kwargs,
        )

    def checkpoint_state(self) -> PCAState:
        if self._model is None:
            raise RuntimeError(
                "Reducer must be fitted before checkpoint serialization."
            )
        payload = pickle.dumps(self._model, protocol=pickle.HIGHEST_PROTOCOL)
        return PCAState(pickle_b64=base64.b64encode(payload).decode("ascii"))

    @classmethod
    def from_checkpoint_models(
        cls, config: ReducerConfigModel, state: ReducerStateModel
    ) -> "PCAReducer":
        if not isinstance(config, PCAConfig) or not isinstance(state, PCAState):
            raise TypeError("PCAReducer requires PCAConfig and PCAState payloads")
        reducer = cls(
            n_components=config.n_components,
            random_state=config.random_state,
            **dict(config.kwargs),
        )
        reducer._model = pickle.loads(
            base64.b64decode(state.pickle_b64.encode("ascii"))
        )
        return reducer

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        """Percentage of variance explained by each component."""
        if self._model is None:
            return None
        return self._model.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray | None:
        """Principal axes in feature space."""
        if self._model is None:
            return None
        return self._model.components_

    def __repr__(self) -> str:
        return f"PCAReducer(n_components={self.n_components})"
