"""UMAP dimensionality reducer."""

from __future__ import annotations

import base64
import pickle

import numpy as np
from pydantic import Field, JsonValue
from umap import UMAP

from nordlys.reduction.base import (
    Reducer,
    ReducerConfigModel,
    ReducerStateModel,
    register_reducer,
)


class UMAPConfig(ReducerConfigModel):
    """Strict checkpoint schema for UMAP constructor config."""

    n_components: int = 3
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class UMAPState(ReducerStateModel):
    """Strict checkpoint schema for UMAP fitted state."""

    pickle_b64: str = Field(min_length=1)


@register_reducer
class UMAPReducer(Reducer):
    """UMAP dimensionality reduction wrapper.

    Thin wrapper over umap.UMAP with sensible defaults for text embeddings.

    Example:
        >>> reducer = UMAPReducer(n_components=3)
        >>> reduced = reducer.fit_transform(embeddings)
    """

    kind = "umap"
    config_model = UMAPConfig
    state_model = UMAPState

    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Initialize UMAP reducer.

        Args:
            n_components: Dimension of the reduced space (default: 3)
            n_neighbors: Size of local neighborhood (default: 15)
            min_dist: Minimum distance between points (default: 0.1)
            metric: Distance metric (default: "cosine")
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional arguments passed to UMAP
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self._kwargs = kwargs
        self._model: UMAP | None = None

    def _create_model(self) -> UMAP:
        """Create the underlying UMAP model."""
        return UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "UMAPReducer":
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

    def checkpoint_config(self) -> UMAPConfig:
        return UMAPConfig(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            kwargs=self._kwargs,
        )

    def checkpoint_state(self) -> UMAPState:
        if self._model is None:
            raise RuntimeError(
                "Reducer must be fitted before checkpoint serialization."
            )
        payload = pickle.dumps(self._model, protocol=pickle.HIGHEST_PROTOCOL)
        return UMAPState(pickle_b64=base64.b64encode(payload).decode("ascii"))

    @classmethod
    def from_checkpoint_models(
        cls, config: ReducerConfigModel, state: ReducerStateModel
    ) -> "UMAPReducer":
        if not isinstance(config, UMAPConfig) or not isinstance(state, UMAPState):
            raise TypeError("UMAPReducer requires UMAPConfig and UMAPState payloads")
        reducer = cls(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
            **dict(config.kwargs),
        )
        reducer._model = pickle.loads(
            base64.b64decode(state.pickle_b64.encode("ascii"))
        )
        return reducer

    @property
    def embedding_(self) -> np.ndarray | None:
        """The embedding of the training data."""
        if self._model is None:
            return None
        return self._model.embedding_

    def __repr__(self) -> str:
        return (
            f"UMAPReducer(n_components={self.n_components}, "
            f"n_neighbors={self.n_neighbors}, metric='{self.metric}')"
        )
