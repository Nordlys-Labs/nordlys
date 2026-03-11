"""PCA dimensionality reducer."""

from __future__ import annotations

import numpy as np
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
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

    components: list[list[float]]
    mean: list[float]
    explained_variance: list[float]
    explained_variance_ratio: list[float]
    singular_values: list[float]
    n_components_fitted: int
    n_features_in: int
    n_samples: int
    noise_variance: float


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


def _validate_json_kwargs(kwargs: dict[str, object]) -> None:
    invalid_keys: list[str] = []
    for key, value in kwargs.items():
        try:
            _JSON_VALUE_ADAPTER.validate_python(value)
        except ValidationError:
            invalid_keys.append(key)

    if invalid_keys:
        invalid_list = ", ".join(sorted(invalid_keys))
        raise ValueError(
            "PCAReducer kwargs must be JSON-serializable for checkpoints. "
            f"Invalid keys: {invalid_list}"
        )


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
        _validate_json_kwargs(kwargs)
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
        return PCAState(
            components=np.asarray(self._model.components_, dtype=np.float64).tolist(),
            mean=np.asarray(self._model.mean_, dtype=np.float64).tolist(),
            explained_variance=np.asarray(
                self._model.explained_variance_, dtype=np.float64
            ).tolist(),
            explained_variance_ratio=np.asarray(
                self._model.explained_variance_ratio_, dtype=np.float64
            ).tolist(),
            singular_values=np.asarray(
                self._model.singular_values_, dtype=np.float64
            ).tolist(),
            n_components_fitted=int(self._model.n_components_),
            n_features_in=int(self._model.components_.shape[1]),
            n_samples=int(self._model.n_samples_),
            noise_variance=float(self._model.noise_variance_),
        )

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
        model = reducer._create_model()
        model.components_ = np.asarray(state.components, dtype=np.float64)
        model.mean_ = np.asarray(state.mean, dtype=np.float64)
        model.explained_variance_ = np.asarray(
            state.explained_variance, dtype=np.float64
        )
        model.explained_variance_ratio_ = np.asarray(
            state.explained_variance_ratio, dtype=np.float64
        )
        model.singular_values_ = np.asarray(state.singular_values, dtype=np.float64)
        model.n_components_ = state.n_components_fitted
        model.n_features_in_ = state.n_features_in  # ty: ignore[unresolved-attribute]
        model.n_samples_ = state.n_samples
        model.noise_variance_ = state.noise_variance
        reducer._model = model
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
