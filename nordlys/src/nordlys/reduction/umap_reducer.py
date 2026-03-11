"""UMAP dimensionality reducer."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numba.core.registry import CPUDispatcher
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from scipy.sparse import csr_matrix, issparse
from umap import UMAP

from nordlys.reduction.base import (
    Reducer,
    ReducerConfigModel,
    ReducerStateModel,
    ReductionPayload,
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

    model_state: dict[str, JsonValue] = Field(default_factory=dict)


class SerializedNdArray(BaseModel):
    """Safe ndarray checkpoint payload."""

    kind: Literal["ndarray"]
    dtype: str = Field(min_length=1)
    data: JsonValue

    model_config = ConfigDict(extra="forbid", frozen=True)


class SerializedCSRMatrix(BaseModel):
    """Safe CSR matrix checkpoint payload."""

    kind: Literal["csr_matrix"]
    dtype: str = Field(min_length=1)
    data: list[float]
    indices: list[int]
    indptr: list[int]
    shape: tuple[int, int]

    model_config = ConfigDict(extra="forbid", frozen=True)


class SerializedTuple(BaseModel):
    """Safe tuple checkpoint payload."""

    kind: Literal["tuple"]
    items: list[JsonValue]

    model_config = ConfigDict(extra="forbid", frozen=True)


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)
_IGNORED_UMAP_STATE_KEYS = {
    "_input_distance_func",
    "_inverse_distance_func",
    "_output_distance_func",
}


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
            "UMAPReducer kwargs must be JSON-serializable for checkpoints. "
            f"Invalid keys: {invalid_list}"
        )


def _serialize_umap_state_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return cast(JsonValue, value)
    if isinstance(value, np.generic):
        return cast(JsonValue, value.item())
    if isinstance(value, np.ndarray):
        return SerializedNdArray(
            kind="ndarray",
            dtype=str(value.dtype),
            data=cast(JsonValue, np.asarray(value).tolist()),
        ).model_dump(mode="json")
    if issparse(value):
        matrix = cast(csr_matrix, value).tocsr()
        return SerializedCSRMatrix(
            kind="csr_matrix",
            dtype=str(matrix.dtype),
            data=matrix.data.tolist(),
            indices=matrix.indices.tolist(),
            indptr=matrix.indptr.tolist(),
            shape=matrix.shape,
        ).model_dump(mode="json")
    if isinstance(value, tuple):
        return SerializedTuple(
            kind="tuple",
            items=[_serialize_umap_state_value(item) for item in value],
        ).model_dump(mode="json")
    if isinstance(value, list):
        return [_serialize_umap_state_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("UMAP state dict keys must be strings")
        return {
            str(key): _serialize_umap_state_value(item) for key, item in value.items()
        }
    if isinstance(value, CPUDispatcher):
        raise ValueError("CPUDispatcher values must be filtered before serialization")

    raise ValueError(f"Unsupported UMAP state value type: {type(value).__name__}")


def _deserialize_umap_state_value(value: JsonValue) -> object:
    if isinstance(value, list):
        return [_deserialize_umap_state_value(item) for item in value]
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "ndarray":
            array_value = SerializedNdArray.model_validate(value)
            return np.asarray(array_value.data, dtype=np.dtype(array_value.dtype))
        if kind == "csr_matrix":
            matrix_value = SerializedCSRMatrix.model_validate(value)
            data = np.asarray(matrix_value.data, dtype=np.dtype(matrix_value.dtype))
            indices = np.asarray(matrix_value.indices, dtype=np.int32)
            indptr = np.asarray(matrix_value.indptr, dtype=np.int32)
            rows, cols = matrix_value.shape
            return csr_matrix((data, indices, indptr), shape=(rows, cols))
        if kind == "tuple":
            tuple_value = SerializedTuple.model_validate(value)
            return tuple(
                _deserialize_umap_state_value(item) for item in tuple_value.items
            )
        return {key: _deserialize_umap_state_value(item) for key, item in value.items()}
    return value


@register_reducer
class UMAPReducer(Reducer):
    """UMAP dimensionality reduction wrapper.

    Thin wrapper over umap.UMAP with sensible defaults for text embeddings.

    Example:
        >>> reducer = UMAPReducer(n_components=3)
        >>> reduced = reducer.fit_transform(embeddings)
    """

    kind = "umap"

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
        _validate_json_kwargs(kwargs)
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

    def checkpoint_payload(self) -> ReductionPayload:
        config = UMAPConfig(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            kwargs=self._kwargs,
        )
        if self._model is None:
            raise RuntimeError(
                "Reducer must be fitted before checkpoint serialization."
            )
        model_state = {
            key: _serialize_umap_state_value(value)
            for key, value in self._model.__getstate__().items()
            if key not in _IGNORED_UMAP_STATE_KEYS
        }
        validated_state = {
            key: _JSON_VALUE_ADAPTER.validate_python(value)
            for key, value in model_state.items()
        }
        state = UMAPState(model_state=validated_state)
        return ReductionPayload(
            kind=self.kind,
            config=config.model_dump(mode="json"),
            state=state.model_dump(mode="json"),
        )

    @classmethod
    def from_checkpoint_payload(cls, payload: ReductionPayload) -> "UMAPReducer":
        config = UMAPConfig.model_validate(payload.config)
        state = UMAPState.model_validate(payload.state)
        reducer = cls(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
            **dict(config.kwargs),
        )
        model = reducer._create_model()
        restored_state = model.__getstate__()
        restored_state.update(
            {
                key: _deserialize_umap_state_value(value)
                for key, value in state.model_state.items()
            }
        )
        model.__setstate__(restored_state)
        reducer._model = model
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
