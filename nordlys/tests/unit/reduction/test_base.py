"""Unit tests for reduction base helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys.reduction.base import (
    Reducer,
    ReducerConfigModel,
    ReducerStateModel,
    register_reducer,
)


class _Config(ReducerConfigModel):
    value: int


class _State(ReducerStateModel):
    value: int


@register_reducer
class _BaseTestReducer(Reducer):
    kind = "base-test-reducer"
    config_model = _Config
    state_model = _State

    def fit(self, embeddings: np.ndarray) -> "_BaseTestReducer":
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings

    def checkpoint_config(self) -> ReducerConfigModel:
        return _Config(value=1)

    def checkpoint_state(self) -> ReducerStateModel:
        return _State(value=1)

    @classmethod
    def from_checkpoint_models(
        cls, config: ReducerConfigModel, state: ReducerStateModel
    ) -> "_BaseTestReducer":
        return cls()


def test_register_reducer_rejects_duplicate_kind() -> None:
    class DuplicateReducer(Reducer):
        kind = "base-test-reducer"
        config_model = _Config
        state_model = _State

        def fit(self, embeddings: np.ndarray) -> "DuplicateReducer":
            return self

        def transform(self, embeddings: np.ndarray) -> np.ndarray:
            return embeddings

        def checkpoint_config(self) -> ReducerConfigModel:
            return _Config(value=1)

        def checkpoint_state(self) -> ReducerStateModel:
            return _State(value=1)

        @classmethod
        def from_checkpoint_models(
            cls, config: ReducerConfigModel, state: ReducerStateModel
        ) -> "DuplicateReducer":
            return cls()

    with pytest.raises(ValueError, match="already registered"):
        register_reducer(DuplicateReducer)
