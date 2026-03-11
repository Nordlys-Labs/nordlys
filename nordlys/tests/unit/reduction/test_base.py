"""Unit tests for reduction base helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys.reduction.base import (
    Reducer,
    ReducerConfigModel,
    ReducerStateModel,
    ReductionPayload,
    register_reducer,
)


class _Config(ReducerConfigModel):
    value: int


class _State(ReducerStateModel):
    value: int


@register_reducer
class _BaseTestReducer(Reducer):
    kind = "base-test-reducer"

    def fit(self, embeddings: np.ndarray) -> "_BaseTestReducer":
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings

    def checkpoint_payload(self) -> ReductionPayload:
        return ReductionPayload(
            kind=self.kind,
            config=_Config(value=1).model_dump(mode="json"),
            state=_State(value=1).model_dump(mode="json"),
        )

    @classmethod
    def from_checkpoint_payload(cls, payload: ReductionPayload) -> "_BaseTestReducer":
        return cls()


def test_register_reducer_rejects_duplicate_kind() -> None:
    class DuplicateReducer(Reducer):
        kind = "base-test-reducer"

        def fit(self, embeddings: np.ndarray) -> "DuplicateReducer":
            return self

        def transform(self, embeddings: np.ndarray) -> np.ndarray:
            return embeddings

        def checkpoint_payload(self) -> ReductionPayload:
            return ReductionPayload(
                kind=self.kind,
                config=_Config(value=1).model_dump(mode="json"),
                state=_State(value=1).model_dump(mode="json"),
            )

        @classmethod
        def from_checkpoint_payload(
            cls, payload: ReductionPayload
        ) -> "DuplicateReducer":
            return cls()

    with pytest.raises(ValueError, match="already registered"):
        register_reducer(DuplicateReducer)
