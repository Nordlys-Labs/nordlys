"""Dimensionality reduction abstractions and checkpoint restoration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

JSONDict = dict[str, JsonValue]


class ReducerConfigModel(BaseModel):
    """Base class for reducer config schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReducerStateModel(BaseModel):
    """Base class for reducer state schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReductionPayload(BaseModel):
    """Serialized reducer payload stored in checkpoints."""

    kind: str = Field(min_length=1)
    config: JSONDict
    state: JSONDict

    model_config = ConfigDict(extra="forbid", frozen=True)


_REDUCER_REGISTRY: dict[str, type["Reducer"]] = {}


class Reducer(ABC):
    """Abstract base class for checkpointable reducers."""

    kind: ClassVar[str]
    config_model: ClassVar[type[ReducerConfigModel]]
    state_model: ClassVar[type[ReducerStateModel]]

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> "Reducer":
        """Fit the reducer on embeddings."""

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings into reduced space."""

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)

    @abstractmethod
    def checkpoint_config(self) -> ReducerConfigModel:
        """Return constructor-time reducer config."""

    @abstractmethod
    def checkpoint_state(self) -> ReducerStateModel:
        """Return fitted reducer state."""

    @classmethod
    @abstractmethod
    def from_checkpoint_models(
        cls, config: ReducerConfigModel, state: ReducerStateModel
    ) -> "Reducer":
        """Restore a fitted reducer from validated checkpoint models."""

    def checkpoint_payload(self) -> ReductionPayload:
        """Return serialized reducer payload for checkpoints."""
        return ReductionPayload(
            kind=self.kind,
            config=self.checkpoint_config().model_dump(mode="json"),
            state=self.checkpoint_state().model_dump(mode="json"),
        )


def register_reducer(reducer_cls: type[Reducer]) -> type[Reducer]:
    """Register a reducer class for checkpoint restoration."""
    if not reducer_cls.kind:
        raise ValueError(
            f"Reducer class {reducer_cls.__name__} must define a non-empty kind"
        )
    _REDUCER_REGISTRY[reducer_cls.kind] = reducer_cls
    return reducer_cls


def restore_reducer(payload: ReductionPayload | None) -> Reducer | None:
    """Restore a reducer from a checkpoint payload."""
    if payload is None:
        return None

    reducer_cls = _REDUCER_REGISTRY.get(payload.kind)
    if reducer_cls is None:
        raise ValueError(f"Unknown reducer kind '{payload.kind}' in checkpoint")

    config = reducer_cls.config_model.model_validate(payload.config)
    state = reducer_cls.state_model.model_validate(payload.state)
    return reducer_cls.from_checkpoint_models(config, state)
