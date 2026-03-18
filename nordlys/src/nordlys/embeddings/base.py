"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nordlys.checkpoint_types import EmbeddingConfig


class Embedder(ABC):
    """Abstract embedder used by Trainer."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode input texts into a 2D embedding matrix."""

    @abstractmethod
    def checkpoint_config(self) -> EmbeddingConfig:
        """Return embedding config stored in checkpoints."""
