"""Dimensionality reduction protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Reducer(Protocol):
    """Protocol for dimensionality reduction components.

    Implementations should provide sklearn-like fit/transform methods.
    """

    def fit(self, embeddings: np.ndarray) -> "Reducer":
        """Fit the reducer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        ...

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to reduced dimensions.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        ...

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        ...
