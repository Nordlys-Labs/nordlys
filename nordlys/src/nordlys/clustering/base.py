"""Clusterer protocol for clustering algorithms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Clusterer(Protocol):
    """Protocol for clustering components.

    Implementations should provide sklearn-like fit/predict methods
    and expose cluster centers and labels as properties.
    """

    def fit(self, embeddings: np.ndarray) -> "Clusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        ...

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        ...

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""
        ...

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        ...

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found."""
        ...
