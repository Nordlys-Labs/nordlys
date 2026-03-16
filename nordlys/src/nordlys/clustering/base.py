"""Clusterer base class for clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Clusterer(ABC):
    """Abstract base class for clustering components.

    Implementations must provide sklearn-like fit/predict methods
    and expose cluster centers and labels as properties.
    """

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> "Clusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        ...

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        ...

    @property
    @abstractmethod
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""
        ...

    @property
    @abstractmethod
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        ...

    @property
    @abstractmethod
    def n_clusters_(self) -> int:
        """Number of clusters found."""
        ...

    @property
    def inertia_(self) -> float | None:
        """Inertia (sum of squared distances to closest cluster center), or None if not supported."""
        return None
