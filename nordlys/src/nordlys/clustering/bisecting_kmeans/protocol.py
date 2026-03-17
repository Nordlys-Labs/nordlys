"""Protocol for fitted BisectingKMeans-like models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BisectingKMeansModel(Protocol):
    """Protocol for fitted BisectingKMeans-like models."""

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit of shape (n_samples,)."""

    @property
    def inertia_(self) -> float:
        """Sum of squared distances to closest centroid."""

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
