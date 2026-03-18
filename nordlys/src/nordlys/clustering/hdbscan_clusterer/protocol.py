"""Protocol for fitted HDBSCAN-like models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HDBSCANModel(Protocol):
    """Protocol for fitted HDBSCAN-like models.

    Both CPU (hdbscan) and CUDA (cuml) implementations must satisfy this.
    """

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""
        ...

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit of shape (n_samples,)."""
        ...

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding noise)."""
        ...

    @property
    def probabilities_(self) -> np.ndarray:
        """Cluster membership probabilities of shape (n_samples,)."""
        ...

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        ...
