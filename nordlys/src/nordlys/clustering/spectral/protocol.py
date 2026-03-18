"""Protocol for fitted Spectral-like models."""

from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpectralModel(Protocol):
    """Protocol for fitted Spectral-like models.

    Both CPU (sklearn) and CUDA (custom) implementations must satisfy this.
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
        """Number of clusters."""
        ...

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        ...
