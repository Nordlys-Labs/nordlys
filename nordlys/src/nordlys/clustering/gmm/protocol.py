"""Protocol for fitted GMM-like models."""

from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class GMMModel(Protocol):
    """Protocol for fitted GMM-like models.

    Both CPU (sklearn) and CUDA (custom) implementations must satisfy this.
    """

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers (means) of shape (n_components, n_features)."""
        ...

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit of shape (n_samples,)."""
        ...

    @property
    def n_components_(self) -> int:
        """Number of mixture components."""
        ...

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        ...

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities."""
        ...
