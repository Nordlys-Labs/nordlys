"""HDBSCAN clusterer."""

from __future__ import annotations

import logging

import numpy as np
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)


class HDBSCANClusterer:
    """HDBSCAN clustering wrapper.

    Thin wrapper over hdbscan.HDBSCAN with sensible defaults.
    HDBSCAN automatically determines the number of clusters.

    Example:
        >>> clusterer = HDBSCANClusterer(min_cluster_size=100)
        >>> clusterer.fit(embeddings)
        >>> # Note: HDBSCAN doesn't support predict() for new samples by default
    """

    def __init__(
        self,
        min_cluster_size: int = 100,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        prediction_data: bool = True,
        **kwargs,
    ) -> None:
        """Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum size of clusters (default: 100)
            min_samples: Minimum samples for core points (default: None, uses min_cluster_size)
            metric: Distance metric (default: "euclidean")
            cluster_selection_epsilon: Distance threshold for merging (default: 0.0)
            cluster_selection_method: "eom" or "leaf" (default: "eom")
            prediction_data: Generate prediction data for approximate_predict (default: True)
            **kwargs: Additional arguments passed to HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.prediction_data = prediction_data
        self._kwargs = kwargs
        self._model: HDBSCAN | None = None
        self._cluster_centers: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

    def _create_model(self) -> HDBSCAN:
        """Create the underlying HDBSCAN model."""
        return HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=self.prediction_data,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "HDBSCANClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        self._model = self._create_model()
        self._model.fit(embeddings)
        self._embeddings = embeddings
        self._compute_cluster_centers()
        return self

    def _compute_cluster_centers(self) -> None:
        """Compute cluster centers as mean of cluster members."""
        if self._model is None or self._embeddings is None:
            return

        labels = self._model.labels_
        unique_labels = np.unique(labels)
        # Filter out noise label (-1)
        valid_labels = unique_labels[unique_labels >= 0]

        if len(valid_labels) == 0:
            self._cluster_centers = np.empty((0, self._embeddings.shape[1]))
            return

        centers = []
        for label in sorted(valid_labels):
            mask = labels == label
            center = self._embeddings[mask].mean(axis=0)
            centers.append(center)

        self._cluster_centers = np.array(centers)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new embeddings.

        Uses approximate_predict for soft cluster assignment.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self._model is None:
            raise RuntimeError(
                "Clusterer must be fitted before predict. Call fit() first."
            )

        # Try to import and use approximate_predict
        try:
            from hdbscan import approximate_predict
        except ImportError as e:
            logger.warning(
                "hdbscan.approximate_predict not available: %s. "
                "Falling back to nearest centroid assignment.",
                e,
            )
            # Fallback: assign to nearest centroid
            if self._cluster_centers is None or len(self._cluster_centers) == 0:
                return np.full(len(embeddings), -1)
            distances = np.linalg.norm(
                embeddings[:, np.newaxis] - self._cluster_centers, axis=2
            )
            return distances.argmin(axis=1)

        # Try to use approximate_predict
        try:
            labels, _ = approximate_predict(self._model, embeddings)
            return labels
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(
                "approximate_predict failed with %s: %s. "
                "Falling back to nearest centroid assignment.",
                type(e).__name__,
                e,
            )
            # Fallback: assign to nearest centroid
            if self._cluster_centers is None or len(self._cluster_centers) == 0:
                return np.full(len(embeddings), -1)
            distances = np.linalg.norm(
                embeddings[:, np.newaxis] - self._cluster_centers, axis=2
            )
            return distances.argmin(axis=1)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and return cluster assignments.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        self.fit(embeddings)
        assert self._model is not None
        labels = self._model.labels_
        assert labels is not None
        return labels

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features).

        Computed as the mean of cluster members.
        """
        if self._cluster_centers is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,).

        -1 indicates noise points.
        """
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        labels = self._model.labels_
        assert labels is not None
        return labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding noise)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        labels = self._model.labels_
        return len(set(labels)) - (1 if -1 in labels else 0)

    @property
    def probabilities_(self) -> np.ndarray:
        """Cluster membership probabilities of shape (n_samples,)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._model.probabilities_

    def __repr__(self) -> str:
        return f"HDBSCANClusterer(min_cluster_size={self.min_cluster_size})"
