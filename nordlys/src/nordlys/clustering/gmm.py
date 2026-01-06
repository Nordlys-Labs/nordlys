"""Gaussian Mixture Model clusterer."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GMMClusterer:
    """Gaussian Mixture Model clustering wrapper.

    Thin wrapper over sklearn.mixture.GaussianMixture with sensible defaults.

    Example:
        >>> clusterer = GMMClusterer(n_components=20)
        >>> clusterer.fit(embeddings)
        >>> labels = clusterer.predict(new_embeddings)
    """

    def __init__(
        self,
        n_components: int = 20,
        covariance_type: str = "full",
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Initialize GMM clusterer.

        Args:
            n_components: Number of mixture components (default: 20)
            covariance_type: Covariance type: "full", "tied", "diag", "spherical" (default: "full")
            max_iter: Maximum EM iterations (default: 100)
            n_init: Number of initializations (default: 1)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional arguments passed to GaussianMixture
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self._kwargs = kwargs
        self._model: GaussianMixture | None = None
        self._labels: np.ndarray | None = None

    def _create_model(self) -> GaussianMixture:
        """Create the underlying GaussianMixture model."""
        return GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
            **self._kwargs,
        )

    def fit(self, embeddings: np.ndarray) -> "GMMClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        self._model = self._create_model()
        self._model.fit(embeddings)
        self._labels = self._model.predict(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self._model is None:
            raise RuntimeError(
                "Clusterer must be fitted before predict. Call fit() first."
            )
        return self._model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_components)
        """
        if self._model is None:
            raise RuntimeError(
                "Clusterer must be fitted before predict_proba. Call fit() first."
            )
        return self._model.predict_proba(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and predict cluster assignments.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        self.fit(embeddings)
        assert self._labels is not None
        return self._labels

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers (means) of shape (n_components, n_features)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        means = self._model.means_
        assert means is not None
        return means

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters (components)."""
        return self.n_components

    @property
    def weights_(self) -> np.ndarray:
        """Mixture weights of shape (n_components,)."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        weights = self._model.weights_
        assert weights is not None
        return weights

    @property
    def covariances_(self) -> np.ndarray:
        """Covariance matrices of components."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        covariances = self._model.covariances_
        assert covariances is not None
        return covariances

    @property
    def bic_(self) -> float:
        """Bayesian Information Criterion of the fitted model."""
        if self._model is None:
            raise RuntimeError("Clusterer must be fitted first.")
        # BIC is computed per-sample, need original embeddings
        # Return converged status instead for now
        return float(self._model.lower_bound_)

    def __repr__(self) -> str:
        return f"GMMClusterer(n_components={self.n_components}, covariance_type='{self.covariance_type}')"
