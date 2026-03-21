"""GMM clusterer."""

from __future__ import annotations

from nordlys.clustering.base import Clusterer
from nordlys.clustering.gmm.cpu import create_sklearn_model, fit as fit_cpu
from nordlys.clustering.gmm.cuda import fit as fit_cuda
from nordlys.clustering.gmm.protocol import GMMModel
from nordlys.device import DeviceType, get_device, require_cuda

import numpy as np


class GMMClusterer(Clusterer):
    """Gaussian Mixture Model clustering wrapper.

    Thin wrapper over sklearn.mixture.GaussianMixture with sensible defaults.
    Supports both CPU (sklearn) and CUDA (custom cupy) execution.

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
        device: DeviceType = "cpu",
        **kwargs,
    ) -> None:
        """Initialize GMM clusterer.

        Args:
            n_components: Number of mixture components (default: 20)
            covariance_type: Covariance type: "full", "tied", "diag", "spherical" (default: "full")
            max_iter: Maximum EM iterations (default: 100)
            n_init: Number of initializations (default: 1)
            random_state: Random seed for reproducibility (default: 42)
            device: Execution device - "cpu" or "cuda" (default: "cpu")
            **kwargs: Additional arguments passed to GaussianMixture
        """
        match device:
            case "cuda":
                require_cuda()
                if covariance_type not in ("full", "diag"):
                    msg = (
                        f"GMM CUDA supports only covariance_type='full' or 'diag', "
                        f"got '{covariance_type}'"
                    )
                    raise ValueError(msg)
                if kwargs:
                    msg = f"GMM CUDA does not support kwargs: {list(kwargs.keys())}"
                    raise ValueError(msg)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.device = get_device(device)
        self._kwargs = kwargs
        self._model: GMMModel | None = None
        self._embeddings: np.ndarray | None = None
        self._bic: float | None = None

    def _require_model(self, context: str = "") -> GMMModel:
        """Return the fitted model, raising if not fitted."""
        if self._model is None:
            if context == "predict":
                raise RuntimeError("Clusterer must be fitted before predict")
            if context == "predict_proba":
                raise RuntimeError("Clusterer must be fitted before predict_proba")
            raise RuntimeError("Clusterer must be fitted first")
        return self._model

    def fit(self, embeddings: np.ndarray) -> "GMMClusterer":
        """Fit the clusterer on embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._embeddings = embeddings

        match self.device:
            case "cuda":
                self._model = fit_cuda(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=self.random_state,
                    embeddings=embeddings,
                )
                self._bic = self._compute_bic_from_model(embeddings)
            case _:
                model = create_sklearn_model(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    random_state=self.random_state,
                    **self._kwargs,
                )
                self._model = fit_cpu(model, embeddings)
                # Compute BIC from fitted parameters only
                self._bic = self._compute_bic_from_model(embeddings)
        self._embeddings = embeddings
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        return self._require_model("predict").predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster probabilities for embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, n_components)
        """
        return self._require_model("predict_proba").predict_proba(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and predict cluster assignments.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        self.fit(embeddings)
        return self.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers (means) of shape (n_components, n_features)."""
        return self._require_model().cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        return self._require_model().labels_

    @property
    def n_clusters_(self) -> int:
        """Number of clusters (components)."""
        return self.n_components

    @property
    def weights_(self) -> np.ndarray:
        """Mixture weights of shape (n_components,)."""
        return self._require_model().weights_

    @property
    def covariances_(self) -> np.ndarray:
        """Covariance matrices of components."""
        return self._require_model().covariances_

    @property
    def bic_(self) -> float:
        """Bayesian Information Criterion for the fitted model."""
        if self._bic is None:
            raise RuntimeError("Clusterer must be fitted first")
        return self._bic

    def _compute_bic_from_model(self, embeddings: np.ndarray) -> float:
        """Compute BIC for fitted model using public fitted parameters."""
        model = self._require_model()
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]

        weights = model.weights_
        means = model.cluster_centers_
        covariances = model.covariances_

        log_likelihood = 0.0
        for i in range(n_samples):
            sample_ll = 0.0
            for k in range(self.n_components):
                diff = embeddings[i] - means[k]
                try:
                    cov = covariances[k]
                    if self.covariance_type == "full":
                        sign, log_det = np.linalg.slogdet(cov)
                        if sign <= 0:
                            log_det = np.log(np.linalg.det(cov) + 1e-10)
                        mahal = diff @ np.linalg.inv(cov) @ diff
                    else:
                        cov_safe = np.maximum(cov, 1e-10)
                        log_det = np.sum(np.log(cov_safe))
                        mahal = np.sum((diff**2) / cov_safe)
                    sample_ll += weights[k] * np.exp(
                        -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
                    )
                except np.linalg.LinAlgError:
                    sample_ll += 0.0
            log_likelihood += np.log(sample_ll + 1e-10)

        if self.covariance_type == "full":
            n_params = (
                self.n_components * n_features
                + self.n_components * n_features * (n_features + 1) // 2
                + self.n_components
                - 1
            )
        else:
            n_params = 2 * self.n_components * n_features + self.n_components - 1
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        return float(bic)

    def __repr__(self) -> str:
        return f"GMMClusterer(n_components={self.n_components}, covariance_type='{self.covariance_type}', device={self.device!r})"
