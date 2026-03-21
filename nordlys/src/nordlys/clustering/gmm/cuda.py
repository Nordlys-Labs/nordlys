"""CUDA adapter for GMM using cupy."""

from __future__ import annotations

from nordlys.clustering.gmm.protocol import GMMModel

import numpy as np


class CupyGMMModel:
    """GPU implementation of GMM using cupy."""

    def __init__(
        self,
        means: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        covariances: np.ndarray,
        converged: bool,
        covariance_type: str,
    ) -> None:
        self._means = means
        self._labels = labels
        self._weights = weights
        self._covariances = covariances
        self._converged = converged
        self._covariance_type = covariance_type

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._means

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def n_components_(self) -> int:
        return len(self._means)

    @property
    def weights_(self) -> np.ndarray:
        return self._weights

    @property
    def covariances_(self) -> np.ndarray:
        return self._covariances

    @property
    def converged_(self) -> bool:
        return self._converged

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        import cupy as cp

        X = cp.asarray(embeddings, dtype=cp.float32)
        means = cp.asarray(self._means, dtype=cp.float32)
        weights = cp.asarray(self._weights, dtype=cp.float32)
        covariances = cp.asarray(self._covariances, dtype=cp.float32)

        log_resp = _compute_log_resp(
            X, means, weights, covariances, self._covariance_type
        )
        labels = np.argmax(cp.asnumpy(log_resp), axis=1)

        return labels

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        import cupy as cp

        X = cp.asarray(embeddings, dtype=cp.float32)
        means = cp.asarray(self._means, dtype=cp.float32)
        weights = cp.asarray(self._weights, dtype=cp.float32)
        covariances = cp.asarray(self._covariances, dtype=cp.float32)

        log_resp = _compute_log_resp(
            X, means, weights, covariances, self._covariance_type
        )
        resp = cp.exp(log_resp - cp.max(log_resp, axis=1, keepdims=True))
        resp = resp / resp.sum(axis=1, keepdims=True)

        return cp.asnumpy(resp)


def fit(
    n_components: int,
    covariance_type: str,
    max_iter: int,
    n_init: int,
    random_state: int,
    embeddings: np.ndarray,
) -> GMMModel:
    """Fit using GPU (cupy).

    Implements EM algorithm for Gaussian Mixture Models.
    Supports 'full' and 'diag' covariance types.
    """
    import cupy as cp

    if covariance_type not in ("full", "diag"):
        msg = f"GMM CUDA supports only 'full' or 'diag', got '{covariance_type}'"
        raise ValueError(msg)

    data = cp.asarray(embeddings, dtype=cp.float32)
    n_samples, n_features = data.shape

    best_labels: np.ndarray | None = None
    best_means: np.ndarray | None = None
    best_weights: np.ndarray | None = None
    best_covariances: np.ndarray | None = None
    best_converged = False

    best_score = -np.inf

    for init_idx in range(n_init):
        init_converged = False
        init_rng = cp.random.RandomState(random_state + init_idx)

        weights = cp.ones(n_components, dtype=cp.float32) / n_components

        indices = init_rng.choice(n_samples, n_components, replace=False)
        means = data[indices].copy()

        data_var = cp.var(data, axis=0)

        if covariance_type == "full":
            covariances = cp.zeros(
                (n_components, n_features, n_features), dtype=cp.float32
            )
            for k in range(n_components):
                covariances[k] = cp.diag(data_var + 0.1)
        else:
            covariances = data_var + 0.1

        old_means = means.copy()

        for iteration in range(max_iter):
            log_resp = _compute_log_resp(
                data, means, weights, covariances, covariance_type
            )
            resp = cp.exp(log_resp - cp.max(log_resp, axis=1, keepdims=True))
            resp = resp / resp.sum(axis=1, keepdims=True)

            Nk = resp.sum(axis=0) + 1e-10

            new_weights = Nk / n_samples

            new_means = cp.zeros((n_components, n_features), dtype=cp.float32)
            for k in range(n_components):
                new_means[k] = (resp[:, k : k + 1] * data).sum(axis=0) / Nk[k]

            if covariance_type == "full":
                new_covariances = cp.zeros(
                    (n_components, n_features, n_features), dtype=cp.float32
                )
                for k in range(n_components):
                    diff = data - new_means[k]
                    weighted_diff = resp[:, k : k + 1] * diff
                    new_covariances[k] = (weighted_diff.T @ diff) / Nk[k]
                    new_covariances[k] += cp.eye(n_features, dtype=cp.float32) * 1e-6
            else:
                new_covariances = cp.zeros((n_components, n_features), dtype=cp.float32)
                for k in range(n_components):
                    diff = data - new_means[k]
                    weighted_diff = resp[:, k : k + 1] * diff
                    new_covariances[k] = (weighted_diff * diff).sum(axis=0) / Nk[k]
                    new_covariances[k] += 1e-6

            weights = new_weights
            means = new_means
            covariances = new_covariances

            if iteration > 0:
                mean_change = float(cp.max(cp.abs(means - old_means)))
                if mean_change < 1e-4:
                    init_converged = True
                    break
            old_means = means.copy()

        final_log_resp = _compute_log_resp(
            data, means, weights, covariances, covariance_type
        )
        log_likelihood = cp.sum(
            cp.log(
                cp.sum(
                    cp.exp(
                        final_log_resp - cp.max(final_log_resp, axis=1, keepdims=True)
                    )
                    + 1e-10,
                    axis=1,
                )
                + 1e-10
            )
            + cp.max(final_log_resp, axis=1)
        )
        current_score = float(log_likelihood)

        final_labels = np.argmax(cp.asnumpy(final_log_resp), axis=1)

        if current_score > best_score:
            best_score = current_score
            best_labels = final_labels
            best_means = cp.asnumpy(means)
            best_weights = cp.asnumpy(weights)
            best_covariances = cp.asnumpy(covariances)
            best_converged = init_converged

    if (
        best_labels is None
        or best_means is None
        or best_weights is None
        or best_covariances is None
    ):
        raise RuntimeError("GMM fit failed: no valid solution found")

    return CupyGMMModel(
        best_means,
        best_labels,
        best_weights,
        best_covariances,
        best_converged,
        covariance_type,
    )


def _compute_log_resp(X, means, weights, covariances, covariance_type):
    """Compute log responsibilities."""
    import cupy as cp
    from cupy.linalg import solve

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = means.shape[0]

    log_weights = cp.log(weights + 1e-10)

    log_resp = cp.zeros((n_samples, n_components), dtype=cp.float32)

    for k in range(n_components):
        mean = means[k]
        cov = covariances[k]

        diff = X - mean

        if covariance_type == "full":
            try:
                cov_chol = cp.linalg.cholesky(cov)
                log_det = 2.0 * cp.sum(cp.log(cp.diag(cov_chol)))
                solve_result = solve(cov_chol, diff.T).T
            except cp.linalg.LinAlgError:
                cov_reg = cov + cp.eye(cov.shape[0], dtype=cp.float32) * 1e-4
                cov_chol = cp.linalg.cholesky(cov_reg)
                log_det = 2.0 * cp.sum(cp.log(cp.diag(cov_chol)))
                solve_result = solve(cov_chol, diff.T).T
            mahalanobis = cp.sum(solve_result**2, axis=1)
            log_prob = -0.5 * (n_features * cp.log(2 * cp.pi) + log_det + mahalanobis)
        else:
            cov_reg = cov + 1e-6
            log_det = cp.sum(cp.log(cov_reg))
            solve_result = diff / cov_reg
            mahalanobis = cp.sum(solve_result**2, axis=1)
            log_prob = -0.5 * (n_features * cp.log(2 * cp.pi) + log_det + mahalanobis)

        log_resp[:, k] = log_prob + log_weights[k]

    return log_resp
