"""Benchmark reduction algorithms."""

import numpy as np
import pytest

from nordlys.reduction import PCAReducer, UMAPReducer


@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [3, 10, 50])
@pytest.mark.parametrize("input_dim", [100, 384, 768])
def bench_umap_fit_transform(benchmark, n_components, input_dim):
    """Benchmark UMAP reduction performance."""
    np.random.seed(42)
    n_samples = 1000
    embeddings = np.random.randn(n_samples, input_dim).astype(np.float32)
    reducer = UMAPReducer(
        n_components=n_components,
        random_state=42,
        n_neighbors=15,
    )

    def _fit_transform():
        return reducer.fit_transform(embeddings)

    result = benchmark(_fit_transform)
    assert result is not None
    assert result.shape == (n_samples, n_components)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [3, 10, 50, 100])
@pytest.mark.parametrize("input_dim", [100, 384, 768])
def bench_pca_fit_transform(benchmark, n_components, input_dim):
    """Benchmark PCA reduction performance."""
    np.random.seed(42)
    n_samples = 1000
    embeddings = np.random.randn(n_samples, input_dim).astype(np.float32)
    reducer = PCAReducer(
        n_components=n_components,
        random_state=42,
    )

    def _fit_transform():
        return reducer.fit_transform(embeddings)

    result = benchmark(_fit_transform)
    assert result is not None
    assert result.shape == (n_samples, n_components)
