"""Benchmark clustering algorithms."""

import pytest

from nordlys.clustering import (
    AgglomerativeClusterer,
    GMMClusterer,
    HDBSCANClusterer,
    KMeansClusterer,
    SpectralClusterer,
)


@pytest.mark.benchmark
@pytest.mark.parametrize("n_clusters", [5, 10, 20, 50])
@pytest.mark.parametrize("data_size", [100, 500, 1000])
def bench_kmeans_fit(benchmark, synthetic_embeddings, n_clusters, data_size):
    """Benchmark KMeans fitting performance."""
    # Use subset of embeddings based on data_size
    embeddings = synthetic_embeddings[:data_size]
    clusterer = KMeansClusterer(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    def _fit():
        clusterer.fit(embeddings)
        return clusterer

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("min_cluster_size", [10, 50, 100])
@pytest.mark.parametrize("data_size", [100, 500, 1000])
def bench_hdbscan_fit(benchmark, synthetic_embeddings, min_cluster_size, data_size):
    """Benchmark HDBSCAN fitting performance."""
    # Use subset of embeddings based on data_size
    embeddings = synthetic_embeddings[:data_size]
    clusterer = HDBSCANClusterer(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
    )

    def _fit():
        clusterer.fit(embeddings)
        return clusterer

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [5, 10, 20])
@pytest.mark.parametrize("data_size", [100, 500, 1000])
def bench_gmm_fit(benchmark, synthetic_embeddings, n_components, data_size):
    """Benchmark GMM fitting performance."""
    # Use subset of embeddings based on data_size
    embeddings = synthetic_embeddings[:data_size]
    clusterer = GMMClusterer(
        n_components=n_components,
        random_state=42,
        max_iter=100,
    )

    def _fit():
        clusterer.fit(embeddings)
        return clusterer

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("n_clusters", [5, 10, 20])
@pytest.mark.parametrize("data_size", [100, 500, 1000])
def bench_agglomerative_fit(benchmark, synthetic_embeddings, n_clusters, data_size):
    """Benchmark Agglomerative clustering performance."""
    # Use subset of embeddings based on data_size
    embeddings = synthetic_embeddings[:data_size]
    clusterer = AgglomerativeClusterer(
        n_clusters=n_clusters,
        linkage="ward",
    )

    def _fit():
        clusterer.fit(embeddings)
        return clusterer

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("n_clusters", [5, 10, 20])
@pytest.mark.parametrize("data_size", [100, 500, 1000])
def bench_spectral_fit(benchmark, synthetic_embeddings, n_clusters, data_size):
    """Benchmark Spectral clustering performance."""
    # Use subset of embeddings based on data_size
    embeddings = synthetic_embeddings[:data_size]
    clusterer = SpectralClusterer(
        n_clusters=n_clusters,
        random_state=42,
        n_neighbors=10,
    )

    def _fit():
        clusterer.fit(embeddings)
        return clusterer

    result = benchmark(_fit)
    assert result is not None
