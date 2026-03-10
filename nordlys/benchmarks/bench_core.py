"""Benchmark core Router operations."""

import pytest

from nordlys import Trainer
from nordlys.clustering import KMeansClusterer


@pytest.mark.benchmark
def bench_fit_small(benchmark, benchmark_models, small_dataset):
    """Benchmark Trainer.fit() with small dataset (~100 samples)."""

    def _fit():
        trainer = Trainer(
            models=benchmark_models,
            clusterer=KMeansClusterer(n_clusters=10, random_state=42),
        )
        return trainer.fit(small_dataset)

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
def bench_fit_medium(benchmark, benchmark_models, medium_dataset):
    """Benchmark Trainer.fit() with medium dataset (~1000 samples)."""

    def _fit():
        trainer = Trainer(
            models=benchmark_models,
            clusterer=KMeansClusterer(n_clusters=20, random_state=42),
        )
        return trainer.fit(medium_dataset)

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
def bench_fit_large(benchmark, benchmark_models, large_dataset):
    """Benchmark Trainer.fit() with large dataset (~10000 samples)."""

    def _fit():
        trainer = Trainer(
            models=benchmark_models,
            clusterer=KMeansClusterer(n_clusters=30, random_state=42),
        )
        return trainer.fit(large_dataset)

    result = benchmark(_fit)
    assert result is not None


@pytest.mark.benchmark
def bench_route_single(benchmark, fitted_nordlys):
    """Benchmark single route() call."""
    prompt = "Explain quantum computing in simple terms"

    def _route():
        return fitted_nordlys.route(prompt)

    result = benchmark(_route)
    assert result is not None
    assert result.model_id is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("batch_size", [10, 50, 100, 500])
def bench_route_batch(benchmark, fitted_nordlys, batch_size):
    """Benchmark route_batch() with varying batch sizes."""
    prompts = [f"Explain concept {i} in detail" for i in range(batch_size)]

    def _route_batch():
        return fitted_nordlys.route_batch(prompts)

    result = benchmark(_route_batch)
    assert result is not None
    assert len(result) == batch_size


@pytest.mark.benchmark
def bench_compute_embedding_cache_hit(benchmark, fitted_nordlys):
    """Benchmark embedding computation with cache hit."""
    prompt = "This is a test prompt for caching"
    # Prime the cache
    fitted_nordlys.compute_embedding(prompt)

    def _compute():
        return fitted_nordlys.compute_embedding(prompt)

    result = benchmark(_compute)
    assert result is not None


@pytest.mark.benchmark
def bench_compute_embedding_cache_miss(benchmark, fitted_nordlys):
    """Benchmark embedding computation with cache miss."""
    # Use unique prompts to ensure cache misses
    prompt_template = "Unique prompt for cache miss test {counter}"

    counter = [0]

    def _compute():
        prompt = prompt_template.format(counter=counter[0])
        counter[0] += 1
        return fitted_nordlys.compute_embedding(prompt)

    result = benchmark(_compute)
    assert result is not None
