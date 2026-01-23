"""Benchmark embedding cache performance."""

import pytest


@pytest.mark.benchmark
def bench_embedding_cache_hit(benchmark, fitted_nordlys):
    """Benchmark embedding computation with cache hits."""
    # Prime the cache with a set of prompts
    prompts = [f"Test prompt {i}" for i in range(100)]
    for prompt in prompts:
        fitted_nordlys.compute_embedding(prompt)

    # Now benchmark cache hits
    prompt = prompts[0]

    def _compute():
        return fitted_nordlys.compute_embedding(prompt)

    result = benchmark(_compute)
    assert result is not None


@pytest.mark.benchmark
def bench_embedding_cache_miss(benchmark, fitted_nordlys):
    """Benchmark embedding computation with cache misses."""
    # Use unique prompts to ensure cache misses
    prompt_template = "Unique prompt for cache miss benchmark {counter}"

    counter = [0]

    def _compute():
        prompt = prompt_template.format(counter=counter[0])
        counter[0] += 1
        return fitted_nordlys.compute_embedding(prompt)

    result = benchmark(_compute)
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("hit_ratio", [0.0, 0.5, 0.8, 1.0])
def bench_embedding_cache_mixed(benchmark, fitted_nordlys, hit_ratio):
    """Benchmark embedding computation with mixed hit/miss ratio."""
    # Create a pool of prompts
    pool_size = 200
    prompts_pool = [f"Cached prompt {i}" for i in range(pool_size)]

    # Prime cache with some prompts based on hit_ratio
    num_cached = int(pool_size * hit_ratio)
    for prompt in prompts_pool[:num_cached]:
        fitted_nordlys.compute_embedding(prompt)

    # Create a sequence that mixes cached and uncached prompts
    import random

    random.seed(42)
    prompt_sequence = []
    for _ in range(100):
        if random.random() < hit_ratio and num_cached > 0:
            # Use cached prompt
            prompt_sequence.append(random.choice(prompts_pool[:num_cached]))
        else:
            # Use uncached prompt
            prompt_sequence.append(f"Uncached prompt {random.randint(10000, 99999)}")

    sequence_idx = [0]

    def _compute():
        prompt = prompt_sequence[sequence_idx[0] % len(prompt_sequence)]
        sequence_idx[0] += 1
        return fitted_nordlys.compute_embedding(prompt)

    result = benchmark(_compute)
    assert result is not None
