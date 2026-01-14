"""Unit tests for Nordlys embedding cache functionality."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nordlys import ModelConfig, Nordlys


@pytest.fixture
def sample_models() -> list[ModelConfig]:
    """Return sample model configurations for testing."""
    return [
        ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
        ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ]


class TestEmbeddingCacheInitialization:
    """Test embedding cache initialization."""

    def test_default_cache_size(self, sample_models: list[ModelConfig]) -> None:
        """Test that default cache size is 1000."""
        nordlys = Nordlys(models=sample_models)
        assert nordlys._embedding_cache_size == 1000
        assert nordlys._embedding_cache.maxsize == 1000

    def test_custom_cache_size(self, sample_models: list[ModelConfig]) -> None:
        """Test creating Nordlys with custom cache size."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=500)
        assert nordlys._embedding_cache_size == 500
        assert nordlys._embedding_cache.maxsize == 500

    def test_cache_size_zero_uses_minimum(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that cache size 0 still creates cache with maxsize 1."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=0)
        assert nordlys._embedding_cache_size == 0
        # Cache is created with max(1, size) to avoid errors
        assert nordlys._embedding_cache.maxsize == 1

    def test_initial_cache_stats(self, sample_models: list[ModelConfig]) -> None:
        """Test that cache stats are zero initially."""
        nordlys = Nordlys(models=sample_models)
        assert nordlys._embedding_cache_hits == 0
        assert nordlys._embedding_cache_misses == 0
        assert len(nordlys._embedding_cache) == 0


class TestEmbeddingCacheInfo:
    """Test embedding_cache_info() method."""

    def test_cache_info_initial_state(self, sample_models: list[ModelConfig]) -> None:
        """Test cache info returns correct initial state."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)
        info = nordlys.embedding_cache_info()

        assert info["size"] == 0
        assert info["maxsize"] == 100
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["hit_rate"] == 0.0

    def test_cache_info_after_operations(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test cache info reflects operations correctly."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        # Simulate cache operations
        nordlys._embedding_cache["test_prompt"] = np.zeros(384)
        nordlys._embedding_cache_hits = 5
        nordlys._embedding_cache_misses = 10

        info = nordlys.embedding_cache_info()

        assert info["size"] == 1
        assert info["hits"] == 5
        assert info["misses"] == 10
        assert info["hit_rate"] == pytest.approx(5 / 15)


class TestClearEmbeddingCache:
    """Test clear_embedding_cache() method."""

    def test_clear_removes_entries(self, sample_models: list[ModelConfig]) -> None:
        """Test that clear removes all cached entries."""
        nordlys = Nordlys(models=sample_models)

        # Add entries to cache
        nordlys._embedding_cache["prompt1"] = np.zeros(384)
        nordlys._embedding_cache["prompt2"] = np.ones(384)
        assert len(nordlys._embedding_cache) == 2

        nordlys.clear_embedding_cache()

        assert len(nordlys._embedding_cache) == 0

    def test_clear_resets_stats(self, sample_models: list[ModelConfig]) -> None:
        """Test that clear resets hit/miss statistics."""
        nordlys = Nordlys(models=sample_models)

        # Simulate some cache activity
        nordlys._embedding_cache_hits = 50
        nordlys._embedding_cache_misses = 25

        nordlys.clear_embedding_cache()

        assert nordlys._embedding_cache_hits == 0
        assert nordlys._embedding_cache_misses == 0


class TestComputeEmbeddingCached:
    """Test _compute_embedding_cached() method."""

    def test_cache_miss_computes_embedding(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that cache miss triggers embedding computation."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([mock_embedding])

        with patch.object(nordlys, "_load_embedding_model", return_value=mock_model):
            result = nordlys._compute_embedding_cached("test prompt")

        mock_model.encode.assert_called_once_with(
            ["test prompt"], convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, mock_embedding)
        assert nordlys._embedding_cache_misses == 1
        assert "test prompt" in nordlys._embedding_cache

    def test_cache_hit_returns_cached(self, sample_models: list[ModelConfig]) -> None:
        """Test that cache hit returns cached embedding without recomputing."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        # Pre-populate cache
        cached_embedding = np.random.randn(384).astype(np.float32)
        nordlys._embedding_cache["test prompt"] = cached_embedding

        mock_model = MagicMock()
        with patch.object(nordlys, "_load_embedding_model", return_value=mock_model):
            result = nordlys._compute_embedding_cached("test prompt")

        # Model should not be called on cache hit
        mock_model.encode.assert_not_called()
        np.testing.assert_array_equal(result, cached_embedding)
        assert nordlys._embedding_cache_hits == 1

    def test_cache_disabled_always_computes(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that with cache_size=0, embeddings are always computed."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=0)

        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([mock_embedding])

        with patch.object(nordlys, "_load_embedding_model", return_value=mock_model):
            # Call twice with same prompt
            nordlys._compute_embedding_cached("test prompt")
            nordlys._compute_embedding_cached("test prompt")

        # Should compute both times since caching is disabled
        assert mock_model.encode.call_count == 2


class TestCacheThreadSafety:
    """Test thread safety of embedding cache."""

    def test_concurrent_cache_access(self, sample_models: list[ModelConfig]) -> None:
        """Test that concurrent access doesn't corrupt cache state."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)
        errors: list[Exception] = []

        def cache_operation(prompt_id: int) -> None:
            try:
                prompt = f"test prompt {prompt_id}"
                # Simulate cache operations
                nordlys._embedding_cache[prompt] = np.zeros(384)
                _ = nordlys.embedding_cache_info()
                if prompt_id % 2 == 0:
                    nordlys.clear_embedding_cache()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cache_operation, args=(i,)) for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur
        assert len(errors) == 0

    def test_concurrent_cache_info_reads(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that concurrent reads of cache_info are safe."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)
        results: list[dict] = []

        def read_cache_info() -> None:
            for _ in range(100):
                info = nordlys.embedding_cache_info()
                results.append(info)

        threads = [threading.Thread(target=read_cache_info) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should complete without error
        assert len(results) == 500
        # All should have valid structure
        for info in results:
            assert "size" in info
            assert "maxsize" in info
            assert "hit_rate" in info


class TestCacheLRUEviction:
    """Test LRU eviction behavior."""

    def test_lru_eviction_on_full_cache(self, sample_models: list[ModelConfig]) -> None:
        """Test that oldest entries are evicted when cache is full."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=3)

        # Fill cache
        nordlys._embedding_cache["prompt1"] = np.zeros(384)
        nordlys._embedding_cache["prompt2"] = np.ones(384)
        nordlys._embedding_cache["prompt3"] = np.full(384, 2.0)

        assert len(nordlys._embedding_cache) == 3
        assert "prompt1" in nordlys._embedding_cache

        # Add one more - should evict oldest (prompt1)
        nordlys._embedding_cache["prompt4"] = np.full(384, 3.0)

        assert len(nordlys._embedding_cache) == 3
        assert "prompt1" not in nordlys._embedding_cache
        assert "prompt4" in nordlys._embedding_cache
