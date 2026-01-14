"""Unit tests for Nordlys embedding cache functionality."""

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

    def test_cache_size_zero_raises_error(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that cache size 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="embedding_cache_size must be greater than 0"
        ):
            Nordlys(models=sample_models, embedding_cache_size=0)

    def test_cache_size_negative_raises_error(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that negative cache size raises ValueError."""
        with pytest.raises(
            ValueError, match="embedding_cache_size must be greater than 0"
        ):
            Nordlys(models=sample_models, embedding_cache_size=-1)

    def test_initial_cache_empty(self, sample_models: list[ModelConfig]) -> None:
        """Test that cache is empty initially."""
        nordlys = Nordlys(models=sample_models)
        assert len(nordlys._embedding_cache) == 0


class TestEmbeddingCacheInfo:
    """Test embedding_cache_info() method."""

    def test_cache_info_initial_state(self, sample_models: list[ModelConfig]) -> None:
        """Test cache info returns correct initial state."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)
        info = nordlys.embedding_cache_info()

        assert info["size"] == 0
        assert info["maxsize"] == 100

    def test_cache_info_after_adding_entries(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test cache info reflects size correctly."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        nordlys._embedding_cache["test_prompt"] = np.zeros(384)
        info = nordlys.embedding_cache_info()

        assert info["size"] == 1
        assert info["maxsize"] == 100


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


class TestComputeEmbedding:
    """Test compute_embedding() method."""

    def test_cache_miss_computes_embedding(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Test that cache miss triggers embedding computation."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([mock_embedding])

        with patch.object(nordlys, "_load_embedding_model", return_value=mock_model):
            result = nordlys.compute_embedding("test prompt")

        mock_model.encode.assert_called_once_with(
            ["test prompt"], convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, mock_embedding)
        assert "test prompt" in nordlys._embedding_cache

    def test_cache_hit_returns_cached(self, sample_models: list[ModelConfig]) -> None:
        """Test that cache hit returns cached embedding without recomputing."""
        nordlys = Nordlys(models=sample_models, embedding_cache_size=100)

        # Pre-populate cache
        cached_embedding = np.random.randn(384).astype(np.float32)
        nordlys._embedding_cache["test prompt"] = cached_embedding

        mock_model = MagicMock()
        with patch.object(nordlys, "_load_embedding_model", return_value=mock_model):
            result = nordlys.compute_embedding("test prompt")

        # Model should not be called on cache hit
        mock_model.encode.assert_not_called()
        np.testing.assert_array_equal(result, cached_embedding)


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
