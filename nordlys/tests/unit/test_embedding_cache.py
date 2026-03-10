"""Unit tests for Router embedding cache functionality."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from nordlys import ModelConfig, Router
from nordlys_core import NordlysCheckpoint


@pytest.fixture
def sample_models() -> list[ModelConfig]:
    """Return sample model configurations for testing."""
    return [
        ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
        ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ]


@pytest.fixture
def sample_checkpoint(sample_models: list[ModelConfig]) -> NordlysCheckpoint:
    models = [
        {
            "model_id": m.id,
            "cost_per_1m_input_tokens": m.cost_input,
            "cost_per_1m_output_tokens": m.cost_output,
            "error_rates": [0.2, 0.4],
        }
        for m in sample_models
    ]
    payload = {
        "version": "2.0",
        "cluster_centers": [[1.0, 0.0], [0.0, 1.0]],
        "models": models,
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "trust_remote_code": False,
        },
        "clustering": {
            "n_clusters": 2,
            "random_state": 42,
            "max_iter": 300,
            "n_init": 10,
            "algorithm": "lloyd",
            "normalization": "l2",
        },
        "metrics": {
            "n_samples": 2,
            "cluster_sizes": [1, 1],
            "silhouette_score": None,
            "inertia": None,
        },
    }
    return NordlysCheckpoint.from_json_string(json.dumps(payload))


class TestEmbeddingCacheInitialization:
    """Test embedding cache initialization."""

    def test_default_cache_size(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test that default cache size is 1000."""
        nordlys = Router(checkpoint=sample_checkpoint)
        assert nordlys._embedding_cache_size == 1000
        assert nordlys._embedding_cache.maxsize == 1000

    def test_custom_cache_size(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test creating Router with custom cache size."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=500)
        assert nordlys._embedding_cache_size == 500
        assert nordlys._embedding_cache.maxsize == 500

    def test_cache_size_zero_raises_error(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that cache size 0 raises ValueError."""
        with pytest.raises(
            ValueError, match="embedding_cache_size must be greater than 0"
        ):
            Router(checkpoint=sample_checkpoint, embedding_cache_size=0)

    def test_cache_size_negative_raises_error(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that negative cache size raises ValueError."""
        with pytest.raises(
            ValueError, match="embedding_cache_size must be greater than 0"
        ):
            Router(checkpoint=sample_checkpoint, embedding_cache_size=-1)

    def test_initial_cache_empty(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test that cache is empty initially."""
        nordlys = Router(checkpoint=sample_checkpoint)
        assert len(nordlys._embedding_cache) == 0


class TestEmbeddingCacheInfo:
    """Test embedding_cache_info() method."""

    def test_cache_info_initial_state(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test cache info returns correct initial state."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)
        info = nordlys.embedding_cache_info()

        assert info["size"] == 0
        assert info["maxsize"] == 100

    def test_cache_info_after_adding_entries(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test cache info reflects size correctly."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        nordlys._embedding_cache["test_prompt"] = np.zeros(384)
        info = nordlys.embedding_cache_info()

        assert info["size"] == 1
        assert info["maxsize"] == 100


class TestClearEmbeddingCache:
    """Test clear_embedding_cache() method."""

    def test_clear_removes_entries(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test that clear removes all cached entries."""
        nordlys = Router(checkpoint=sample_checkpoint)

        # Add entries to cache
        nordlys._embedding_cache["prompt1"] = np.zeros(384)
        nordlys._embedding_cache["prompt2"] = np.ones(384)
        assert len(nordlys._embedding_cache) == 2

        nordlys.clear_embedding_cache()

        assert len(nordlys._embedding_cache) == 0


class TestComputeEmbedding:
    """Test compute_embedding() method."""

    def test_cache_miss_computes_embedding(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that cache miss triggers embedding computation."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        mock_embedding = np.random.randn(384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([mock_embedding])

        # Replace the embedding model with mock
        nordlys._embedding_model = mock_model

        result = nordlys.compute_embedding("test prompt")

        mock_model.encode.assert_called_once_with(
            ["test prompt"], convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, mock_embedding)
        assert "test prompt" in nordlys._embedding_cache

    def test_cache_hit_returns_cached(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test that cache hit returns cached embedding without recomputing."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        # Pre-populate cache
        cached_embedding = np.random.randn(384).astype(np.float32)
        nordlys._embedding_cache["test prompt"] = cached_embedding

        mock_model = MagicMock()
        # Replace the embedding model with mock
        nordlys._embedding_model = mock_model

        result = nordlys.compute_embedding("test prompt")

        # Model should not be called on cache hit
        mock_model.encode.assert_not_called()
        np.testing.assert_array_equal(result, cached_embedding)


class TestCacheLRUEviction:
    """Test LRU eviction behavior."""

    def test_lru_eviction_on_full_cache(self, sample_checkpoint: NordlysCheckpoint) -> None:
        """Test that oldest entries are evicted when cache is full."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=3)

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


class TestComputeEmbeddingsBatch:
    """Test _compute_embeddings() batch method with caching."""

    def test_compute_embeddings_all_cache_hits(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that _compute_embeddings uses cache for all texts."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        # Pre-populate cache
        texts = ["text1", "text2", "text3"]
        cached_embeddings = {
            "text1": np.random.randn(384).astype(np.float32),
            "text2": np.random.randn(384).astype(np.float32),
            "text3": np.random.randn(384).astype(np.float32),
        }
        for text, emb in cached_embeddings.items():
            nordlys._embedding_cache[text] = emb

        mock_model = MagicMock()
        nordlys._embedding_model = mock_model

        result = nordlys._compute_embeddings(texts)

        # Model should not be called if all are cache hits
        mock_model.encode.assert_not_called()
        assert result.shape == (3, 384)
        np.testing.assert_array_equal(result[0], cached_embeddings["text1"])
        np.testing.assert_array_equal(result[1], cached_embeddings["text2"])
        np.testing.assert_array_equal(result[2], cached_embeddings["text3"])

    def test_compute_embeddings_all_cache_misses(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that _compute_embeddings computes all texts in batch on cache miss."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        texts = ["text1", "text2", "text3"]
        mock_embeddings = np.random.randn(3, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        nordlys._embedding_model = mock_model

        result = nordlys._compute_embeddings(texts)

        # Model should be called once with all texts
        mock_model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        assert result.shape == (3, 384)
        np.testing.assert_array_equal(result, mock_embeddings)

        # All texts should be in cache now
        for text in texts:
            assert text in nordlys._embedding_cache

    def test_compute_embeddings_mixed_cache_hits_misses(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that _compute_embeddings handles mixed cache hits and misses."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)

        # Pre-populate cache for some texts
        cached_emb = np.random.randn(384).astype(np.float32)
        nordlys._embedding_cache["text1"] = cached_emb.copy()
        nordlys._embedding_cache["text3"] = cached_emb.copy()

        texts = ["text1", "text2", "text3", "text4"]
        # text1 and text3 are cached, text2 and text4 need computation
        new_embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = new_embeddings
        nordlys._embedding_model = mock_model

        result = nordlys._compute_embeddings(texts)

        # Model should be called once with only cache misses
        mock_model.encode.assert_called_once_with(
            ["text2", "text4"], convert_to_numpy=True, show_progress_bar=False
        )
        assert result.shape == (4, 384)

        # Verify order is preserved: text1 (cached), text2 (new), text3 (cached), text4 (new)
        np.testing.assert_array_equal(result[0], cached_emb)
        np.testing.assert_array_equal(result[1], new_embeddings[0])
        np.testing.assert_array_equal(result[2], cached_emb)
        np.testing.assert_array_equal(result[3], new_embeddings[1])

        # New texts should be in cache
        assert "text2" in nordlys._embedding_cache
        assert "text4" in nordlys._embedding_cache

    def test_compute_embeddings_empty_list(
        self, sample_checkpoint: NordlysCheckpoint
    ) -> None:
        """Test that _compute_embeddings handles empty list."""
        nordlys = Router(checkpoint=sample_checkpoint, embedding_cache_size=100)
        result = nordlys._compute_embeddings([])
        assert result.shape == (0,)
        assert len(result) == 0
