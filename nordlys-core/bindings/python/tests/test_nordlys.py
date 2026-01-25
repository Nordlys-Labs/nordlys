"""Tests for Python bindings - Nordlys32/Nordlys64 classes."""

import numpy as np
import pytest


class TestNordlys32Creation:
    """Test Nordlys32 factory methods."""

    def test_from_checkpoint(self, sample_checkpoint_json: str):
        """Test creating nordlys32 from checkpoint."""
        from nordlys_core import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        nordlys32 = Nordlys32.from_checkpoint(checkpoint)
        assert nordlys32.n_clusters == 3
        assert nordlys32.embedding_dim == 4

    def test_from_checkpoint_file(self, sample_checkpoint_path):
        """Test creating nordlys32 from checkpoint loaded from file."""
        from nordlys_core import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_file(str(sample_checkpoint_path))
        nordlys32 = Nordlys32.from_checkpoint(checkpoint)
        assert nordlys32.n_clusters == 3

    def test_dtype_mismatch_raises(self, sample_checkpoint_json_float64: str):
        """Test that loading float64 checkpoint into Nordlys32 raises error."""
        from nordlys_core import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json_float64)
        with pytest.raises(ValueError):
            Nordlys32.from_checkpoint(checkpoint)

    def test_get_supported_models(self, nordlys32):
        """Test getting supported models."""
        models = nordlys32.get_supported_models()
        # This will fail if vector.h is missing in nordlys.cpp
        assert isinstance(models, list)
        assert len(models) == 2
        assert "openai/gpt-4" in models
        assert "anthropic/claude-3" in models

        # Test that it's iterable and can be used in list operations
        model_set = set(models)
        assert len(model_set) == 2


class TestRouting:
    """Test routing functionality."""

    def test_route_float32(self, nordlys32, sample_embedding):
        """Test routing with float32 embedding."""
        from nordlys_core import RouteResult32

        response = nordlys32.route(sample_embedding)

        assert isinstance(response, RouteResult32)
        assert response.selected_model in ["openai/gpt-4", "anthropic/claude-3"]
        assert isinstance(response.cluster_id, int)
        assert 0 <= response.cluster_id < 3
        assert response.cluster_distance >= 0.0

    def test_route_float64(self, nordlys64):
        """Test routing with float64 embedding."""
        from nordlys_core import RouteResult64

        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding)

        assert isinstance(response, RouteResult64)
        assert response.selected_model is not None

    def test_alternatives_returned(self, nordlys32, sample_embedding):
        """Test that alternatives are returned."""
        response = nordlys32.route(sample_embedding)

        # This will fail if vector.h is missing in results.cpp
        assert isinstance(response.alternatives, list)
        # max_alternatives is 2, but we only have 2 models, so max 1 alternative
        assert len(response.alternatives) <= 1

        # Test that alternatives can be iterated and used in list operations
        if response.alternatives:
            assert all(isinstance(alt, str) for alt in response.alternatives)
            assert response.selected_model not in response.alternatives


class TestBatchRouting:
    """Test batch routing functionality."""

    def test_batch_route_float32(self, nordlys32):
        """Test batch routing with float32 embeddings."""
        from nordlys_core import RouteResult32

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        responses = nordlys32.route_batch(embeddings)

        # This will fail if vector.h is missing in nordlys.cpp
        assert isinstance(responses, list)
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert isinstance(response, RouteResult32)
            assert response.selected_model is not None
            assert response.cluster_id == i  # Each should match its cluster
            # Test that alternatives vector works for each response
            assert isinstance(response.alternatives, list)

    def test_batch_route_float64(self, nordlys64):
        """Test batch routing with float64 embeddings."""
        from nordlys_core import RouteResult64

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        responses = nordlys64.route_batch(embeddings)

        # This will fail if vector.h is missing in nordlys.cpp
        assert isinstance(responses, list)
        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, RouteResult64)
            assert response.selected_model is not None
            assert isinstance(response.alternatives, list)

    def test_batch_single_embedding(self, nordlys32, sample_embedding):
        """Test batch routing with single embedding."""
        embeddings = sample_embedding.reshape(1, -1)
        responses = nordlys32.route_batch(embeddings)

        assert isinstance(responses, list)
        assert len(responses) == 1
        assert responses[0].selected_model is not None

    def test_batch_with_model_filter(self, nordlys32):
        """Test batch routing with model filter."""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Filter to specific model
        responses = nordlys32.route_batch(embeddings, models=["openai/gpt-4"])
        assert len(responses) == 2
        for response in responses:
            assert response.selected_model == "openai/gpt-4"

    def test_batch_dimension_mismatch_raises(self, nordlys32):
        """Test that wrong embedding dimension raises error."""
        wrong_dim = np.array([[1.0, 0.0]], dtype=np.float32)  # 2-dim instead of 4

        with pytest.raises((ValueError, RuntimeError)):
            nordlys32.route_batch(wrong_dim)

    def test_batch_empty_raises(self, nordlys32):
        """Test that empty batch raises error."""
        empty = np.array([[]], dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            nordlys32.route_batch(empty)


class TestNordlysProperties:
    """Test Nordlys property access."""

    def test_nordlys32_properties(self, nordlys32):
        """Test Nordlys32 properties."""
        assert isinstance(nordlys32.n_clusters, int)
        assert nordlys32.n_clusters == 3
        assert isinstance(nordlys32.embedding_dim, int)
        assert nordlys32.embedding_dim == 4
        assert nordlys32.dtype == "float32"

    def test_nordlys64_properties(self, nordlys64):
        """Test Nordlys64 properties."""
        assert isinstance(nordlys64.n_clusters, int)
        assert nordlys64.n_clusters == 3
        assert isinstance(nordlys64.embedding_dim, int)
        assert nordlys64.embedding_dim == 4
        assert nordlys64.dtype == "float64"


class TestErrorHandling:
    """Test error handling."""

    def test_dimension_mismatch_raises(self, nordlys32):
        """Test that wrong embedding dimension raises error."""
        wrong_dim = np.array([1.0, 0.0], dtype=np.float32)  # 2-dim instead of 4

        with pytest.raises((ValueError, RuntimeError)):
            nordlys32.route(wrong_dim)

    def test_empty_embedding_raises(self, nordlys32):
        """Test that empty embedding raises error."""
        empty = np.array([], dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            nordlys32.route(empty)

    def test_route_with_invalid_model_filter(self, nordlys32, sample_embedding):
        """Test routing with invalid model filter."""
        # Empty filter should work (no filtering)
        response = nordlys32.route(sample_embedding, models=[])
        assert response.selected_model is not None

        # Invalid model ID should still work (just won't match anything)
        response = nordlys32.route(sample_embedding, models=["invalid/model"])
        # Should still return a valid response (may be empty model if no match)
        assert isinstance(response.selected_model, str)
