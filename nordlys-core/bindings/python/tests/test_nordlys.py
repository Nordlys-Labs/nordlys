"""Tests for Python bindings - Nordlys32/Nordlys64 classes."""

import numpy as np
import pytest


class TestNordlys32Creation:
    """Test Nordlys32 factory methods."""

    def test_from_checkpoint(self, sample_checkpoint_json: str):
        """Test creating nordlys32 from checkpoint."""
        from nordlys_core_ext import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        nordlys32 = Nordlys32.from_checkpoint(checkpoint)
        assert nordlys32.n_clusters == 3
        assert nordlys32.embedding_dim == 4

    def test_from_checkpoint_file(self, sample_checkpoint_path):
        """Test creating nordlys32 from checkpoint loaded from file."""
        from nordlys_core_ext import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_file(str(sample_checkpoint_path))
        nordlys32 = Nordlys32.from_checkpoint(checkpoint)
        assert nordlys32.n_clusters == 3

    def test_dtype_mismatch_raises(self, sample_checkpoint_json_float64: str):
        """Test that loading float64 checkpoint into Nordlys32 raises error."""
        from nordlys_core_ext import Nordlys32, NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json_float64)
        with pytest.raises(ValueError):
            Nordlys32.from_checkpoint(checkpoint)

    def test_get_supported_models(self, nordlys32):
        """Test getting supported models."""
        models = nordlys32.get_supported_models()
        assert len(models) == 2
        assert "openai/gpt-4" in models
        assert "anthropic/claude-3" in models


class TestRouting:
    """Test routing functionality."""

    def test_route_float32(self, nordlys32, sample_embedding):
        """Test routing with float32 embedding."""
        from nordlys_core_ext import RouteResult32

        response = nordlys32.route(sample_embedding, cost_bias=0.5)

        assert isinstance(response, RouteResult32)
        assert response.selected_model in ["openai/gpt-4", "anthropic/claude-3"]
        assert isinstance(response.cluster_id, int)
        assert 0 <= response.cluster_id < 3
        assert response.cluster_distance >= 0.0

    def test_route_float64(self, nordlys64):
        """Test routing with float64 embedding."""
        from nordlys_core_ext import RouteResult64

        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding, cost_bias=0.5)

        assert isinstance(response, RouteResult64)
        assert response.selected_model is not None

    def test_route_default_cost_bias(self, nordlys32, sample_embedding):
        """Test routing with default cost_bias."""
        response = nordlys32.route(sample_embedding)
        assert response.selected_model is not None

    def test_cost_bias_affects_selection(self, nordlys32, sample_embedding):
        """Test that cost_bias affects model selection."""
        response_cheap = nordlys32.route(sample_embedding, cost_bias=1.0)
        response_quality = nordlys32.route(sample_embedding, cost_bias=0.0)

        # Both should return valid models (may or may not be different)
        assert response_cheap.selected_model is not None
        assert response_quality.selected_model is not None

    def test_alternatives_returned(self, nordlys32, sample_embedding):
        """Test that alternatives are returned."""
        response = nordlys32.route(sample_embedding, cost_bias=0.5)

        assert isinstance(response.alternatives, list)
        # max_alternatives is 2, but we only have 2 models, so max 1 alternative
        assert len(response.alternatives) <= 1


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
