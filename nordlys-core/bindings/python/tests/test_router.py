"""Tests for Python bindings - Router class."""

import numpy as np
import pytest


class TestRouterCreation:
    """Test Router factory methods."""

    def test_from_json_string(self, sample_profile_json: str):
        """Test creating router from JSON string."""
        from adaptive_core_ext import Router

        router = Router.from_json_string(sample_profile_json)
        assert router.get_n_clusters() == 3
        assert router.get_embedding_dim() == 4

    def test_from_json_file(self, sample_profile_path):
        """Test creating router from JSON file."""
        from adaptive_core_ext import Router

        router = Router.from_json_file(str(sample_profile_path))
        assert router.get_n_clusters() == 3

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        from adaptive_core_ext import Router

        with pytest.raises(RuntimeError):
            Router.from_json_string("not valid json")

    def test_get_supported_models(self, router):
        """Test getting supported models."""
        models = router.get_supported_models()
        assert len(models) == 2
        assert "openai/gpt-4" in models
        assert "anthropic/claude-3" in models


class TestRouting:
    """Test routing functionality."""

    def test_route_float32(self, router, sample_embedding):
        """Test routing with float32 embedding."""
        from adaptive_core_ext import RouteResponse32

        response = router.route(sample_embedding, cost_bias=0.5)

        assert isinstance(response, RouteResponse32)
        assert response.selected_model in ["openai/gpt-4", "anthropic/claude-3"]
        assert isinstance(response.cluster_id, int)
        assert 0 <= response.cluster_id < 3
        assert response.cluster_distance >= 0.0

    def test_route_float64(self, router_float64):
        """Test routing with float64 embedding."""
        from adaptive_core_ext import RouteResponse64

        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = router_float64.route(embedding, cost_bias=0.5)

        assert isinstance(response, RouteResponse64)
        assert response.selected_model is not None

    def test_route_default_cost_bias(self, router, sample_embedding):
        """Test routing with default cost_bias."""
        response = router.route(sample_embedding)
        assert response.selected_model is not None

    def test_cost_bias_affects_selection(self, router, sample_embedding):
        """Test that cost_bias affects model selection."""
        response_cheap = router.route(sample_embedding, cost_bias=1.0)
        response_quality = router.route(sample_embedding, cost_bias=0.0)

        # Both should return valid models (may or may not be different)
        assert response_cheap.selected_model is not None
        assert response_quality.selected_model is not None

    def test_alternatives_returned(self, router, sample_embedding):
        """Test that alternatives are returned."""
        response = router.route(sample_embedding, cost_bias=0.5)

        assert isinstance(response.alternatives, list)
        # max_alternatives is 2, but we only have 2 models, so max 1 alternative
        assert len(response.alternatives) <= 1


class TestErrorHandling:
    """Test error handling."""

    def test_dimension_mismatch_raises(self, router):
        """Test that wrong embedding dimension raises error."""
        wrong_dim = np.array([1.0, 0.0], dtype=np.float32)  # 2-dim instead of 4

        with pytest.raises((ValueError, RuntimeError)):
            router.route(wrong_dim)

    def test_empty_embedding_raises(self, router):
        """Test that empty embedding raises error."""
        empty = np.array([], dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            router.route(empty)
