"""Tests for Python bindings - RouteResult classes."""

import numpy as np
import pytest


class TestRouteResult32:
    """Test RouteResult32 type bindings."""

    def test_route_result_properties(self, nordlys32, sample_embedding):
        """Test RouteResult32 property access."""
        from nordlys_core import RouteResult32

        response = nordlys32.route(sample_embedding)

        assert isinstance(response, RouteResult32)
        assert isinstance(response.selected_model, str)
        assert response.selected_model in ["openai/gpt-4", "anthropic/claude-3"]
        assert isinstance(response.cluster_id, int)
        assert 0 <= response.cluster_id < 3
        assert isinstance(response.cluster_distance, float)
        assert response.cluster_distance >= 0.0

        # Test alternatives vector - this will fail if vector.h is missing in results.cpp
        assert isinstance(response.alternatives, list)
        assert all(isinstance(alt, str) for alt in response.alternatives)
        assert response.selected_model not in response.alternatives

    def test_alternatives_vector_operations(self, nordlys32, sample_embedding):
        """Test that alternatives vector supports list operations."""
        response = nordlys32.route(sample_embedding)

        # Test list conversion
        alternatives_list = list(response.alternatives)
        assert isinstance(alternatives_list, list)

        # Test iteration
        for alt in response.alternatives:
            assert isinstance(alt, str)
            assert alt in ["openai/gpt-4", "anthropic/claude-3"]

        # Test length
        assert len(response.alternatives) <= 1  # Max 1 alternative with 2 models

        # Test membership
        if response.alternatives:
            assert response.alternatives[0] in ["openai/gpt-4", "anthropic/claude-3"]

    def test_route_result_repr(self, nordlys32, sample_embedding):
        """Test RouteResult32 __repr__ method."""
        response = nordlys32.route(sample_embedding)

        repr_str = repr(response)
        assert isinstance(repr_str, str)
        assert "RouteResult32" in repr_str
        assert response.selected_model in repr_str
        assert str(response.cluster_id) in repr_str

    def test_route_result_all_models(self, nordlys32):
        """Test RouteResult32 with different embeddings to get all models."""
        from nordlys_core import RouteResult32

        # Test with different embeddings to potentially get different models
        embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        ]

        selected_models = set()
        for embedding in embeddings:
            response = nordlys32.route(embedding)
            assert isinstance(response, RouteResult32)
            selected_models.add(response.selected_model)
            assert isinstance(response.alternatives, list)


class TestRouteResult64:
    """Test RouteResult64 type bindings."""

    def test_route_result_properties(self, nordlys64):
        """Test RouteResult64 property access."""
        from nordlys_core import RouteResult64

        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding)

        assert isinstance(response, RouteResult64)
        assert isinstance(response.selected_model, str)
        assert isinstance(response.cluster_id, int)
        assert 0 <= response.cluster_id < 3
        assert isinstance(response.cluster_distance, float)
        assert response.cluster_distance >= 0.0

        # Test alternatives vector - this will fail if vector.h is missing in results.cpp
        assert isinstance(response.alternatives, list)
        assert all(isinstance(alt, str) for alt in response.alternatives)

    def test_alternatives_vector_operations(self, nordlys64):
        """Test that alternatives vector supports list operations."""
        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding)

        # Test list conversion
        alternatives_list = list(response.alternatives)
        assert isinstance(alternatives_list, list)

        # Test iteration
        for alt in response.alternatives:
            assert isinstance(alt, str)

        # Test length
        assert len(response.alternatives) <= 1

    def test_route_result_repr(self, nordlys64):
        """Test RouteResult64 __repr__ method."""
        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding)

        repr_str = repr(response)
        assert isinstance(repr_str, str)
        assert "RouteResult64" in repr_str
        assert response.selected_model in repr_str
        assert str(response.cluster_id) in repr_str

    def test_route_result_with_model_filter(self, nordlys64):
        """Test RouteResult64 with model filter."""
        from nordlys_core import RouteResult64

        embedding = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float64)
        response = nordlys64.route(embedding, models=["openai/gpt-4"])

        assert isinstance(response, RouteResult64)
        assert response.selected_model == "openai/gpt-4"
        assert isinstance(response.alternatives, list)
        # With filter, alternatives should be empty or contain only filtered models
        assert len(response.alternatives) == 0
