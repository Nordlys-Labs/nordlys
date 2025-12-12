"""Tests for batch routing functionality."""

import numpy as np
import pytest


class TestBatchRouting:
    """Test batch routing functionality."""

    def test_batch_route_float32(self, router):
        """Test batch routing with float32 embeddings."""
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        responses = router.route_batch(embeddings, cost_bias=0.5)
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.selected_model is not None
            assert response.cluster_id == i  # Each should match its cluster

    def test_batch_route_float64(self, router):
        """Test batch routing with float64 embeddings."""
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        
        responses = router.route_batch(embeddings, cost_bias=0.5)
        
        assert len(responses) == 2

    def test_batch_single_embedding(self, router, sample_embedding):
        """Test batch routing with single embedding."""
        embeddings = sample_embedding.reshape(1, -1)
        responses = router.route_batch(embeddings, cost_bias=0.5)
        
        assert len(responses) == 1

    def test_batch_dimension_mismatch_raises(self, router):
        """Test that wrong embedding dimension raises error."""
        wrong_dim = np.array([[1.0, 0.0]], dtype=np.float32)  # 2-dim instead of 4
        
        with pytest.raises((ValueError, RuntimeError)):
            router.route_batch(wrong_dim)