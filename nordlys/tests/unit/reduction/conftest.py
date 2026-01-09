"""Fixtures for reduction unit tests."""

import numpy as np
import pytest


@pytest.fixture
def high_dim_embeddings():
    """Generate high-dimensional embeddings (100 samples, 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(100, 384).astype(np.float32)


@pytest.fixture
def small_embeddings():
    """Generate small embeddings (20 samples, 50 dimensions)."""
    np.random.seed(42)
    return np.random.randn(20, 50).astype(np.float32)


@pytest.fixture
def medium_embeddings():
    """Generate medium embeddings (50 samples, 100 dimensions)."""
    np.random.seed(42)
    return np.random.randn(50, 100).astype(np.float32)
