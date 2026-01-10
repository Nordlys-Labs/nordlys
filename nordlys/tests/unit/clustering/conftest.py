"""Fixtures for clustering unit tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_2d_clusters():
    """Generate 3 well-separated 2D clusters (60 samples total)."""
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([10, 10])
    cluster3 = np.random.randn(20, 2) + np.array([20, 0])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def simple_5d_clusters():
    """Generate 3 well-separated 5D clusters (60 samples total)."""
    np.random.seed(42)
    cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
    cluster3 = np.random.randn(20, 5) + np.array([20, 20, 20, 20, 20])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def overlapping_clusters():
    """Generate overlapping clusters for edge case testing."""
    np.random.seed(42)
    cluster1 = np.random.randn(30, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(30, 3) + np.array([2, 2, 2])  # Close to cluster1
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def identical_points():
    """All identical points (edge case)."""
    return np.ones((20, 5))
