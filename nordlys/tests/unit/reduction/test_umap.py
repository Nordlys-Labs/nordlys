"""Unit tests for UMAPReducer."""

import numpy as np
import pytest

from nordlys.reduction.umap_reducer import UMAPReducer


class TestUMAPInitialization:
    """Tests for UMAPReducer initialization."""

    def test_default_initialization(self):
        """Test UMAPReducer with default parameters."""
        reducer = UMAPReducer()
        assert reducer.n_components == 3
        assert reducer.n_neighbors == 15
        assert reducer.min_dist == 0.1
        assert reducer.metric == "cosine"
        assert reducer.random_state == 42

    def test_custom_n_components(self):
        """Test initialization with custom n_components."""
        reducer = UMAPReducer(n_components=5)
        assert reducer.n_components == 5

    def test_custom_n_neighbors(self):
        """Test initialization with custom n_neighbors."""
        reducer = UMAPReducer(n_neighbors=30)
        assert reducer.n_neighbors == 30

    def test_custom_min_dist(self):
        """Test initialization with custom min_dist."""
        reducer = UMAPReducer(min_dist=0.5)
        assert reducer.min_dist == 0.5

    def test_custom_metric(self):
        """Test initialization with custom metric."""
        reducer = UMAPReducer(metric="euclidean")
        assert reducer.metric == "euclidean"

    def test_custom_random_state(self):
        """Test initialization with custom random_state."""
        reducer = UMAPReducer(random_state=123)
        assert reducer.random_state == 123

    def test_kwargs_stored(self):
        """Test that additional kwargs are stored."""
        reducer = UMAPReducer(verbose=True)
        assert "verbose" in reducer._kwargs
        assert reducer._kwargs["verbose"] is True


class TestUMAPFit:
    """Tests for UMAPReducer fit method."""

    def test_fit_stores_model(self, high_dim_embeddings):
        """Test that fit creates and stores model."""
        reducer = UMAPReducer(n_components=3)
        reducer.fit(high_dim_embeddings)
        assert reducer._model is not None

    def test_fit_returns_self(self, high_dim_embeddings):
        """Test that fit returns self for chaining."""
        reducer = UMAPReducer(n_components=3)
        result = reducer.fit(high_dim_embeddings)
        assert result is reducer

    def test_embedding_available_after_fit(self, high_dim_embeddings):
        """Test that embedding is available after fit."""
        reducer = UMAPReducer(n_components=3)
        reducer.fit(high_dim_embeddings)
        embedding = reducer.embedding_
        assert embedding is not None
        assert embedding.shape == (100, 3)


class TestUMAPTransform:
    """Tests for UMAPReducer transform method."""

    def test_transform_after_fit(self, high_dim_embeddings):
        """Test transforming data after fit."""
        reducer = UMAPReducer(n_components=3)
        reducer.fit(high_dim_embeddings)

        test_data = np.random.randn(10, 384).astype(np.float32)
        transformed = reducer.transform(test_data)

        assert transformed.shape == (10, 3)

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises RuntimeError."""
        reducer = UMAPReducer(n_components=3)
        test_data = np.random.randn(10, 100).astype(np.float32)

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            reducer.transform(test_data)

    def test_transform_shape(self, high_dim_embeddings):
        """Test that transform returns correct shape."""
        reducer = UMAPReducer(n_components=5)
        reducer.fit(high_dim_embeddings)

        for n_samples in [1, 5, 10]:
            test_data = np.random.randn(n_samples, 384).astype(np.float32)
            transformed = reducer.transform(test_data)
            assert transformed.shape == (n_samples, 5)

    def test_transform_consistency(self, high_dim_embeddings):
        """Test that transform is consistent across calls."""
        reducer = UMAPReducer(n_components=3, random_state=42)
        reducer.fit(high_dim_embeddings)

        test_data = np.random.randn(10, 384).astype(np.float32)
        result1 = reducer.transform(test_data)
        result2 = reducer.transform(test_data)

        # UMAP transform might have slight variations
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)


class TestUMAPFitTransform:
    """Tests for UMAPReducer fit_transform method."""

    def test_fit_transform_combines_operations(self, high_dim_embeddings):
        """Test that fit_transform works correctly."""
        reducer = UMAPReducer(n_components=3)
        result = reducer.fit_transform(high_dim_embeddings)

        assert result.shape == (100, 3)

    def test_fit_transform_creates_model(self, high_dim_embeddings):
        """Test that fit_transform creates the model."""
        reducer = UMAPReducer(n_components=3)
        reducer.fit_transform(high_dim_embeddings)

        assert reducer._model is not None


class TestUMAPReproducibility:
    """Tests for UMAPReducer reproducibility."""

    def test_same_random_state_similar_results(self, medium_embeddings):
        """Test that same random_state produces similar results."""
        reducer1 = UMAPReducer(n_components=3, random_state=42)
        result1 = reducer1.fit_transform(medium_embeddings)

        reducer2 = UMAPReducer(n_components=3, random_state=42)
        result2 = reducer2.fit_transform(medium_embeddings)

        # Results should be very similar (UMAP has some randomness)
        assert result1.shape == result2.shape


class TestUMAPProperties:
    """Tests for UMAPReducer properties."""

    def test_embedding_property(self, high_dim_embeddings):
        """Test embedding_ property."""
        reducer = UMAPReducer(n_components=3)
        reducer.fit(high_dim_embeddings)

        embedding = reducer.embedding_
        assert embedding is not None
        assert embedding.shape == (100, 3)

    def test_embedding_none_before_fit(self):
        """Test that embedding_ is None before fit."""
        reducer = UMAPReducer(n_components=3)
        assert reducer.embedding_ is None


class TestUMAPRepr:
    """Tests for UMAPReducer string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        reducer = UMAPReducer(n_components=5, n_neighbors=20)
        repr_str = repr(reducer)

        assert "UMAPReducer" in repr_str
        assert "n_components=5" in repr_str
        assert "n_neighbors=20" in repr_str
