"""Unit tests for PCAReducer."""

import numpy as np
import pytest

from nordlys.reduction.pca import PCAReducer


class TestPCAInitialization:
    """Tests for PCAReducer initialization."""

    def test_default_initialization(self):
        """Test PCAReducer with default parameters."""
        reducer = PCAReducer()
        assert reducer.n_components == 50
        assert reducer.random_state == 42

    def test_custom_n_components_int(self):
        """Test initialization with custom n_components (int)."""
        reducer = PCAReducer(n_components=10)
        assert reducer.n_components == 10

    def test_custom_n_components_float(self):
        """Test initialization with variance ratio (float)."""
        reducer = PCAReducer(n_components=0.95)
        assert reducer.n_components == 0.95

    def test_custom_random_state(self):
        """Test initialization with custom random_state."""
        reducer = PCAReducer(random_state=123)
        assert reducer.random_state == 123

    def test_kwargs_stored(self):
        """Test that additional kwargs are stored."""
        reducer = PCAReducer(whiten=True)
        assert "whiten" in reducer._kwargs
        assert reducer._kwargs["whiten"] is True


class TestPCAFit:
    """Tests for PCAReducer fit method."""

    def test_fit_stores_model(self, high_dim_embeddings):
        """Test that fit creates and stores model."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)
        assert reducer._model is not None

    def test_fit_returns_self(self, high_dim_embeddings):
        """Test that fit returns self for chaining."""
        reducer = PCAReducer(n_components=50)
        result = reducer.fit(high_dim_embeddings)
        assert result is reducer

    def test_components_available_after_fit(self, high_dim_embeddings):
        """Test that components are available after fit."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)
        components = reducer.components_
        assert components is not None
        assert components.shape == (50, 384)

    def test_explained_variance_available(self, high_dim_embeddings):
        """Test that explained_variance_ratio is available after fit."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)
        var_ratio = reducer.explained_variance_ratio_
        assert var_ratio is not None
        assert len(var_ratio) == 50
        assert np.all(var_ratio >= 0)
        assert np.all(var_ratio <= 1)


class TestPCATransform:
    """Tests for PCAReducer transform method."""

    def test_transform_after_fit(self, high_dim_embeddings):
        """Test transforming data after fit."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)

        test_data = np.random.randn(10, 384).astype(np.float32)
        transformed = reducer.transform(test_data)

        assert transformed.shape == (10, 50)

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises RuntimeError."""
        reducer = PCAReducer(n_components=50)
        test_data = np.random.randn(10, 100).astype(np.float32)

        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            reducer.transform(test_data)

    def test_transform_shape(self, high_dim_embeddings):
        """Test that transform returns correct shape."""
        reducer = PCAReducer(n_components=30)
        reducer.fit(high_dim_embeddings)

        for n_samples in [1, 5, 10, 50]:
            test_data = np.random.randn(n_samples, 384).astype(np.float32)
            transformed = reducer.transform(test_data)
            assert transformed.shape == (n_samples, 30)

    def test_transform_consistency(self, high_dim_embeddings):
        """Test that transform is consistent across calls."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)

        test_data = np.random.randn(10, 384).astype(np.float32)
        result1 = reducer.transform(test_data)
        result2 = reducer.transform(test_data)

        np.testing.assert_array_almost_equal(result1, result2)


class TestPCAFitTransform:
    """Tests for PCAReducer fit_transform method."""

    def test_fit_transform_combines_operations(self, high_dim_embeddings):
        """Test that fit_transform works correctly."""
        reducer = PCAReducer(n_components=50)
        result = reducer.fit_transform(high_dim_embeddings)

        assert result.shape == (100, 50)

    def test_fit_transform_matches_fit_then_transform(self, high_dim_embeddings):
        """Test that fit_transform matches fit() then transform()."""
        reducer1 = PCAReducer(n_components=50, random_state=42)
        result_fit_transform = reducer1.fit_transform(high_dim_embeddings)

        reducer2 = PCAReducer(n_components=50, random_state=42)
        reducer2.fit(high_dim_embeddings)
        result_separate = reducer2.transform(high_dim_embeddings)

        np.testing.assert_array_almost_equal(
            result_fit_transform, result_separate, decimal=4
        )


class TestPCAProperties:
    """Tests for PCAReducer properties."""

    def test_explained_variance_ratio_property(self, high_dim_embeddings):
        """Test explained_variance_ratio_ property."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)

        var_ratio = reducer.explained_variance_ratio_
        assert var_ratio is not None
        assert len(var_ratio) == 50

    def test_components_property(self, high_dim_embeddings):
        """Test components_ property."""
        reducer = PCAReducer(n_components=50)
        reducer.fit(high_dim_embeddings)

        components = reducer.components_
        assert components is not None
        assert components.shape == (50, 384)

    def test_properties_none_before_fit(self):
        """Test that properties return None before fit."""
        reducer = PCAReducer(n_components=50)

        assert reducer.explained_variance_ratio_ is None
        assert reducer.components_ is None


class TestPCAEdgeCases:
    """Tests for PCAReducer edge cases."""

    def test_n_components_greater_than_features(self, small_embeddings):
        """Test when n_components > min(n_samples, n_features) raises error."""
        # small_embeddings: (20, 50) -> min is 20
        reducer = PCAReducer(n_components=100)  # More than min(20, 50)

        # sklearn raises ValueError when n_components > min(n_samples, n_features)
        with pytest.raises(ValueError, match="must be between 0 and"):
            reducer.fit(small_embeddings)

    def test_variance_ratio_095(self, high_dim_embeddings):
        """Test with 95% variance ratio."""
        reducer = PCAReducer(n_components=0.95)
        result = reducer.fit_transform(high_dim_embeddings)

        # Should reduce dimensions while keeping 95% variance
        assert result.shape[0] == 100
        assert result.shape[1] < 384  # Reduced

        # Check that we captured 95% variance
        var_ratio = reducer.explained_variance_ratio_
        assert var_ratio is not None
        assert np.sum(var_ratio) >= 0.94  # Close to 95%

    def test_n_components_equals_min_dim(self, small_embeddings):
        """Test when n_components equals min(n_samples, n_features)."""
        # small_embeddings: (20, 50) -> min is 20
        reducer = PCAReducer(n_components=20)
        result = reducer.fit_transform(small_embeddings)

        assert result.shape == (20, 20)


class TestPCAReproducibility:
    """Tests for PCAReducer reproducibility."""

    def test_same_random_state_same_results(self, high_dim_embeddings):
        """Test that same random_state produces identical results."""
        reducer1 = PCAReducer(n_components=50, random_state=42)
        result1 = reducer1.fit_transform(high_dim_embeddings)

        reducer2 = PCAReducer(n_components=50, random_state=42)
        result2 = reducer2.fit_transform(high_dim_embeddings)

        np.testing.assert_array_almost_equal(result1, result2)


class TestPCARepr:
    """Tests for PCAReducer string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        reducer = PCAReducer(n_components=25)
        repr_str = repr(reducer)

        assert "PCAReducer" in repr_str
        assert "n_components=25" in repr_str
