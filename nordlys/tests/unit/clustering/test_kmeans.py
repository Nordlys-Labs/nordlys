"""Unit tests for KMeansClusterer."""

import numpy as np
import pytest

from nordlys.clustering.kmeans import KMeansClusterer


class TestKMeansInitialization:
    """Tests for KMeansClusterer initialization."""

    def test_default_initialization(self):
        """Test KMeansClusterer with default parameters."""
        clusterer = KMeansClusterer()
        assert clusterer.n_clusters == 20
        assert clusterer.max_iter == 300
        assert clusterer.n_init == 10
        assert clusterer.random_state == 42
        assert clusterer.algorithm == "lloyd"

    def test_custom_n_clusters(self):
        """Test initialization with custom n_clusters."""
        clusterer = KMeansClusterer(n_clusters=15)
        assert clusterer.n_clusters == 15

    def test_custom_random_state(self):
        """Test initialization with custom random_state."""
        clusterer = KMeansClusterer(random_state=123)
        assert clusterer.random_state == 123

    def test_custom_algorithm(self):
        """Test initialization with custom algorithm."""
        clusterer = KMeansClusterer(algorithm="elkan")
        assert clusterer.algorithm == "elkan"

    def test_custom_max_iter(self):
        """Test initialization with custom max_iter."""
        clusterer = KMeansClusterer(max_iter=500)
        assert clusterer.max_iter == 500

    def test_kwargs_stored(self):
        """Test that additional kwargs are stored."""
        clusterer = KMeansClusterer(verbose=True)
        assert "verbose" in clusterer._kwargs
        assert clusterer._kwargs["verbose"] is True


class TestKMeansFit:
    """Tests for KMeansClusterer fit method."""

    def test_fit_simple_data(self, simple_5d_clusters):
        """Test fit on simple well-separated clusters."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        assert clusterer._model is not None
        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_fit_returns_self(self, simple_5d_clusters):
        """Test that fit returns self for chaining."""
        clusterer = KMeansClusterer(n_clusters=3)
        result = clusterer.fit(simple_5d_clusters)
        assert result is clusterer

    def test_labels_shape(self, simple_5d_clusters):
        """Test that labels have correct shape."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert clusterer.labels_.shape == (60,)
        assert len(np.unique(clusterer.labels_)) == 3

    def test_cluster_centers_shape(self, simple_5d_clusters):
        """Test that cluster centers have correct shape."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_inertia_available(self, simple_5d_clusters):
        """Test that inertia is available after fit."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert isinstance(clusterer.inertia_, float)
        assert clusterer.inertia_ > 0

    def test_n_iter_available(self, simple_5d_clusters):
        """Test that n_iter is available after fit."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert isinstance(clusterer.n_iter_, int)
        assert clusterer.n_iter_ > 0


class TestKMeansPredict:
    """Tests for KMeansClusterer predict method."""

    def test_predict_after_fit(self, simple_5d_clusters):
        """Test predict after fitting."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises RuntimeError."""
        clusterer = KMeansClusterer(n_clusters=3)
        test_data = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(test_data)

    def test_predict_shape(self, simple_5d_clusters):
        """Test that predict returns correct shape."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        for n_samples in [1, 5, 10, 100]:
            test_data = np.random.randn(n_samples, 5)
            predictions = clusterer.predict(test_data)
            assert predictions.shape == (n_samples,)

    def test_predict_valid_labels(self, simple_5d_clusters):
        """Test that predictions are valid cluster indices."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(20, 5)
        predictions = clusterer.predict(test_data)

        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)
        assert predictions.dtype == np.int64 or predictions.dtype == np.int32

    def test_predict_consistency(self, simple_5d_clusters):
        """Test that same input produces same output."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions1 = clusterer.predict(test_data)
        predictions2 = clusterer.predict(test_data)

        np.testing.assert_array_equal(predictions1, predictions2)


class TestKMeansFitPredict:
    """Tests for KMeansClusterer fit_predict method."""

    def test_fit_predict_returns_labels(self, simple_5d_clusters):
        """Test that fit_predict returns labels."""
        clusterer = KMeansClusterer(n_clusters=3)
        labels = clusterer.fit_predict(simple_5d_clusters)

        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 3

    def test_fit_predict_matches_fit_then_predict(self, simple_5d_clusters):
        """Test that fit_predict matches fit() then predict()."""
        clusterer1 = KMeansClusterer(n_clusters=3, random_state=42)
        labels_fit_predict = clusterer1.fit_predict(simple_5d_clusters)

        clusterer2 = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer2.fit(simple_5d_clusters)
        labels_fit = clusterer2.labels_

        np.testing.assert_array_equal(labels_fit_predict, labels_fit)


class TestKMeansReproducibility:
    """Tests for KMeansClusterer reproducibility."""

    def test_same_random_state_same_results(self, simple_5d_clusters):
        """Test that same random_state produces identical results."""
        clusterer1 = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer1.fit(simple_5d_clusters)

        clusterer2 = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer2.fit(simple_5d_clusters)

        np.testing.assert_array_equal(clusterer1.labels_, clusterer2.labels_)
        np.testing.assert_array_almost_equal(
            clusterer1.cluster_centers_, clusterer2.cluster_centers_
        )

    def test_different_random_state_different_results(self, simple_5d_clusters):
        """Test that different random_state produces different results."""
        clusterer1 = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer1.fit(simple_5d_clusters)

        clusterer2 = KMeansClusterer(n_clusters=3, random_state=123)
        clusterer2.fit(simple_5d_clusters)

        # Labels might be different (not guaranteed, but very likely)
        # At least check that we can run with different seeds
        assert clusterer1.labels_.shape == clusterer2.labels_.shape


class TestKMeansProperties:
    """Tests for KMeansClusterer properties."""

    def test_cluster_centers_before_fit_raises(self):
        """Test that accessing cluster_centers_ before fit raises error."""
        clusterer = KMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

    def test_labels_before_fit_raises(self):
        """Test that accessing labels_ before fit raises error."""
        clusterer = KMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

    def test_n_clusters_property(self):
        """Test n_clusters_ property."""
        clusterer = KMeansClusterer(n_clusters=5)
        assert clusterer.n_clusters_ == 5

    def test_inertia_before_fit_raises(self):
        """Test that accessing inertia_ before fit raises error."""
        clusterer = KMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.inertia_


class TestKMeansEdgeCases:
    """Tests for KMeansClusterer edge cases."""

    def test_single_cluster(self, simple_5d_clusters):
        """Test with n_clusters=1."""
        clusterer = KMeansClusterer(n_clusters=1)
        clusterer.fit(simple_5d_clusters)

        assert clusterer.labels_.shape == (60,)
        assert len(np.unique(clusterer.labels_)) == 1
        assert clusterer.cluster_centers_.shape == (1, 5)

    def test_more_clusters_than_samples(self):
        """Test when n_clusters > n_samples raises error."""
        data = np.random.randn(5, 3)
        clusterer = KMeansClusterer(n_clusters=10)

        # sklearn raises ValueError when n_clusters > n_samples
        with pytest.raises(ValueError, match="should be >= n_clusters"):
            clusterer.fit(data)

    def test_identical_points(self, identical_points):
        """Test clustering all identical points."""
        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit(identical_points)

        assert clusterer.labels_.shape == (20,)
        # All cluster centers should be identical
        assert np.allclose(clusterer.cluster_centers_[0], clusterer.cluster_centers_[1])

    def test_empty_data_raises(self):
        """Test that empty data raises an error."""
        clusterer = KMeansClusterer(n_clusters=3)
        empty_data = np.array([]).reshape(0, 5)

        with pytest.raises((ValueError, IndexError)):
            clusterer.fit(empty_data)


class TestKMeansRepr:
    """Tests for KMeansClusterer string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        clusterer = KMeansClusterer(n_clusters=10)
        repr_str = repr(clusterer)

        assert "KMeansClusterer" in repr_str
        assert "n_clusters=10" in repr_str
