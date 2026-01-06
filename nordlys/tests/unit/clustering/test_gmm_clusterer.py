"""Unit tests for GMMClusterer."""

import numpy as np
import pytest

from nordlys.clustering.gmm import GMMClusterer


class TestGMMClusterer:
    """Tests for GMMClusterer."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data with clear clusters."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 5) + np.array([10, 10, 10, 10, 10])
        cluster3 = np.random.randn(30, 5) + np.array([20, 20, 20, 20, 20])
        return np.vstack([cluster1, cluster2, cluster3])

    def test_gmm_init_defaults(self):
        """Test GMMClusterer initialization with default params."""
        clusterer = GMMClusterer()
        assert clusterer.n_components == 20
        assert clusterer.covariance_type == "full"
        assert clusterer.max_iter == 100
        assert clusterer.n_init == 1
        assert clusterer.random_state == 42

    def test_gmm_init_custom(self):
        """Test GMMClusterer initialization with custom params."""
        clusterer = GMMClusterer(
            n_components=10,
            covariance_type="diag",
            max_iter=50,
            n_init=3,
            random_state=123,
        )
        assert clusterer.n_components == 10
        assert clusterer.covariance_type == "diag"
        assert clusterer.max_iter == 50
        assert clusterer.n_init == 3
        assert clusterer.random_state == 123

    def test_gmm_fit(self, simple_data):
        """Test fit method sets labels_ and cluster_centers_ with correct shapes."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        # Check labels_ shape
        assert clusterer.labels_.shape == (90,)
        assert len(np.unique(clusterer.labels_)) <= 3

        # Check cluster_centers_ shape (GMM uses means_)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_gmm_fit_weights_and_covariances(self, simple_data):
        """Test fit computes weights_ and covariances_ with correct shapes."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        # Check weights_ shape
        assert clusterer.weights_.shape == (3,)
        # Weights should sum to ~1
        np.testing.assert_almost_equal(np.sum(clusterer.weights_), 1.0, decimal=5)

        # Check covariances_ shape (for full covariance)
        assert clusterer.covariances_.shape == (3, 5, 5)

    def test_gmm_predict(self, simple_data):
        """Test predict assigns cluster labels with correct shape."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        # Create test data
        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        # Check predictions shape
        assert predictions.shape == (10,)
        # All predictions should be valid cluster indices
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_gmm_predict_proba(self, simple_data):
        """Test predict_proba returns probabilities with correct shape and row sums."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        # Create test data
        test_data = np.random.randn(10, 5)
        probabilities = clusterer.predict_proba(test_data)

        # Check shape
        assert probabilities.shape == (10, 3)

        # Check that probabilities sum to ~1 for each row
        row_sums = np.sum(probabilities, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=5)

        # Check all probabilities are in [0, 1]
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_gmm_fit_predict(self, simple_data):
        """Test fit_predict returns same labels as fit."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        labels_fit_predict = clusterer.fit_predict(simple_data)

        # Create new instance and fit separately
        clusterer2 = GMMClusterer(n_components=3, random_state=42)
        clusterer2.fit(simple_data)
        labels_fit = clusterer2.labels_

        # They should be the same
        np.testing.assert_array_equal(labels_fit_predict, labels_fit)

    def test_gmm_errors_before_fit(self):
        """Test RuntimeError is raised when calling methods or properties before fit."""
        clusterer = GMMClusterer(n_components=3)

        # predict before fit
        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(np.random.randn(10, 5))

        # predict_proba before fit
        with pytest.raises(RuntimeError, match="must be fitted before predict_proba"):
            clusterer.predict_proba(np.random.randn(10, 5))

        # cluster_centers_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

        # labels_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

        # weights_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.weights_

        # covariances_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.covariances_

        # bic_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.bic_

    def test_gmm_n_clusters_property(self, simple_data):
        """Test n_clusters_ property."""
        clusterer = GMMClusterer(n_components=3)
        assert clusterer.n_clusters_ == 3

        clusterer.fit(simple_data)
        assert clusterer.n_clusters_ == 3

    def test_gmm_bic_property(self, simple_data):
        """Test bic_ property returns a float."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        bic = clusterer.bic_
        assert isinstance(bic, float)

    def test_gmm_covariance_types(self, simple_data):
        """Test different covariance types."""
        for cov_type in ["full", "tied", "diag", "spherical"]:
            clusterer = GMMClusterer(
                n_components=3, covariance_type=cov_type, random_state=42
            )
            clusterer.fit(simple_data)

            # Should fit successfully
            assert clusterer.labels_.shape == (90,)
            assert clusterer.cluster_centers_.shape == (3, 5)

    def test_gmm_predict_consistency(self, simple_data):
        """Test predict is consistent across calls."""
        clusterer = GMMClusterer(n_components=3, random_state=42)
        clusterer.fit(simple_data)

        test_data = np.random.randn(10, 5)
        predictions1 = clusterer.predict(test_data)
        predictions2 = clusterer.predict(test_data)

        # Same input should give same output
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_gmm_repr(self):
        """Test __repr__ method."""
        clusterer = GMMClusterer(n_components=5, covariance_type="diag")
        repr_str = repr(clusterer)
        assert "GMMClusterer" in repr_str
        assert "n_components=5" in repr_str
        assert "covariance_type='diag'" in repr_str

    def test_gmm_small_data(self):
        """Test GMM with minimal data."""
        np.random.seed(42)
        # Just enough samples for 2 components
        data = np.random.randn(10, 3)
        clusterer = GMMClusterer(n_components=2, random_state=42)
        clusterer.fit(data)

        assert clusterer.labels_.shape == (10,)
        assert clusterer.cluster_centers_.shape == (2, 3)
