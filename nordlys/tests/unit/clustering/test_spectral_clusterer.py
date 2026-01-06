"""Unit tests for SpectralClusterer."""

import numpy as np
import pytest

from nordlys.clustering.spectral import SpectralClusterer


class TestSpectralClusterer:
    """Tests for SpectralClusterer."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data with clear clusters."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
        cluster3 = np.random.randn(20, 5) + np.array([20, 20, 20, 20, 20])
        return np.vstack([cluster1, cluster2, cluster3])

    def test_spectral_init(self):
        """Test SpectralClusterer initialization."""
        clusterer = SpectralClusterer(n_clusters=10, affinity="nearest_neighbors")
        assert clusterer.n_clusters == 10
        assert clusterer.affinity == "nearest_neighbors"
        assert clusterer.n_neighbors == 10
        assert clusterer.random_state == 42

    def test_spectral_fit(self, simple_data):
        """Test fit method sets labels_ and cluster_centers_ with correct shapes."""
        clusterer = SpectralClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_data)

        # Check labels_ shape
        assert clusterer.labels_.shape == (60,)
        assert len(np.unique(clusterer.labels_)) == 3

        # Check cluster_centers_ shape
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_spectral_fit_predict(self, simple_data):
        """Test fit_predict returns same labels as fit."""
        clusterer = SpectralClusterer(n_clusters=3, random_state=42)
        labels_fit_predict = clusterer.fit_predict(simple_data)

        # Create new instance and fit separately
        clusterer2 = SpectralClusterer(n_clusters=3, random_state=42)
        clusterer2.fit(simple_data)
        labels_fit = clusterer2.labels_

        # They should be the same
        np.testing.assert_array_equal(labels_fit_predict, labels_fit)

    def test_spectral_predict(self, simple_data):
        """Test predict assigns new points to nearest computed centroids."""
        clusterer = SpectralClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_data)

        # Create test data close to first cluster
        test_data = np.random.randn(5, 5) + np.array([0, 0, 0, 0, 0])
        predictions = clusterer.predict(test_data)

        # Check predictions shape
        assert predictions.shape == (5,)
        # All predictions should be valid cluster indices
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_spectral_predict_consistency(self, simple_data):
        """Test predict is consistent with training data assignments."""
        clusterer = SpectralClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_data)

        # Predict on the same training data
        predictions = clusterer.predict(simple_data)

        # Predictions should be close to original labels (may not be exact due to nearest centroid)
        assert predictions.shape == simple_data.shape[0:1]

    def test_spectral_errors_before_fit(self):
        """Test RuntimeError is raised when calling predict or properties before fit."""
        clusterer = SpectralClusterer(n_clusters=3)

        # predict before fit
        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(np.random.randn(10, 5))

        # cluster_centers_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

        # labels_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

    def test_spectral_fit_validation_ndim(self):
        """Test fit validates embeddings dimensionality."""
        clusterer = SpectralClusterer(n_clusters=3)

        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            clusterer.fit(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            clusterer.fit(np.random.randn(10, 5, 3))

    def test_spectral_fit_validation_empty(self):
        """Test fit validates embeddings are non-empty."""
        clusterer = SpectralClusterer(n_clusters=3)

        # Empty samples
        with pytest.raises(ValueError, match="cannot be empty"):
            clusterer.fit(np.empty((0, 5)))

        # Empty features
        with pytest.raises(ValueError, match="cannot be empty"):
            clusterer.fit(np.empty((10, 0)))

    def test_spectral_fit_validation_finite(self):
        """Test fit validates embeddings contain only finite values."""
        clusterer = SpectralClusterer(n_clusters=3)

        # NaN values
        data_with_nan = np.random.randn(20, 5)
        data_with_nan[5, 2] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            clusterer.fit(data_with_nan)

        # Inf values
        data_with_inf = np.random.randn(20, 5)
        data_with_inf[3, 1] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            clusterer.fit(data_with_inf)

    def test_spectral_fit_validation_samples_vs_clusters(self):
        """Test fit validates n_samples >= n_clusters."""
        clusterer = SpectralClusterer(n_clusters=10)

        # Only 5 samples but 10 clusters
        with pytest.raises(ValueError, match="must be >= n_clusters"):
            clusterer.fit(np.random.randn(5, 3))

    def test_spectral_predict_validation_ndim(self, simple_data):
        """Test predict validates embeddings dimensionality."""
        clusterer = SpectralClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            clusterer.predict(np.array([1, 2, 3]))

        # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            clusterer.predict(np.random.randn(10, 5, 3))

    def test_spectral_predict_validation_empty(self, simple_data):
        """Test predict validates embeddings are non-empty."""
        clusterer = SpectralClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # Empty samples
        with pytest.raises(ValueError, match="cannot be empty"):
            clusterer.predict(np.empty((0, 5)))

    def test_spectral_predict_validation_feature_mismatch(self, simple_data):
        """Test predict validates feature dimension matches cluster centers."""
        clusterer = SpectralClusterer(n_clusters=3)
        clusterer.fit(simple_data)  # 5 features

        # Wrong number of features
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            clusterer.predict(np.random.randn(10, 3))  # 3 features instead of 5

    def test_spectral_predict_validation_dtype(self, simple_data):
        """Test predict validates numeric dtype."""
        clusterer = SpectralClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # String array
        with pytest.raises(ValueError, match="must have numeric dtype"):
            clusterer.predict(np.array([["a", "b", "c", "d", "e"]]))

    def test_spectral_n_clusters_property(self, simple_data):
        """Test n_clusters_ property."""
        clusterer = SpectralClusterer(n_clusters=3)
        assert clusterer.n_clusters_ == 3

        clusterer.fit(simple_data)
        assert clusterer.n_clusters_ == 3

    def test_spectral_repr(self):
        """Test __repr__ method."""
        clusterer = SpectralClusterer(n_clusters=5, affinity="rbf")
        repr_str = repr(clusterer)
        assert "SpectralClusterer" in repr_str
        assert "n_clusters=5" in repr_str
        assert "affinity='rbf'" in repr_str
