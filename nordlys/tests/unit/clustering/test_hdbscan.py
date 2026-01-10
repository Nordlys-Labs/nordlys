"""Unit tests for HDBSCANClusterer."""

import numpy as np
import pytest

from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer


class TestHDBSCANInitialization:
    """Tests for HDBSCANClusterer initialization."""

    def test_default_initialization(self):
        """Test HDBSCANClusterer with default parameters."""
        clusterer = HDBSCANClusterer()
        assert clusterer.min_cluster_size == 100
        assert clusterer.min_samples is None  # Uses min_cluster_size by default
        assert clusterer.metric == "euclidean"

    def test_custom_min_cluster_size(self):
        """Test initialization with custom min_cluster_size."""
        clusterer = HDBSCANClusterer(min_cluster_size=20)
        assert clusterer.min_cluster_size == 20

    def test_custom_min_samples(self):
        """Test initialization with custom min_samples."""
        clusterer = HDBSCANClusterer(min_samples=10)
        assert clusterer.min_samples == 10

    def test_custom_metric(self):
        """Test initialization with custom metric."""
        clusterer = HDBSCANClusterer(metric="manhattan")
        assert clusterer.metric == "manhattan"

    def test_kwargs_stored(self):
        """Test that additional kwargs are stored."""
        clusterer = HDBSCANClusterer(alpha=1.5)
        assert "alpha" in clusterer._kwargs
        assert clusterer._kwargs["alpha"] == 1.5


class TestHDBSCANFit:
    """Tests for HDBSCANClusterer fit method."""

    def test_fit_simple_data(self, simple_5d_clusters):
        """Test fit on simple well-separated clusters."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        assert clusterer._model is not None
        assert clusterer.labels_.shape == (60,)

    def test_fit_returns_self(self, simple_5d_clusters):
        """Test that fit returns self for chaining."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        result = clusterer.fit(simple_5d_clusters)
        assert result is clusterer

    def test_fit_stores_embeddings(self, simple_5d_clusters):
        """Test that fit stores embeddings."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)
        assert clusterer._embeddings is not None
        np.testing.assert_array_equal(clusterer._embeddings, simple_5d_clusters)

    def test_labels_include_possible_noise(self, simple_5d_clusters):
        """Test that HDBSCAN may produce noise label -1."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        # All labels should be >= -1
        assert np.all(clusterer.labels_ >= -1)

    def test_cluster_centers_computed(self, simple_5d_clusters):
        """Test that cluster centers are computed."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        centers = clusterer.cluster_centers_
        assert centers is not None
        assert centers.ndim == 2
        assert centers.shape[1] == 5  # Feature dimension


class TestHDBSCANPredict:
    """Tests for HDBSCANClusterer predict method."""

    def test_predict_after_fit(self, simple_5d_clusters):
        """Test predict assigns to nearest centroid."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(5, 5) + np.array([0, 0, 0, 0, 0])
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (5,)
        assert np.all(predictions >= -1)  # Can include noise

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises RuntimeError."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        test_data = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(test_data)

    def test_predict_consistency(self, simple_5d_clusters):
        """Test predict is consistent across calls."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions1 = clusterer.predict(test_data)
        predictions2 = clusterer.predict(test_data)

        np.testing.assert_array_equal(predictions1, predictions2)


class TestHDBSCANFitPredict:
    """Tests for HDBSCANClusterer fit_predict method."""

    def test_fit_predict_returns_labels(self, simple_5d_clusters):
        """Test that fit_predict returns labels."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        labels = clusterer.fit_predict(simple_5d_clusters)

        assert labels.shape == (60,)
        assert np.all(labels >= -1)


class TestHDBSCANProperties:
    """Tests for HDBSCANClusterer properties."""

    def test_n_clusters_excludes_noise(self, simple_5d_clusters):
        """Test that n_clusters_ excludes noise (-1)."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        n_clusters = clusterer.n_clusters_
        unique_labels = np.unique(clusterer.labels_)
        valid_labels = unique_labels[unique_labels >= 0]

        assert n_clusters == len(valid_labels)

    def test_labels_property(self, simple_5d_clusters):
        """Test labels_ property."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        labels = clusterer.labels_
        assert labels is not None
        assert labels.shape == (60,)

    def test_cluster_centers_property(self, simple_5d_clusters):
        """Test cluster_centers_ property."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(simple_5d_clusters)

        centers = clusterer.cluster_centers_
        assert centers is not None
        assert centers.ndim == 2

    def test_properties_before_fit_raise(self):
        """Test that properties raise before fit."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)

        with pytest.raises(RuntimeError):
            _ = clusterer.labels_

        with pytest.raises(RuntimeError):
            _ = clusterer.cluster_centers_


class TestHDBSCANEdgeCases:
    """Tests for HDBSCANClusterer edge cases."""

    def test_identical_points(self, identical_points):
        """Test clustering all identical points."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        clusterer.fit(identical_points)

        # Should still produce some clustering
        assert clusterer.labels_.shape == (20,)

    def test_small_min_cluster_size(self, simple_5d_clusters):
        """Test with very small min_cluster_size."""
        clusterer = HDBSCANClusterer(min_cluster_size=2)
        clusterer.fit(simple_5d_clusters)

        assert clusterer.labels_.shape == (60,)


class TestHDBSCANRepr:
    """Tests for HDBSCANClusterer string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        clusterer = HDBSCANClusterer(min_cluster_size=15)
        repr_str = repr(clusterer)

        assert "HDBSCANClusterer" in repr_str
        assert "min_cluster_size=15" in repr_str
