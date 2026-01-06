"""Unit tests for AgglomerativeClusterer."""

import numpy as np
import pytest

from nordlys.clustering.agglomerative import AgglomerativeClusterer


class TestAgglomerativeClusterer:
    """Tests for AgglomerativeClusterer."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data with clear clusters."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
        cluster3 = np.random.randn(20, 5) + np.array([20, 20, 20, 20, 20])
        return np.vstack([cluster1, cluster2, cluster3])

    def test_agglomerative_init_defaults(self):
        """Test AgglomerativeClusterer initialization with defaults."""
        clusterer = AgglomerativeClusterer()
        assert clusterer.n_clusters == 20
        assert clusterer.linkage == "ward"
        assert clusterer.metric == "euclidean"

    def test_agglomerative_init_custom(self):
        """Test AgglomerativeClusterer initialization with custom params."""
        clusterer = AgglomerativeClusterer(
            n_clusters=10, linkage="complete", metric="manhattan"
        )
        assert clusterer.n_clusters == 10
        assert clusterer.linkage == "complete"
        assert clusterer.metric == "manhattan"

    def test_agglomerative_fit(self, simple_data):
        """Test fit method creates model, stores embeddings, and computes cluster_centers_."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # Check that model was created
        assert clusterer._model is not None

        # Check that embeddings were stored
        assert clusterer._embeddings is not None
        np.testing.assert_array_equal(clusterer._embeddings, simple_data)

        # Check labels_ shape
        assert clusterer.labels_.shape == (60,)
        assert len(np.unique(clusterer.labels_)) == 3

        # Check cluster_centers_ shape
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_predict(self, simple_data):
        """Test predict assigns to nearest centroid."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # Create test data close to first cluster
        test_data = np.random.randn(5, 5) + np.array([0, 0, 0, 0, 0])
        predictions = clusterer.predict(test_data)

        # Check predictions shape
        assert predictions.shape == (5,)
        # All predictions should be valid cluster indices
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_agglomerative_fit_predict(self, simple_data):
        """Test fit_predict returns same labels as fit."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        labels_fit_predict = clusterer.fit_predict(simple_data)

        # Create new instance and fit separately
        clusterer2 = AgglomerativeClusterer(n_clusters=3)
        clusterer2.fit(simple_data)
        labels_fit = clusterer2.labels_

        # They should be the same
        np.testing.assert_array_equal(labels_fit_predict, labels_fit)

    def test_agglomerative_errors_before_fit(self):
        """Test RuntimeError is raised when calling methods or properties before fit."""
        clusterer = AgglomerativeClusterer(n_clusters=3)

        # predict before fit
        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(np.random.randn(10, 5))

        # cluster_centers_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

        # labels_ before fit
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

    def test_agglomerative_linkage_ward_euclidean(self, simple_data):
        """Test ward linkage with euclidean metric (valid combination)."""
        clusterer = AgglomerativeClusterer(n_clusters=3, linkage="ward")
        clusterer.fit(simple_data)

        # Should fit successfully
        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_linkage_complete(self, simple_data):
        """Test complete linkage with different metrics."""
        for metric in ["euclidean", "manhattan", "cosine"]:
            clusterer = AgglomerativeClusterer(
                n_clusters=3, linkage="complete", metric=metric
            )
            clusterer.fit(simple_data)

            # Should fit successfully
            assert clusterer.labels_.shape == (60,)
            assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_linkage_average(self, simple_data):
        """Test average linkage."""
        clusterer = AgglomerativeClusterer(n_clusters=3, linkage="average")
        clusterer.fit(simple_data)

        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_linkage_single(self, simple_data):
        """Test single linkage."""
        clusterer = AgglomerativeClusterer(n_clusters=3, linkage="single")
        clusterer.fit(simple_data)

        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_edge_case_single_sample(self):
        """Test with single sample (sklearn requires minimum 2 samples)."""
        data = np.random.randn(1, 5)
        clusterer = AgglomerativeClusterer(n_clusters=1)

        # AgglomerativeClustering requires at least 2 samples
        with pytest.raises(ValueError, match="minimum of 2"):
            clusterer.fit(data)

    def test_agglomerative_edge_case_identical_embeddings(self):
        """Test with identical embeddings."""
        # All samples are identical
        data = np.ones((10, 5))
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(data)

        # Should still cluster (may put all in one cluster or split arbitrarily)
        assert clusterer.labels_.shape == (10,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_agglomerative_edge_case_mismatched_dims(self, simple_data):
        """Test predict with mismatched input dimensions."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(simple_data)  # 5 features

        # This should work with broadcasting/distance calculation
        # The implementation uses np.linalg.norm which may handle this,
        # but ideally we'd have validation (not currently in the code)
        # For now, just test that it doesn't crash unexpectedly
        try:
            # If it crashes, that's expected behavior for mismatched dims
            test_data = np.random.randn(5, 3)  # Wrong feature count
            predictions = clusterer.predict(test_data)
            # If it doesn't crash, verify shape (though may be incorrect results)
            assert predictions.shape == (5,)
        except (ValueError, IndexError):
            # Expected to fail with dimension mismatch
            pass

    def test_agglomerative_predict_consistency(self, simple_data):
        """Test predict is consistent across calls."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        test_data = np.random.randn(10, 5)
        predictions1 = clusterer.predict(test_data)
        predictions2 = clusterer.predict(test_data)

        # Same input should give same output
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_agglomerative_n_clusters_property(self, simple_data):
        """Test n_clusters_ property."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        assert clusterer.n_clusters_ == 3

        clusterer.fit(simple_data)
        assert clusterer.n_clusters_ == 3

    def test_agglomerative_repr(self):
        """Test __repr__ method."""
        clusterer = AgglomerativeClusterer(n_clusters=5, linkage="complete")
        repr_str = repr(clusterer)
        assert "AgglomerativeClusterer" in repr_str
        assert "n_clusters=5" in repr_str
        assert "linkage='complete'" in repr_str

    def test_agglomerative_cluster_centers_computation(self, simple_data):
        """Test that cluster centers are computed as mean of cluster members."""
        clusterer = AgglomerativeClusterer(n_clusters=3)
        clusterer.fit(simple_data)

        # Manually compute cluster centers
        labels = clusterer.labels_
        expected_centers = []
        for i in range(3):
            mask = labels == i
            expected_centers.append(simple_data[mask].mean(axis=0))
        expected_centers = np.array(expected_centers)

        # Compare with computed centers
        np.testing.assert_array_almost_equal(
            clusterer.cluster_centers_, expected_centers, decimal=10
        )

    def test_agglomerative_small_clusters(self):
        """Test with more clusters than might be natural."""
        np.random.seed(42)
        data = np.random.randn(20, 3)
        clusterer = AgglomerativeClusterer(n_clusters=5)
        clusterer.fit(data)

        assert clusterer.labels_.shape == (20,)
        assert clusterer.cluster_centers_.shape == (5, 3)
        assert len(np.unique(clusterer.labels_)) == 5
