"""Unit tests for BisectingKMeansClusterer."""

import numpy as np
import pytest

from nordlys.clustering import BisectingKMeansClusterer


class TestBisectingKMeansClusterer:
    """Tests for BisectingKMeansClusterer."""

    def test_fit_simple_data(self, simple_5d_clusters):
        """Test fit on simple well-separated clusters."""
        clusterer = BisectingKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_labels_contiguous(self):
        """Test that labels are always contiguous 0..k-1."""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        clusterer = BisectingKMeansClusterer(n_clusters=5, random_state=42)
        clusterer.fit(data)

        unique_labels = sorted(set(clusterer.labels_))
        assert unique_labels == list(range(5)), (
            f"Expected [0,1,2,3,4], got {unique_labels}"
        )

    def test_n_clusters_matches_labels(self):
        """Test that n_clusters_ matches actual number of unique labels."""
        np.random.seed(123)
        data = np.random.randn(80, 8)

        clusterer = BisectingKMeansClusterer(n_clusters=4, random_state=42)
        clusterer.fit(data)

        actual_clusters = len(set(clusterer.labels_))
        assert clusterer.n_clusters_ == actual_clusters

    def test_cluster_centers_shape(self, simple_5d_clusters):
        """Test that cluster centers have correct shape."""
        clusterer = BisectingKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_predict_after_fit(self, simple_5d_clusters):
        """Test predict after fitting."""
        clusterer = BisectingKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_fit_predict(self, simple_5d_clusters):
        """Test fit_predict returns labels."""
        clusterer = BisectingKMeansClusterer(n_clusters=3)
        labels = clusterer.fit_predict(simple_5d_clusters)

        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 3

    def test_single_cluster(self):
        """Test with n_clusters=1."""
        np.random.seed(42)
        data = np.random.randn(50, 5)

        clusterer = BisectingKMeansClusterer(n_clusters=1)
        clusterer.fit(data)

        assert len(set(clusterer.labels_)) == 1
        assert clusterer.cluster_centers_.shape == (1, 5)

    def test_reproducibility(self):
        """Test that same random_state produces same results."""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        clusterer1 = BisectingKMeansClusterer(n_clusters=4, random_state=42)
        clusterer1.fit(data)

        clusterer2 = BisectingKMeansClusterer(n_clusters=4, random_state=42)
        clusterer2.fit(data)

        np.testing.assert_array_equal(clusterer1.labels_, clusterer2.labels_)
        np.testing.assert_array_almost_equal(
            clusterer1.cluster_centers_, clusterer2.cluster_centers_
        )

    def test_access_before_fit_raises(self):
        """Test that accessing properties before fit raises error."""
        clusterer = BisectingKMeansClusterer(n_clusters=3)

        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(np.random.randn(10, 5))


class TestBisectingKMeansCUDA:
    """Tests for BisectingKMeansClusterer CUDA implementation."""

    @pytest.fixture(autouse=True)
    def check_cuda_available(self):
        """Skip tests if CUDA is not available."""
        pytest.importorskip("cuml")

    def test_cuda_fit_simple_data(self, simple_5d_clusters):
        """Test CUDA fit on simple well-separated clusters."""
        clusterer = BisectingKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)

        assert clusterer._model is not None
        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_cuda_predict_after_fit(self, simple_5d_clusters):
        """Test CUDA predict after fitting."""
        clusterer = BisectingKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_cuda_inertia_available(self, simple_5d_clusters):
        """Test that inertia is available after CUDA fit."""
        clusterer = BisectingKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)
        assert isinstance(clusterer.inertia_, float)
        assert clusterer.inertia_ > 0


class TestBisectingKMeansParity:
    """Tests for CPU vs CUDA parity."""

    @pytest.fixture(autouse=True)
    def check_cuda_available(self):
        """Skip tests if CUDA is not available."""
        pytest.importorskip("cuml")

    def test_cpu_cuda_labels_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same labels with same random state."""
        cpu_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        np.testing.assert_array_equal(cpu_clusterer.labels_, cuda_clusterer.labels_)

    def test_cpu_cuda_centroids_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same centroids with same random state."""
        cpu_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        np.testing.assert_array_almost_equal(
            cpu_clusterer.cluster_centers_, cuda_clusterer.cluster_centers_, decimal=4
        )

    def test_cpu_cuda_predict_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same predictions."""
        cpu_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = BisectingKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(20, 5)
        cpu_preds = cpu_clusterer.predict(test_data)
        cuda_preds = cuda_clusterer.predict(test_data)

        np.testing.assert_array_equal(cpu_preds, cuda_preds)
