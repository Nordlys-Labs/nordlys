"""Unit tests for MiniBatchKMeansClusterer."""

import numpy as np
import pytest

from nordlys.clustering import MiniBatchKMeansClusterer


class TestMiniBatchKMeansInitialization:
    """Tests for MiniBatchKMeansClusterer initialization."""

    def test_default_initialization(self):
        """Test MiniBatchKMeansClusterer with default parameters."""
        clusterer = MiniBatchKMeansClusterer()
        assert clusterer.n_clusters == 20
        assert clusterer.max_iter == 100
        assert clusterer.batch_size == 1024
        assert clusterer.random_state == 42

    def test_custom_n_clusters(self):
        """Test initialization with custom n_clusters."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=15)
        assert clusterer.n_clusters == 15

    def test_custom_random_state(self):
        """Test initialization with custom random_state."""
        clusterer = MiniBatchKMeansClusterer(random_state=123)
        assert clusterer.random_state == 123


class TestMiniBatchKMeansFit:
    """Tests for MiniBatchKMeansClusterer fit method."""

    def test_fit_simple_data(self, simple_5d_clusters):
        """Test fit on simple well-separated clusters."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        assert clusterer._model is not None
        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_fit_returns_self(self, simple_5d_clusters):
        """Test that fit returns self for chaining."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        result = clusterer.fit(simple_5d_clusters)
        assert result is clusterer

    def test_labels_shape(self, simple_5d_clusters):
        """Test that labels have correct shape."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert clusterer.labels_.shape == (60,)
        assert len(np.unique(clusterer.labels_)) == 3

    def test_inertia_available(self, simple_5d_clusters):
        """Test that inertia is available after fit."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)
        assert isinstance(clusterer.inertia_, float)
        assert clusterer.inertia_ > 0


class TestMiniBatchKMeansPredict:
    """Tests for MiniBatchKMeansClusterer predict method."""

    def test_predict_after_fit(self, simple_5d_clusters):
        """Test predict after fitting."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_predict_before_fit_raises_error(self):
        """Test that predict before fit raises RuntimeError."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        test_data = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="must be fitted before predict"):
            clusterer.predict(test_data)


class TestMiniBatchKMeansProperties:
    """Tests for MiniBatchKMeansClusterer properties."""

    def test_cluster_centers_before_fit_raises(self):
        """Test that accessing cluster_centers_ before fit raises error."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.cluster_centers_

    def test_labels_before_fit_raises(self):
        """Test that accessing labels_ before fit raises error."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="must be fitted first"):
            _ = clusterer.labels_

    def test_n_clusters_property(self):
        """Test n_clusters_ property."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=5)
        assert clusterer.n_clusters_ == 5


class TestMiniBatchKMeansCUDA:
    """Tests for MiniBatchKMeansClusterer CUDA implementation."""

    @pytest.fixture(autouse=True)
    def check_cuda_available(self):
        """Skip tests if CUDA is not available."""
        pytest.importorskip("cuml")

    def test_cuda_fit_simple_data(self, simple_5d_clusters):
        """Test CUDA fit on simple well-separated clusters."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)

        assert clusterer._model is not None
        assert clusterer.labels_.shape == (60,)
        assert clusterer.cluster_centers_.shape == (3, 5)

    def test_cuda_predict_after_fit(self, simple_5d_clusters):
        """Test CUDA predict after fitting."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(10, 5)
        predictions = clusterer.predict(test_data)

        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))

    def test_cuda_inertia_available(self, simple_5d_clusters):
        """Test that inertia is available after CUDA fit."""
        clusterer = MiniBatchKMeansClusterer(n_clusters=3, device="cuda")
        clusterer.fit(simple_5d_clusters)
        assert isinstance(clusterer.inertia_, float)
        assert clusterer.inertia_ > 0


class TestMiniBatchKMeansParity:
    """Tests for CPU vs CUDA parity."""

    @pytest.fixture(autouse=True)
    def check_cuda_available(self):
        """Skip tests if CUDA is not available."""
        pytest.importorskip("cuml")

    def test_cpu_cuda_labels_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same labels with same random state."""
        cpu_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        np.testing.assert_array_equal(cpu_clusterer.labels_, cuda_clusterer.labels_)

    def test_cpu_cuda_centroids_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same centroids with same random state."""
        cpu_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        np.testing.assert_array_almost_equal(
            cpu_clusterer.cluster_centers_, cuda_clusterer.cluster_centers_, decimal=4
        )

    def test_cpu_cuda_predict_parity(self, simple_5d_clusters):
        """Test that CPU and CUDA produce same predictions."""
        cpu_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cpu"
        )
        cpu_clusterer.fit(simple_5d_clusters)

        cuda_clusterer = MiniBatchKMeansClusterer(
            n_clusters=3, random_state=42, device="cuda"
        )
        cuda_clusterer.fit(simple_5d_clusters)

        test_data = np.random.randn(20, 5)
        cpu_preds = cpu_clusterer.predict(test_data)
        cuda_preds = cuda_clusterer.predict(test_data)

        np.testing.assert_array_equal(cpu_preds, cuda_preds)
