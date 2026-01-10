"""Unit tests for clustering metrics."""

import numpy as np

from nordlys.clustering.metrics import (
    ClusterInfo,
    ClusterMetrics,
    compute_cluster_metrics,
)


class TestClusterInfo:
    """Tests for ClusterInfo dataclass."""

    def test_creation(self):
        """Test creating ClusterInfo."""
        centroid = np.array([1.0, 2.0, 3.0])
        info = ClusterInfo(
            cluster_id=0,
            size=10,
            centroid=centroid,
            model_accuracies={"model1": 0.9, "model2": 0.8},
        )

        assert info.cluster_id == 0
        assert info.size == 10
        np.testing.assert_array_equal(info.centroid, centroid)
        assert info.model_accuracies == {"model1": 0.9, "model2": 0.8}

    def test_default_model_accuracies(self):
        """Test default empty model_accuracies."""
        centroid = np.array([1.0, 2.0])
        info = ClusterInfo(cluster_id=1, size=5, centroid=centroid)

        assert info.model_accuracies == {}

    def test_repr(self):
        """Test __repr__ format."""
        centroid = np.array([1.0, 2.0])
        info = ClusterInfo(
            cluster_id=5,
            size=20,
            centroid=centroid,
            model_accuracies={"a": 0.9, "b": 0.8},
        )

        repr_str = repr(info)
        assert "ClusterInfo" in repr_str
        assert "id=5" in repr_str
        assert "size=20" in repr_str
        assert "models=2" in repr_str


class TestClusterMetrics:
    """Tests for ClusterMetrics dataclass."""

    def test_creation(self):
        """Test creating ClusterMetrics."""
        metrics = ClusterMetrics(
            silhouette_score=0.75,
            n_clusters=5,
            n_samples=100,
            cluster_sizes=[20, 20, 20, 20, 20],
            inertia=150.0,
        )

        assert metrics.silhouette_score == 0.75
        assert metrics.n_clusters == 5
        assert metrics.n_samples == 100
        assert metrics.cluster_sizes == [20, 20, 20, 20, 20]
        assert metrics.inertia == 150.0

    def test_min_cluster_size(self):
        """Test min_cluster_size property."""
        metrics = ClusterMetrics(
            silhouette_score=0.5,
            n_clusters=3,
            n_samples=30,
            cluster_sizes=[5, 10, 15],
        )

        assert metrics.min_cluster_size == 5

    def test_max_cluster_size(self):
        """Test max_cluster_size property."""
        metrics = ClusterMetrics(
            silhouette_score=0.5,
            n_clusters=3,
            n_samples=30,
            cluster_sizes=[5, 10, 15],
        )

        assert metrics.max_cluster_size == 15

    def test_avg_cluster_size(self):
        """Test avg_cluster_size property."""
        metrics = ClusterMetrics(
            silhouette_score=0.5,
            n_clusters=3,
            n_samples=30,
            cluster_sizes=[5, 10, 15],
        )

        assert metrics.avg_cluster_size == 10.0

    def test_properties_with_empty_cluster_sizes(self):
        """Test properties with None/empty cluster_sizes."""
        metrics = ClusterMetrics(
            silhouette_score=0.5,
            n_clusters=0,
            n_samples=0,
            cluster_sizes=None,
        )

        assert metrics.min_cluster_size == 0
        assert metrics.max_cluster_size == 0
        assert metrics.avg_cluster_size == 0.0

    def test_repr(self):
        """Test __repr__ format."""
        metrics = ClusterMetrics(
            silhouette_score=0.75,
            n_clusters=5,
            n_samples=100,
            cluster_sizes=[15, 20, 25, 30, 10],
        )

        repr_str = repr(metrics)
        assert "ClusterMetrics" in repr_str
        assert "n_clusters=5" in repr_str
        assert "silhouette=0.75" in repr_str

    def test_repr_with_none_silhouette(self):
        """Test __repr__ with None silhouette score."""
        metrics = ClusterMetrics(
            silhouette_score=None,
            n_clusters=1,
            n_samples=10,
            cluster_sizes=[10],
        )

        repr_str = repr(metrics)
        assert "N/A" in repr_str

    def test_none_values_handled(self):
        """Test that None values are handled gracefully."""
        metrics = ClusterMetrics(
            silhouette_score=None,
            n_clusters=0,
            n_samples=None,
            cluster_sizes=None,
            inertia=None,
        )

        assert metrics.silhouette_score is None
        assert metrics.n_samples is None
        assert metrics.inertia is None


class TestComputeClusterMetrics:
    """Tests for compute_cluster_metrics function."""

    def test_basic_computation(self, simple_5d_clusters):
        """Test basic metric computation."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        metrics = compute_cluster_metrics(simple_5d_clusters, labels)

        assert metrics.n_clusters == 3
        assert metrics.n_samples == 60

    def test_silhouette_score_computed(self, simple_5d_clusters):
        """Test that silhouette score is computed."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        metrics = compute_cluster_metrics(simple_5d_clusters, labels)

        assert metrics.silhouette_score is not None
        assert -1 <= metrics.silhouette_score <= 1

    def test_cluster_sizes_correct(self, simple_5d_clusters):
        """Test that cluster sizes are computed correctly."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        metrics = compute_cluster_metrics(simple_5d_clusters, labels)

        assert metrics.cluster_sizes is not None
        assert sum(metrics.cluster_sizes) == 60

    def test_n_samples_correct(self, simple_5d_clusters):
        """Test that n_samples is correct."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        metrics = compute_cluster_metrics(simple_5d_clusters, labels)

        assert metrics.n_samples == 60

    def test_inertia_passed_through(self, simple_5d_clusters):
        """Test that inertia is passed through."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        metrics = compute_cluster_metrics(simple_5d_clusters, labels, inertia=123.45)

        assert metrics.inertia == 123.45

    def test_single_cluster_silhouette(self):
        """Test silhouette with single cluster."""
        embeddings = np.random.randn(20, 5)
        labels = np.zeros(20, dtype=int)  # All same cluster

        metrics = compute_cluster_metrics(embeddings, labels)

        # Single cluster has silhouette of 0
        assert metrics.silhouette_score == 0.0
        assert metrics.n_clusters == 1

    def test_with_noise_labels(self, simple_5d_clusters):
        """Test handling of noise label (-1)."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(simple_5d_clusters)

        # Add some noise labels
        labels[0:5] = -1

        metrics = compute_cluster_metrics(simple_5d_clusters, labels)

        # Should still work, excluding noise
        assert metrics.n_clusters == 3
        assert metrics.n_samples == 60  # Total samples including noise
