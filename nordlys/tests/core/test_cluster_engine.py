"""Tests for ClusterEngine."""

import numpy as np
import pytest

from nordlys.core.cluster_engine import ClusterEngine


@pytest.fixture
def sample_questions() -> list[str]:
    """Create sample code questions for testing."""
    return [
        "How do I sort a list in Python?",
        "What is a lambda function?",
        "How to reverse a string in JavaScript?",
        "Explain async/await in Python",
        "How to use map in JavaScript?",
    ]


@pytest.fixture
def cluster_engine() -> ClusterEngine:
    """Create a ClusterEngine with test configuration."""
    return ClusterEngine().configure(
        n_clusters=2,
        max_iter=100,
        random_state=42,
    )


class TestClusterEngineInitialization:
    """Test ClusterEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test ClusterEngine initializes with default parameters."""
        engine = ClusterEngine()
        assert engine.n_clusters is None
        assert len(engine.cluster_assignments) == 0

    def test_configure_custom_parameters(self) -> None:
        """Test ClusterEngine with custom parameters."""
        engine = ClusterEngine().configure(
            n_clusters=10,
            max_iter=200,
            random_state=123,
        )
        assert engine.n_clusters == 10
        assert engine.max_iter == 200
        assert engine.random_state == 123


class TestClusterEngineFit:
    """Test ClusterEngine fitting."""

    def test_fit_updates_state(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test that fit updates engine state correctly."""
        cluster_engine.fit(sample_questions)

        assert hasattr(cluster_engine.kmeans, "cluster_centers_")
        assert len(cluster_engine.cluster_assignments) == len(sample_questions)

        assert cluster_engine.silhouette >= -1.0
        assert cluster_engine.silhouette <= 1.0

    def test_fit_returns_self(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test that fit returns self for method chaining."""
        result = cluster_engine.fit(sample_questions)
        assert result is cluster_engine

    def test_fit_with_empty_questions(self, cluster_engine: ClusterEngine) -> None:
        """Test fit raises error with empty questions list."""
        with pytest.raises(ValueError):
            cluster_engine.fit([])


class TestClusterEnginePredict:
    """Test ClusterEngine prediction."""

    def test_predict_assigns_cluster(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test predict assigns cluster to new question."""
        cluster_engine.fit(sample_questions)

        new_question = "How to use list comprehension in Python?"
        cluster_id, _ = cluster_engine.assign_single(new_question)

        assert isinstance(cluster_id, (int, np.integer))
        assert cluster_engine.n_clusters is not None
        assert 0 <= cluster_id < cluster_engine.n_clusters

    def test_predict_before_fit_raises_error(
        self, cluster_engine: ClusterEngine
    ) -> None:
        """Test predict raises error when not fitted."""
        question = "Test question"

        with pytest.raises(Exception, match="Must call fit"):
            cluster_engine.assign_single(question)

    def test_predict_batch(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test batch prediction of multiple questions."""
        cluster_engine.fit(sample_questions)

        new_questions = [
            "Python list sorting",
            "JavaScript array methods",
        ]

        assignments = cluster_engine.predict(new_questions)

        assert len(assignments) == len(new_questions)
        assert all(isinstance(a, (int, np.integer)) for a in assignments)
        assert all(0 <= a < cluster_engine.n_clusters for a in assignments)


class TestClusterEngineAnalysis:
    """Test ClusterEngine analysis methods."""

    def test_get_cluster_summary(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test get_cluster_summary returns statistics."""
        cluster_engine.fit(sample_questions)
        summary = cluster_engine.cluster_stats

        assert summary.n_clusters == cluster_engine.n_clusters
        assert summary.n_samples == len(sample_questions)
        assert len(summary.cluster_sizes) == cluster_engine.n_clusters


class TestClusterEngineDtype:
    """Test ClusterEngine dtype auto-detection and preservation."""

    def test_initial_dtype_is_none(self) -> None:
        """Test embedding_dtype is None before fit."""
        engine = ClusterEngine()
        assert engine.embedding_dtype is None

    def test_dtype_detected_on_fit(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test dtype is auto-detected during fit."""
        assert cluster_engine.embedding_dtype is None
        cluster_engine.fit(sample_questions)
        assert cluster_engine.embedding_dtype is not None
        assert cluster_engine.embedding_dtype in [np.float32, np.float64]

    def test_default_dtype_is_float32(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test most embedding models output float32."""
        cluster_engine.fit(sample_questions)
        assert cluster_engine.embedding_dtype == np.float32

    def test_get_dtype_string_returns_correct_value(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test get_dtype_string() returns 'float32' or 'float64'."""
        cluster_engine.fit(sample_questions)
        dtype_str = cluster_engine.get_dtype_string()
        assert dtype_str in ["float32", "float64"]

    def test_get_dtype_string_before_fit_raises(
        self, cluster_engine: ClusterEngine
    ) -> None:
        """Test get_dtype_string() raises error before fit."""
        with pytest.raises(ValueError, match="dtype not yet detected"):
            cluster_engine.get_dtype_string()

    def test_cluster_centers_match_dtype(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test cluster centers use detected dtype."""
        cluster_engine.fit(sample_questions)
        centers_dtype = cluster_engine.kmeans.cluster_centers_.dtype
        assert centers_dtype == cluster_engine.embedding_dtype

    def test_from_fitted_state_with_dtype(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test from_fitted_state() uses provided dtype."""
        cluster_engine.fit(sample_questions)

        # Restore with explicit float64
        restored = ClusterEngine.from_fitted_state(
            cluster_centers=cluster_engine.kmeans.cluster_centers_,
            n_clusters=cluster_engine.n_clusters,
            embedding_model=cluster_engine.embedding_model,
            embedding_model_name=cluster_engine.embedding_model_name,
            dtype="float64",
        )

        assert restored.embedding_dtype == np.float64
        assert restored.kmeans.cluster_centers_.dtype == np.float64

    def test_from_fitted_state_defaults_float32(
        self, cluster_engine: ClusterEngine, sample_questions: list[str]
    ) -> None:
        """Test from_fitted_state() defaults to float32."""
        cluster_engine.fit(sample_questions)

        restored = ClusterEngine.from_fitted_state(
            cluster_centers=cluster_engine.kmeans.cluster_centers_,
            n_clusters=cluster_engine.n_clusters,
            embedding_model=cluster_engine.embedding_model,
            embedding_model_name=cluster_engine.embedding_model_name,
        )

        assert restored.embedding_dtype == np.float32

    def test_normalized_features_preserve_dtype(
        self, cluster_engine: ClusterEngine
    ) -> None:
        """Test _normalize_features preserves input dtype."""
        float32_features = np.random.randn(5, 10).astype(np.float32)
        float64_features = np.random.randn(5, 10).astype(np.float64)

        norm32 = cluster_engine._normalize_features(float32_features)
        norm64 = cluster_engine._normalize_features(float64_features)

        assert norm32.dtype == np.float32
        assert norm64.dtype == np.float64
