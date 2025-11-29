"""Tests for ClusterEngine."""

import numpy as np
import pytest

from adaptive_router.core.cluster_engine import ClusterEngine


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
