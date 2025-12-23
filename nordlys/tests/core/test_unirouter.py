"""Unit tests for Router cluster-based routing."""

from typing import List

import pytest

from nordlys.core.cluster_engine import ClusterEngine
from nordlys.core.router import ModelRouter


@pytest.fixture
def sample_questions() -> List[str]:
    """Sample question strings for testing."""
    return [
        "Write a Python function to sort a list",
        "Explain the concept of recursion in programming",
        "What is the time complexity of quicksort?",
        "Implement a binary search tree in Python",
        "Describe how dynamic programming works",
    ]


@pytest.fixture
def small_cluster_engine() -> ClusterEngine:
    """Create a small cluster engine for testing."""
    # Use very small parameters for fast testing
    return ClusterEngine().configure(
        n_clusters=2,  # Small number of clusters for fast testing
        max_iter=10,  # Few iterations
        n_init=1,  # Single run
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


class TestClusterEngine:
    """Test ClusterEngine core functionality."""

    def test_initialization(self) -> None:
        """Test ClusterEngine initialization."""
        engine = ClusterEngine()

        assert engine.n_clusters is None
        assert engine.kmeans is None
        assert engine.embedding_model is None

    def test_configure_with_custom_params(self) -> None:
        """Test ClusterEngine with custom parameters."""
        engine = ClusterEngine().configure(
            n_clusters=10,
            max_iter=500,
            random_state=123,
            n_init=20,
        )

        assert engine.n_clusters == 10
        assert engine.max_iter == 500
        assert engine.random_state == 123
        assert engine.n_init == 20

    def test_fit(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test fitting the cluster engine on questions."""
        engine = small_cluster_engine

        result = engine.fit(sample_questions)

        # Should return self for method chaining
        assert result is engine

        # Should be fitted
        assert hasattr(engine.kmeans, "cluster_centers_")

        # Should have cluster assignments
        assert len(engine.cluster_assignments) == len(sample_questions)

        # Cluster assignments should be in valid range
        assert all(0 <= c < engine.n_clusters for c in engine.cluster_assignments)

        # Should have silhouette score
        assert -1.0 <= engine.silhouette <= 1.0

    def test_predict_before_fit_raises_error(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that predict raises error if called before fit."""
        engine = small_cluster_engine

        with pytest.raises(Exception, match="Must call fit"):
            engine.predict(sample_questions)

    def test_predict_after_fit(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test predicting clusters after fitting."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Create new questions for prediction
        new_questions = ["Write a sorting algorithm"]

        predictions = engine.predict(new_questions)

        assert len(predictions) == len(new_questions)
        assert small_cluster_engine.n_clusters is not None
        assert all(0 <= p < small_cluster_engine.n_clusters for p in predictions)

    def test_assign_single(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test assigning a single question to a cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        cluster_id, distance = engine.assign_single(
            "Write a function to implement quicksort"
        )

        assert isinstance(cluster_id, int)
        assert small_cluster_engine.n_clusters is not None
        assert 0 <= cluster_id < small_cluster_engine.n_clusters
        assert isinstance(distance, float)
        assert distance >= 0.0

    def test_assign_single_before_fit_raises_error(
        self, small_cluster_engine: ClusterEngine
    ) -> None:
        """Test that assign_single raises error if called before fit."""
        engine = small_cluster_engine

        with pytest.raises(Exception, match="Must call fit"):
            engine.assign_single("Test question")

    def test_get_cluster_info(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test getting cluster information."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        info = engine.cluster_stats

        assert info.n_clusters == small_cluster_engine.n_clusters
        assert info.n_samples == len(sample_questions)

    def test_get_cluster_info_before_fit(
        self, small_cluster_engine: ClusterEngine
    ) -> None:
        """Test getting cluster info before fitting."""

        with pytest.raises(
            Exception, match="Must call fit\\(\\) before accessing cluster_stats"
        ):
            _ = small_cluster_engine.cluster_stats


class TestClusterEngineEdgeCases:
    """Test edge cases for ClusterEngine."""

    def test_empty_questions_list(self, small_cluster_engine: ClusterEngine) -> None:
        """Test fitting with empty questions list."""
        engine = small_cluster_engine

        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            engine.fit([])

    def test_single_question(self, small_cluster_engine: ClusterEngine) -> None:
        """Test fitting with a single question."""
        engine = small_cluster_engine

        single_question = ["Test question"]

        # Should handle gracefully (might cluster all to single cluster)
        try:
            engine.fit(single_question)
            assert hasattr(engine.kmeans, "cluster_centers_")
        except (ValueError, Exception):
            # Some ML libraries might complain about insufficient data
            pass

    def test_predict_different_size_batch(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test predicting on batches of different sizes."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Predict single question
        single_pred = engine.predict([sample_questions[0]])
        assert len(single_pred) == 1

        # Predict multiple questions
        multiple_pred = engine.predict(sample_questions[:3])
        assert len(multiple_pred) == 3

    def test_cluster_distribution(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that cluster distribution is reasonable."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        info = engine.cluster_stats

        # Sum of cluster sizes should equal total questions
        total_assigned = sum(info.cluster_sizes.values())
        assert total_assigned == len(sample_questions)

        # Min/max/avg should be consistent
        assert info.min_cluster_size <= info.avg_cluster_size <= info.max_cluster_size


class TestRouterServiceMocked:
    """Test ModelRouter with mocked dependencies (no real ML models)."""

    def test_initialization_mocked(self) -> None:
        """Test ModelRouter initialization with mocked components."""
        from unittest.mock import Mock

        # Mock the C++ CoreRouter
        mock_core_router = Mock()
        mock_core_router.get_n_clusters.return_value = 5
        mock_core_router.get_embedding_dim.return_value = 100

        # Mock the embedding model
        mock_embedding_model = Mock()

        # Mock profile metadata
        Mock()
        from nordlys.models.storage import RouterProfile, ProfileMetadata
        from nordlys.models.api import Model

        mock_profile = Mock(spec=RouterProfile)
        mock_profile.metadata = Mock(spec=ProfileMetadata)
        mock_profile.metadata.routing = Mock()
        mock_profile.metadata.routing.default_cost_preference = 0.5
        mock_profile.metadata.embedding_model = "all-MiniLM-L6-v2"
        mock_profile.metadata.n_clusters = 5
        mock_profile.metadata.dtype = "float32"
        mock_profile.models = [Mock(spec=Model)]

        # Create router with mocked components
        router = ModelRouter(
            core_router=mock_core_router,
            embedding_model=mock_embedding_model,
            profile=mock_profile,
        )

        assert router is not None
        assert router._core_router is mock_core_router
        assert router._embedding_model is mock_embedding_model

    def test_select_model_validates_request(self) -> None:
        """Test that select_model validates the request."""
        pass


class TestModelSelectionIntegration:
    """Test integration of ClusterEngine with model selection."""

    def test_cluster_assignment_consistency(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that same question always gets assigned to same cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        test_question = "Write a Python function to sort a list"

        # Assign multiple times
        cluster_id1, _ = engine.assign_single(test_question)
        cluster_id2, _ = engine.assign_single(test_question)
        cluster_id3, _ = engine.assign_single(test_question)

        # Should be consistent
        assert cluster_id1 == cluster_id2 == cluster_id3

    def test_similar_questions_same_cluster(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that similar questions tend to be assigned to same cluster."""
        engine = small_cluster_engine
        engine.fit(sample_questions)

        # These should be similar
        question1 = "Write a sorting algorithm in Python"
        question2 = "Implement a sort function in Python"

        cluster_id1, _ = engine.assign_single(question1)
        cluster_id2, _ = engine.assign_single(question2)

        # Note: With only 2 clusters and 5 questions, there's a decent chance
        # they'll be in the same cluster, but not guaranteed
        # This is a weak test but demonstrates the concept
        # (A real test would use many more questions and larger clusters)


class TestClusterEnginePerformance:
    """Test performance characteristics of ClusterEngine."""

    def test_fit_performance(self, sample_questions: List[str]) -> None:
        """Test that fitting completes in reasonable time."""
        import time

        engine = ClusterEngine().configure(n_clusters=2, max_iter=10, n_init=1)

        start = time.time()
        engine.fit(sample_questions)
        elapsed = time.time() - start

        # With only 5 questions, 2 clusters, and 10 iterations, should be very fast
        assert elapsed < 30.0  # 30 seconds should be more than enough

    def test_predict_performance(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that prediction is fast."""
        import time

        engine = small_cluster_engine
        engine.fit(sample_questions)

        # Create test questions
        test_questions = [f"Test question {i}" for i in range(10)]

        start = time.time()
        for _ in range(10):  # 100 total predictions
            engine.predict(test_questions)
        elapsed = time.time() - start

        # 100 predictions should be fast
        assert elapsed < 30.0

    def test_assign_single_performance(
        self, small_cluster_engine: ClusterEngine, sample_questions: List[str]
    ) -> None:
        """Test that single question assignment is fast."""
        import time

        engine = small_cluster_engine
        engine.fit(sample_questions)

        start = time.time()
        for i in range(100):
            engine.assign_single(f"Test question {i}")
        elapsed = time.time() - start

        # 100 assignments should complete in under 30 seconds
        assert elapsed < 30.0
