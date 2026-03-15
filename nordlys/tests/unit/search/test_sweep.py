"""Tests for search module."""

from __future__ import annotations


import numpy as np
import pytest

from nordlys.search import (
    ParameterSweep,
    SweepResult,
    SweepResults,
    cluster_balance_constraint,
    cluster_count_scorer,
    max_clusters_constraint,
    min_clusters_constraint,
    silhouette_scorer,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings with clear cluster structure."""
    np.random.seed(42)
    return np.vstack(
        [
            np.random.randn(30, 10) + [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            np.random.randn(30, 10) - [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            np.random.randn(30, 10) + [0, 0, 5, 5, 0, 0, 5, 5, 0, 0],
        ]
    )


@pytest.fixture
def sample_sweep_results(sample_embeddings) -> SweepResults:
    """Create sample sweep results for testing."""
    sweep = ParameterSweep(
        param_grids={
            "kmeans": {"n_clusters": [2, 3, 4]},
        },
        random_state=42,
    )
    return sweep.run(sample_embeddings, algorithms=["kmeans"])


class TestSweepScorerProtocol:
    """Tests for scorer protocol and built-in scorers."""

    def test_silhouette_scorer_returns_float(self, sample_sweep_results):
        """Silhouette scorer should return a float score."""
        scorer = silhouette_scorer()
        result = sample_sweep_results.results[0]
        score = scorer(result)
        assert isinstance(score, float)

    def test_silhouette_scorer_values(self, sample_sweep_results):
        """Silhouette scorer should return values between -1 and 1."""
        scorer = silhouette_scorer()
        for result in sample_sweep_results.results:
            score = scorer(result)
            assert -1.0 <= score <= 1.0

    def test_cluster_count_scorer_prefers_target(self):
        """Cluster count scorer should prefer results close to target."""

        class DummyMetrics10:
            silhouette_score = 0.5
            n_clusters = 10
            cluster_sizes = [10] * 10
            n_samples = 100
            inertia = None

        class DummyMetrics15:
            silhouette_score = 0.5
            n_clusters = 15
            cluster_sizes = [10] * 15
            n_samples = 100
            inertia = None

        scorer = cluster_count_scorer(target=10, penalty=0.1)

        # Exact match
        result_10 = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 10},
            metrics=DummyMetrics10(),
            labels=np.zeros(100),
            centroids=np.zeros((10, 5)),
            clusterer=None,
        )

        # Off by 5
        result_15 = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 15},
            metrics=DummyMetrics15(),
            labels=np.zeros(100),
            centroids=np.zeros((15, 5)),
            clusterer=None,
        )

        score_10 = scorer(result_10)
        score_15 = scorer(result_15)

        assert score_10 > score_15


class TestSweepConstraintProtocol:
    """Tests for constraint protocol and built-in constraints."""

    def test_min_clusters_constraint_passes(self):
        """Min clusters constraint should pass when above threshold."""
        constraint = min_clusters_constraint(min_k=5)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 10
            cluster_sizes = [10] * 10
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 10},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((10, 5)),
            clusterer=None,
        )

        assert constraint(result) is True

    def test_min_clusters_constraint_fails(self):
        """Min clusters constraint should fail when below threshold."""
        constraint = min_clusters_constraint(min_k=15)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 10
            cluster_sizes = [10] * 10
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 10},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((10, 5)),
            clusterer=None,
        )

        assert constraint(result) is False

    def test_max_clusters_constraint_passes(self):
        """Max clusters constraint should pass when below threshold."""
        constraint = max_clusters_constraint(max_k=15)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 10
            cluster_sizes = [10] * 10
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 10},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((10, 5)),
            clusterer=None,
        )

        assert constraint(result) is True

    def test_max_clusters_constraint_fails(self):
        """Max clusters constraint should fail when above threshold."""
        constraint = max_clusters_constraint(max_k=5)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 10
            cluster_sizes = [10] * 10
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 10},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((10, 5)),
            clusterer=None,
        )

        assert constraint(result) is False

    def test_cluster_balance_constraint_balanced(self):
        """Cluster balance constraint should pass for balanced clusters."""
        constraint = cluster_balance_constraint(max_cv=0.6)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 4
            cluster_sizes = [25, 25, 25, 25]  # Perfectly balanced
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 4},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((4, 5)),
            clusterer=None,
        )

        assert constraint(result)

    def test_cluster_balance_constraint_unbalanced(self):
        """Cluster balance constraint should fail for unbalanced clusters."""
        constraint = cluster_balance_constraint(max_cv=0.6)

        class DummyMetrics:
            silhouette_score = 0.5
            n_clusters = 4
            cluster_sizes = [80, 10, 5, 5]  # Very unbalanced
            n_samples = 100
            inertia = None

        result = SweepResult(
            algorithm="kmeans",
            params={"n_clusters": 4},
            metrics=DummyMetrics(),
            labels=np.zeros(100),
            centroids=np.zeros((4, 5)),
            clusterer=None,
        )

        assert not constraint(result)


class TestSweepResults:
    """Tests for SweepResults class."""

    def test_len(self, sample_sweep_results):
        """Length should return number of results."""
        assert len(sample_sweep_results) == 3

    def test_iter(self, sample_sweep_results):
        """Iteration should yield all results."""
        results = list(sample_sweep_results)
        assert len(results) == 3

    def test_filter(self, sample_sweep_results):
        """Filter should return results matching constraint."""
        constraint = min_clusters_constraint(min_k=3)
        filtered = sample_sweep_results.filter(constraint)
        assert len(filtered) > 0
        for result in filtered.results:
            assert result.n_clusters >= 3

    def test_rank(self, sample_sweep_results):
        """Rank should return sorted results."""
        scorer = silhouette_scorer()
        ranked = sample_sweep_results.rank(scorer)
        assert len(ranked) == 3
        # Check descending order
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_select_with_scorer_only(self, sample_sweep_results):
        """Select should work with scorer only."""
        scorer = silhouette_scorer()
        selected = sample_sweep_results.select(scorer=scorer)
        assert selected is not None

    def test_select_with_constraints(self, sample_sweep_results):
        """Select should filter by constraints."""
        scorer = silhouette_scorer()
        constraints = [min_clusters_constraint(min_k=3)]
        selected = sample_sweep_results.select(scorer=scorer, constraints=constraints)
        if selected:
            assert selected.n_clusters >= 3

    def test_select_returns_none_when_no_match(self, sample_sweep_results):
        """Select should return None when no results pass constraints."""
        # Create results with very high silhouette but wrong cluster count
        scorer = silhouette_scorer()
        constraints = [min_clusters_constraint(min_k=100)]
        selected = sample_sweep_results.select(scorer=scorer, constraints=constraints)
        assert selected is None

    def test_best_by_silhouette(self, sample_sweep_results):
        """Best by silhouette should return highest silhouette result."""
        best = sample_sweep_results.best_by_silhouette()
        assert best is not None
        for result in sample_sweep_results.results:
            if result != best:
                assert (result.metrics.silhouette_score or 0) <= (
                    best.metrics.silhouette_score or 0
                )

    def test_best_by_n_clusters_exact_match(self, sample_sweep_results):
        """Best by n_clusters should return exact match when available."""
        best = sample_sweep_results.best_by_n_clusters(target=3)
        assert best is not None
        assert best.n_clusters == 3

    def test_best_by_n_clusters_closest(self):
        """Best by n_clusters should return closest when no exact match."""
        # Create results with specific cluster counts

        class DummyMetrics5:
            silhouette_score = 0.5
            n_clusters = 5
            cluster_sizes = [20] * 5
            n_samples = 100
            inertia = None

        class DummyMetrics15:
            silhouette_score = 0.5
            n_clusters = 15
            cluster_sizes = [10] * 15
            n_samples = 100
            inertia = None

        results = SweepResults(
            [
                SweepResult(
                    algorithm="kmeans",
                    params={"n_clusters": 5},
                    metrics=DummyMetrics5(),
                    labels=np.zeros(100),
                    centroids=np.zeros((5, 5)),
                    clusterer=None,
                ),
                SweepResult(
                    algorithm="kmeans",
                    params={"n_clusters": 15},
                    metrics=DummyMetrics15(),
                    labels=np.zeros(100),
                    centroids=np.zeros((15, 5)),
                    clusterer=None,
                ),
            ]
        )

        best = results.best_by_n_clusters(target=12)
        assert best is not None
        assert best.n_clusters == 15  # Closest to 12

    def test_filter_by_algorithm(self, sample_sweep_results):
        """Filter by algorithm should return matching results."""
        filtered = sample_sweep_results.filter_by_algorithm("kmeans")
        assert len(filtered) == len(sample_sweep_results)


class TestParameterSweep:
    """Tests for ParameterSweep class."""

    def test_default_creation(self):
        """Sweep should be created with defaults."""
        sweep = ParameterSweep()
        assert sweep.random_state == 42
        assert sweep.param_grids is not None

    def test_custom_creation(self):
        """Sweep should accept custom parameters."""
        grids = {"kmeans": {"n_clusters": [5, 10]}}
        sweep = ParameterSweep(param_grids=grids, random_state=123)
        assert sweep.random_state == 123
        assert sweep.param_grids == grids

    def test_run_returns_sweep_results(self, sample_embeddings):
        """Run should return SweepResults."""
        sweep = ParameterSweep(param_grids={"kmeans": {"n_clusters": [2, 3]}})
        results = sweep.run(sample_embeddings, algorithms=["kmeans"])
        assert isinstance(results, SweepResults)
        assert len(results) == 2

    def test_run_with_multiple_algorithms(self, sample_embeddings):
        """Run should handle multiple algorithms."""
        sweep = ParameterSweep(
            param_grids={
                "kmeans": {"n_clusters": [2]},
                "agglomerative": {"n_clusters": [2]},
            }
        )
        results = sweep.run(sample_embeddings, algorithms=["kmeans", "agglomerative"])
        assert len(results) == 2
        algorithms = set(r.algorithm for r in results.results)
        assert "kmeans" in algorithms
        assert "agglomerative" in algorithms

    def test_run_skips_unknown_algorithm(self, sample_embeddings):
        """Run should skip unknown algorithms."""
        sweep = ParameterSweep()
        results = sweep.run(sample_embeddings, algorithms=["unknown_algorithm"])
        assert len(results) == 0

    def test_sweep_result_has_centroids(self, sample_embeddings):
        """Sweep results should have centroids."""
        sweep = ParameterSweep(param_grids={"kmeans": {"n_clusters": [2]}})
        results = sweep.run(sample_embeddings, algorithms=["kmeans"])
        result = results.results[0]
        assert result.centroids is not None
        assert result.centroids.shape[0] == 2

    def test_repr(self):
        """Repr should show algorithms."""
        sweep = ParameterSweep(param_grids={"kmeans": {"n_clusters": [2]}})
        repr_str = repr(sweep)
        assert "kmeans" in repr_str


class TestParallelExecution:
    """Tests for parallel execution of sweep tasks."""

    def test_sequential_vs_parallel_results_equivalence(self, sample_embeddings):
        """Sequential and parallel runs should produce equivalent results."""
        grids = {"kmeans": {"n_clusters": [2, 3, 4]}}

        sweep_seq = ParameterSweep(param_grids=grids, random_state=42, max_workers=None)
        results_seq = sweep_seq.run(sample_embeddings, algorithms=["kmeans"])

        sweep_par = ParameterSweep(param_grids=grids, random_state=42, max_workers=2)
        results_par = sweep_par.run(sample_embeddings, algorithms=["kmeans"])

        assert len(results_seq) == len(results_par)
        assert len(results_seq) == 3

    def test_parallel_with_max_workers_1_is_sequential(self, sample_embeddings):
        """max_workers=1 should behave like sequential."""
        grids = {"kmeans": {"n_clusters": [2, 3]}}

        sweep_seq = ParameterSweep(param_grids=grids, random_state=42, max_workers=None)
        results_seq = sweep_seq.run(sample_embeddings, algorithms=["kmeans"])

        sweep_par1 = ParameterSweep(param_grids=grids, random_state=42, max_workers=1)
        results_par1 = sweep_par1.run(sample_embeddings, algorithms=["kmeans"])

        assert len(results_seq) == len(results_par1)

    def test_parallel_execution_produces_results(self, sample_embeddings):
        """Parallel execution should produce results."""
        grids = {"kmeans": {"n_clusters": [2, 3, 4]}}

        sweep = ParameterSweep(param_grids=grids, random_state=42, max_workers=4)
        results = sweep.run(sample_embeddings, algorithms=["kmeans"])

        assert len(results) == 3

    def test_empty_tasks_returns_empty_results(self, sample_embeddings):
        """Empty task list should return empty results."""
        sweep = ParameterSweep(param_grids={}, random_state=42, max_workers=2)
        results = sweep.run(sample_embeddings, algorithms=[])
        assert len(results) == 0

    def test_parallel_with_mocked_evaluate(self):
        """Test parallel execution uses joblib with loky backend."""
        import joblib
        from unittest.mock import patch, MagicMock

        embeddings = np.random.rand(50, 10)

        call_count = {"parallel": 0}
        original_delayed = joblib.delayed

        def counting_delayed(func):
            def wrapper(*args, **kwargs):
                call_count["parallel"] += 1
                return original_delayed(func)(*args, **kwargs)

            return wrapper

        with patch("nordlys.search.delayed", counting_delayed):
            sweep = ParameterSweep(
                param_grids={"kmeans": {"n_clusters": [2, 3]}},
                random_state=42,
                max_workers=2,
            )
            results = sweep.run(embeddings, algorithms=["kmeans"])
            assert len(results) == 2
            assert call_count["parallel"] == 2


class TestIntegration:
    """Integration tests for search workflow."""

    def test_full_workflow(self, sample_embeddings):
        """Test complete search workflow."""

        # Define custom scorer
        def custom_scorer(result: SweepResult) -> float:
            sil = result.metrics.silhouette_score or 0.0
            # Prefer fewer clusters slightly
            return sil - 0.01 * result.n_clusters

        # Define constraint
        def reasonable_clusters(result: SweepResult) -> bool:
            return 2 <= result.n_clusters <= 4

        # Run sweep
        sweep = ParameterSweep(
            param_grids={
                "kmeans": {"n_clusters": [2, 3, 4, 5, 6]},
            },
            random_state=42,
        )
        results = sweep.run(sample_embeddings, algorithms=["kmeans"])

        # Select best
        selected = results.select(
            scorer=custom_scorer, constraints=[reasonable_clusters]
        )

        assert selected is not None
        assert 2 <= selected.n_clusters <= 4

    def test_built_in_scorers_and_constraints(self, sample_embeddings):
        """Test using built-in scorers and constraints together."""
        sweep = ParameterSweep(
            param_grids={
                "kmeans": {"n_clusters": [2, 3, 4]},
            },
            random_state=42,
        )
        results = sweep.run(sample_embeddings, algorithms=["kmeans"])

        # Use built-in scorer
        scorer = silhouette_scorer()

        # Use built-in constraints
        constraints = [
            min_clusters_constraint(min_k=2),
            max_clusters_constraint(max_k=4),
            cluster_balance_constraint(max_cv=1.0),
        ]

        selected = results.select(scorer=scorer, constraints=constraints)

        assert selected is not None
        assert selected.n_clusters >= 2
        assert selected.n_clusters <= 4
