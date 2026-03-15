"""Structure search and selection primitives.

This module provides tools for exploring clustering candidates and selecting
the best structure for routing tasks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from nordlys.clustering.agglomerative import AgglomerativeClusterer
from nordlys.clustering.bisecting import BisectingKMeansClusterer
from nordlys.clustering.gmm import GMMClusterer
from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer
from nordlys.clustering.kmeans import KMeansClusterer
from nordlys.clustering.metrics import ClusterMetrics, compute_cluster_metrics
from nordlys.clustering.minibatch import MiniBatchKMeansClusterer
from nordlys.clustering.spectral import SpectralClusterer


class SweepScorer(Protocol):
    """Protocol for scoring sweep results.

    A scorer takes a sweep result and returns a numeric score.
    Higher scores are considered better.
    """

    def __call__(self, result: SweepResult) -> float: ...


class SweepConstraint(Protocol):
    """Protocol for filtering sweep results.

    A constraint returns True if the result is acceptable.
    """

    def __call__(self, result: SweepResult) -> bool: ...


@dataclass
class SweepResult:
    """Result of a single clustering configuration.

    Attributes:
        algorithm: Name of the clustering algorithm
        params: Parameters used for clustering
        metrics: Clustering metrics
        labels: Cluster assignments
        centroids: Cluster centroid vectors of shape (n_clusters, n_features)
        clusterer: The fitted clusterer instance
    """

    algorithm: str
    params: dict[str, Any]
    metrics: ClusterMetrics
    labels: np.ndarray
    centroids: np.ndarray
    clusterer: Any  # The fitted clusterer

    @property
    def n_clusters(self) -> int:
        """Number of clusters in this result."""
        return self.metrics.n_clusters


@dataclass
class SweepResults:
    """Results from a parameter sweep.

    Attributes:
        results: List of individual sweep results
    """

    results: list[SweepResult] = field(default_factory=list)

    def filter(self, constraint: SweepConstraint) -> SweepResults:
        """Filter results by a constraint.

        Args:
            constraint: A callable that returns True for acceptable results.

        Returns:
            New SweepResults containing only results that pass the constraint.
        """
        return SweepResults(results=[r for r in self.results if constraint(r)])

    def rank(self, scorer: SweepScorer) -> list[tuple[SweepResult, float]]:
        """Rank results by a scorer.

        Args:
            scorer: A callable that returns a numeric score for each result.

        Returns:
            List of (result, score) tuples sorted by score in descending order.
        """
        scored = [(r, scorer(r)) for r in self.results]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select(
        self,
        scorer: SweepScorer,
        constraints: list[SweepConstraint] | None = None,
    ) -> SweepResult | None:
        """Select the best result by scorer, optionally filtered by constraints.

        Args:
            scorer: A callable that returns a numeric score for each result.
            constraints: Optional list of constraints. Results must pass all.

        Returns:
            The highest-scoring result that passes all constraints, or None if no
            results pass.
        """
        candidates = self.results

        # Apply constraints
        if constraints:
            for constraint in constraints:
                candidates = [r for r in candidates if constraint(r)]

        if not candidates:
            return None

        # Score and sort
        scored = [(r, scorer(r)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def best_by_silhouette(self) -> SweepResult | None:
        """Get the result with the highest silhouette score."""
        if not self.results:
            return None
        valid = [r for r in self.results if r.metrics.silhouette_score is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r.metrics.silhouette_score)

    def best_by_n_clusters(self, target: int) -> SweepResult | None:
        """Get the result closest to target number of clusters with best silhouette."""
        if not self.results:
            return None
        # Filter to results with exact n_clusters
        exact_matches = [r for r in self.results if r.metrics.n_clusters == target]
        if exact_matches:
            valid = [r for r in exact_matches if r.metrics.silhouette_score is not None]
            if valid:
                return max(valid, key=lambda r: r.metrics.silhouette_score)
        # Fall back to closest
        return min(self.results, key=lambda r: abs(r.metrics.n_clusters - target))

    def filter_by_algorithm(self, algorithm: str) -> SweepResults:
        """Filter results by algorithm name."""
        return SweepResults(
            results=[r for r in self.results if r.algorithm == algorithm]
        )

    def to_dataframe(self):
        """Convert results to a pandas DataFrame."""
        import pandas as pd

        records = []
        for r in self.results:
            record = {
                "algorithm": r.algorithm,
                "silhouette_score": r.metrics.silhouette_score,
                "n_clusters": r.metrics.n_clusters,
                "n_samples": r.metrics.n_samples,
                "min_cluster_size": r.metrics.min_cluster_size,
                "max_cluster_size": r.metrics.max_cluster_size,
                "avg_cluster_size": r.metrics.avg_cluster_size,
                **r.params,
            }
            records.append(record)
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)


class ParameterSweep:
    """Grid search over clustering algorithms and parameters.

    Example:
        >>> sweep = ParameterSweep()
        >>> results = sweep.run(embeddings, algorithms=["kmeans", "hdbscan"])
        >>> best = results.best_by_silhouette()
        >>> print(f"Best: {best.algorithm} with silhouette {best.metrics.silhouette_score:.3f}")

        >>> # Using scorer and constraints
        >>> def my_scorer(result):
        ...     return result.metrics.silhouette_score or 0
        >>> def min_clusters(result):
        ...     return result.metrics.n_clusters >= 5
        >>> best = results.select(scorer=my_scorer, constraints=[min_clusters])
    """

    # Default parameter grids for each algorithm
    DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
        "kmeans": {
            "n_clusters": [10, 15, 20, 25, 30],
        },
        "hdbscan": {
            "min_cluster_size": [50, 100, 150, 200],
            "min_samples": [5, 10, 15],
        },
        "gmm": {
            "n_components": [10, 15, 20, 25, 30],
            "covariance_type": ["full", "diag"],
        },
        "agglomerative": {
            "n_clusters": [10, 15, 20, 25, 30],
            "linkage": ["ward", "average"],
        },
        "spectral": {
            "n_clusters": [10, 15, 20, 25, 30],
            "affinity": ["nearest_neighbors"],
        },
    }

    CLUSTERER_MAP = {
        "kmeans": KMeansClusterer,
        "minibatch_kmeans": MiniBatchKMeansClusterer,
        "bisecting_kmeans": BisectingKMeansClusterer,
        "hdbscan": HDBSCANClusterer,
        "gmm": GMMClusterer,
        "agglomerative": AgglomerativeClusterer,
        "spectral": SpectralClusterer,
    }

    def __init__(
        self,
        param_grids: dict[str, dict[str, list[Any]]] | None = None,
        random_state: int = 42,
        max_workers: int | None = None,
    ) -> None:
        """Initialize ParameterSweep.

        Args:
            param_grids: Custom parameter grids. If None, uses defaults.
            random_state: Random seed for reproducibility (default: 42)
            max_workers: Maximum number of parallel workers. If None, runs sequentially.
                         If 1, runs sequentially. If > 1, runs that many in parallel.
        """
        self.param_grids = param_grids or self.DEFAULT_GRIDS
        self.random_state = random_state
        self.max_workers = max_workers

    def run(
        self,
        embeddings: np.ndarray,
        algorithms: list[str] | None = None,
        verbose: bool = False,
    ) -> SweepResults:
        """Run parameter sweep over algorithms.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            algorithms: List of algorithm names to try. Default: ["kmeans", "hdbscan", "gmm"]
            verbose: Print progress (default: False)

        Returns:
            SweepResults containing all evaluated configurations
        """
        if algorithms is None:
            algorithms = ["kmeans", "hdbscan", "gmm"]

        # Build list of all (algorithm, params) combinations to evaluate
        tasks: list[tuple[str, dict[str, Any]]] = []
        for algo_name in algorithms:
            if algo_name not in self.param_grids:
                if verbose:
                    print(f"Skipping {algo_name}: no parameter grid defined")
                continue
            if algo_name not in self.CLUSTERER_MAP:
                if verbose:
                    print(f"Skipping {algo_name}: unknown algorithm")
                continue
            grid = self.param_grids[algo_name]
            param_combinations = self._generate_combinations(grid)
            for params in param_combinations:
                tasks.append((algo_name, params))

        if not tasks:
            return SweepResults()

        # Run sequentially or in parallel based on max_workers
        results = SweepResults()
        if self.max_workers is None or self.max_workers == 1:
            # Sequential execution
            for algo_name, params in tasks:
                if verbose:
                    print(f"Running {algo_name} with {params}")
                try:
                    result = self._evaluate_config(embeddings, algo_name, params)
                    results.results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Failed: {e}")
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_config, embeddings, algo_name, params
                    ): (algo_name, params)
                    for algo_name, params in tasks
                }
                for future in as_completed(futures):
                    algo_name, params = futures[future]
                    if verbose:
                        print(f"Running {algo_name} with {params}")
                    try:
                        result = future.result()
                        results.results.append(result)
                    except Exception as e:
                        if verbose:
                            print(f"  Failed: {e}")

        return results

    def _generate_combinations(
        self, grid: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all combinations of parameters."""
        from itertools import product

        keys = list(grid.keys())
        values = list(grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _evaluate_config(
        self,
        embeddings: np.ndarray,
        algo_name: str,
        params: dict[str, Any],
    ) -> SweepResult:
        """Evaluate a single clustering configuration."""
        clusterer_class = self.CLUSTERER_MAP[algo_name]

        # Add random_state if supported
        if algo_name in [
            "kmeans",
            "minibatch_kmeans",
            "bisecting_kmeans",
            "gmm",
            "spectral",
        ]:
            params = {**params, "random_state": self.random_state}

        clusterer = clusterer_class(**params)
        clusterer.fit(embeddings)

        labels = clusterer.labels_
        centroids = clusterer.cluster_centers_
        inertia = clusterer.inertia_ if isinstance(clusterer, KMeansClusterer) else None

        metrics = compute_cluster_metrics(embeddings, labels, inertia)

        return SweepResult(
            algorithm=algo_name,
            params=params,
            metrics=metrics,
            labels=labels,
            centroids=np.asarray(centroids, dtype=np.float32),
            clusterer=clusterer,
        )

    def __repr__(self) -> str:
        algos = list(self.param_grids.keys())
        return f"ParameterSweep(algorithms={algos})"


# Built-in scorer and constraint factories
def silhouette_scorer() -> Callable[[SweepResult], float]:
    """Create a scorer that ranks by silhouette score."""

    def scorer(result: SweepResult) -> float:
        return result.metrics.silhouette_score or 0.0

    return scorer


def cluster_count_scorer(
    target: int, penalty: float = 0.1
) -> Callable[[SweepResult], float]:
    """Create a scorer that prefers results close to a target cluster count.

    Args:
        target: Target number of clusters
        penalty: Penalty per cluster of difference (default: 0.1)
    """

    def scorer(result: SweepResult) -> float:
        diff = abs(result.metrics.n_clusters - target)
        base = result.metrics.silhouette_score or 0.0
        return base - penalty * diff

    return scorer


def min_clusters_constraint(min_k: int) -> Callable[[SweepResult], bool]:
    """Create a constraint that requires at least min_k clusters."""

    def constraint(result: SweepResult) -> bool:
        return result.metrics.n_clusters >= min_k

    return constraint


def max_clusters_constraint(max_k: int) -> Callable[[SweepResult], bool]:
    """Create a constraint that requires at most max_k clusters."""

    def constraint(result: SweepResult) -> bool:
        return result.metrics.n_clusters <= max_k

    return constraint


def cluster_balance_constraint(max_cv: float = 0.6) -> Callable[[SweepResult], bool]:
    """Create a constraint that requires cluster balance below max_cv.

    Balance is measured as coefficient of variation (std/mean) of cluster sizes.
    """

    def constraint(result: SweepResult) -> bool:
        if not result.metrics.cluster_sizes:
            return True
        sizes = np.array(result.metrics.cluster_sizes, dtype=np.float32)
        if sizes.sum() == 0:
            return True
        cv = sizes.std() / sizes.mean()
        return cv < max_cv

    return constraint
