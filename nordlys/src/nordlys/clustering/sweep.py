"""Parameter sweep for clustering algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nordlys.clustering.agglomerative import AgglomerativeClusterer
from nordlys.clustering.gmm import GMMClusterer
from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer
from nordlys.clustering.kmeans import KMeansClusterer
from nordlys.clustering.metrics import ClusterMetrics, compute_cluster_metrics
from nordlys.clustering.spectral import SpectralClusterer


@dataclass
class SweepResult:
    """Result of a single clustering configuration.

    Attributes:
        algorithm: Name of the clustering algorithm
        params: Parameters used for clustering
        metrics: Clustering metrics
        labels: Cluster assignments
        clusterer: The fitted clusterer instance
    """

    algorithm: str
    params: dict[str, Any]
    metrics: ClusterMetrics
    labels: np.ndarray
    clusterer: Any  # The fitted clusterer


@dataclass
class SweepResults:
    """Results from a parameter sweep.

    Attributes:
        results: List of individual sweep results
    """

    results: list[SweepResult] = field(default_factory=list)

    def best_by_silhouette(self) -> SweepResult | None:
        """Get the result with the highest silhouette score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.metrics.silhouette_score)

    def best_by_n_clusters(self, target: int) -> SweepResult | None:
        """Get the result closest to target number of clusters with best silhouette."""
        if not self.results:
            return None
        # Filter to results with exact n_clusters
        exact_matches = [r for r in self.results if r.metrics.n_clusters == target]
        if exact_matches:
            return max(exact_matches, key=lambda r: r.metrics.silhouette_score)
        # Fall back to closest
        return min(self.results, key=lambda r: abs(r.metrics.n_clusters - target))

    def filter_by_algorithm(self, algorithm: str) -> "SweepResults":
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
        "hdbscan": HDBSCANClusterer,
        "gmm": GMMClusterer,
        "agglomerative": AgglomerativeClusterer,
        "spectral": SpectralClusterer,
    }

    def __init__(
        self,
        param_grids: dict[str, dict[str, list[Any]]] | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize ParameterSweep.

        Args:
            param_grids: Custom parameter grids. If None, uses defaults.
            random_state: Random seed for reproducibility (default: 42)
        """
        self.param_grids = param_grids or self.DEFAULT_GRIDS
        self.random_state = random_state

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

        results = SweepResults()

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
                if verbose:
                    print(f"Running {algo_name} with {params}")

                try:
                    result = self._evaluate_config(embeddings, algo_name, params)
                    results.results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Failed: {e}")
                    continue

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
        if algo_name in ["kmeans", "gmm", "spectral"]:
            params = {**params, "random_state": self.random_state}

        clusterer = clusterer_class(**params)
        clusterer.fit(embeddings)

        labels = clusterer.labels_
        inertia = getattr(clusterer, "inertia_", None)

        metrics = compute_cluster_metrics(embeddings, labels, inertia)

        return SweepResult(
            algorithm=algo_name,
            params=params,
            metrics=metrics,
            labels=labels,
            clusterer=clusterer,
        )

    def __repr__(self) -> str:
        algos = list(self.param_grids.keys())
        return f"ParameterSweep(algorithms={algos})"
