"""Structure search and selection primitives.

This module provides tools for exploring clustering candidates and selecting
the best structure for routing tasks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np

from joblib import Parallel, delayed

from nordlys.clustering.agglomerative import AgglomerativeClusterer
from nordlys.clustering.base import Clusterer
from nordlys.clustering.bisecting import BisectingKMeansClusterer
from nordlys.clustering.gmm import GMMClusterer
from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer
from nordlys.clustering.kmeans import KMeansClusterer
from nordlys.clustering.metrics import ClusterMetrics, compute_cluster_metrics
from nordlys.clustering.minibatch import MiniBatchKMeansClusterer
from nordlys.clustering.spectral import SpectralClusterer

logger = logging.getLogger(__name__)


class CandidateSpec(Protocol):
    """Protocol for clustering candidate specifications.

    A candidate spec defines a single clustering configuration with
    typed parameters and a builder method.
    """

    @property
    def name(self) -> str:
        """Algorithm name."""
        ...

    def build(self, random_state: int) -> Clusterer:
        """Build the clusterer with the given random state."""
        ...

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        """Return parameters as a dictionary for display/export."""
        ...


@dataclass(frozen=True)
class KMeansSpec:
    """K-Means clustering specification."""

    n_clusters: int
    max_iter: int = 300
    n_init: int = 10

    @property
    def name(self) -> str:
        return "kmeans"

    def build(self, random_state: int) -> KMeansClusterer:
        return KMeansClusterer(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
        }


@dataclass(frozen=True)
class MiniBatchKMeansSpec:
    """Mini-Batch K-Means clustering specification."""

    n_clusters: int
    max_iter: int = 100
    batch_size: int = 1024
    init_size: int | None = 3 * 1024

    @property
    def name(self) -> str:
        return "minibatch_kmeans"

    def build(self, random_state: int) -> MiniBatchKMeansClusterer:
        return MiniBatchKMeansClusterer(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=random_state,
            init_size=self.init_size,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class BisectingKMeansSpec:
    """Bisecting K-Means clustering specification."""

    n_clusters: int
    max_iter: int = 100
    batch_size: int = 1024

    @property
    def name(self) -> str:
        return "bisecting_kmeans"

    def build(self, random_state: int) -> BisectingKMeansClusterer:
        return BisectingKMeansClusterer(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class HDBSCANSpec:
    """HDBSCAN clustering specification."""

    min_cluster_size: int
    min_samples: int | None = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"

    @property
    def name(self) -> str:
        return "hdbscan"

    def build(self, random_state: int) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class GMMSpec:
    """Gaussian Mixture Model clustering specification."""

    n_components: int
    covariance_type: str = "full"
    max_iter: int = 100
    n_init: int = 1

    @property
    def name(self) -> str:
        return "gmm"

    def build(self, random_state: int) -> GMMClusterer:
        return GMMClusterer(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "max_iter": self.max_iter,
        }


@dataclass(frozen=True)
class AgglomerativeSpec:
    """Agglomerative clustering specification."""

    n_clusters: int
    linkage: str = "ward"
    metric: str = "euclidean"

    @property
    def name(self) -> str:
        return "agglomerative"

    def build(self, random_state: int) -> AgglomerativeClusterer:
        return AgglomerativeClusterer(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class SpectralSpec:
    """Spectral clustering specification."""

    n_clusters: int
    affinity: str = "nearest_neighbors"
    n_neighbors: int = 10

    @property
    def name(self) -> str:
        return "spectral"

    def build(self, random_state: int) -> SpectralClusterer:
        return SpectralClusterer(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors,
            random_state=random_state,
        )

    def params_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "n_clusters": self.n_clusters,
            "affinity": self.affinity,
            "n_neighbors": self.n_neighbors,
        }


def make_kmeans_candidates(
    n_clusters: list[int] | None = None,
    max_iter: int = 300,
    n_init: int = 10,
) -> list[KMeansSpec]:
    """Generate K-Means candidate specifications."""
    if n_clusters is None:
        n_clusters = [10, 15, 20, 25, 30]
    return [
        KMeansSpec(n_clusters=k, max_iter=max_iter, n_init=n_init) for k in n_clusters
    ]


def make_minibatch_kmeans_candidates(
    n_clusters: list[int] | None = None,
    max_iter: int = 100,
    batch_size: int = 1024,
) -> list[MiniBatchKMeansSpec]:
    """Generate Mini-Batch K-Means candidate specifications."""
    if n_clusters is None:
        n_clusters = [10, 15, 20, 25, 30]
    return [
        MiniBatchKMeansSpec(n_clusters=k, max_iter=max_iter, batch_size=batch_size)
        for k in n_clusters
    ]


def make_bisecting_kmeans_candidates(
    n_clusters: list[int] | None = None,
    max_iter: int = 100,
    batch_size: int = 1024,
) -> list[BisectingKMeansSpec]:
    """Generate Bisecting K-Means candidate specifications."""
    if n_clusters is None:
        n_clusters = [10, 15, 20, 25, 30]
    return [
        BisectingKMeansSpec(n_clusters=k, max_iter=max_iter, batch_size=batch_size)
        for k in n_clusters
    ]


def make_hdbscan_candidates(
    min_cluster_size: list[int] | None = None,
    min_samples: list[int] | None = None,
) -> list[HDBSCANSpec]:
    """Generate HDBSCAN candidate specifications."""
    if min_cluster_size is None:
        min_cluster_size = [50, 100, 150, 200]
    if min_samples is None:
        min_samples = [5, 10, 15]
    return [
        HDBSCANSpec(min_cluster_size=mcs, min_samples=ms)
        for mcs in min_cluster_size
        for ms in min_samples
    ]


def make_gmm_candidates(
    n_components: list[int] | None = None,
    covariance_type: list[str] | None = None,
    max_iter: int = 100,
    n_init: int = 1,
) -> list[GMMSpec]:
    """Generate GMM candidate specifications."""
    if n_components is None:
        n_components = [10, 15, 20, 25, 30]
    if covariance_type is None:
        covariance_type = ["full", "diag"]
    return [
        GMMSpec(n_components=nc, covariance_type=ct, max_iter=max_iter, n_init=n_init)
        for nc in n_components
        for ct in covariance_type
    ]


def make_agglomerative_candidates(
    n_clusters: list[int] | None = None,
    linkage: list[str] | None = None,
) -> list[AgglomerativeSpec]:
    """Generate Agglomerative clustering candidate specifications."""
    if n_clusters is None:
        n_clusters = [10, 15, 20, 25, 30]
    if linkage is None:
        linkage = ["ward", "average"]
    return [
        AgglomerativeSpec(n_clusters=k, linkage=link)
        for k in n_clusters
        for link in linkage
    ]


def make_spectral_candidates(
    n_clusters: list[int] | None = None,
    affinity: list[str] | None = None,
    n_neighbors: int = 10,
) -> list[SpectralSpec]:
    """Generate Spectral clustering candidate specifications."""
    if n_clusters is None:
        n_clusters = [10, 15, 20, 25, 30]
    if affinity is None:
        affinity = ["nearest_neighbors"]
    return [
        SpectralSpec(n_clusters=k, affinity=a, n_neighbors=n_neighbors)
        for k in n_clusters
        for a in affinity
    ]


# Type alias for parameter dictionaries
Params = dict[str, int | float | str | bool | None]


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
    params: Params
    metrics: ClusterMetrics
    labels: np.ndarray
    centroids: np.ndarray
    clusterer: Clusterer

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

        def get_silhouette(r: SweepResult) -> float:
            return r.metrics.silhouette_score or 0.0

        return max(valid, key=get_silhouette)

    def best_by_n_clusters(self, target: int) -> SweepResult | None:
        """Get the result closest to target number of clusters with best silhouette."""
        if not self.results:
            return None
        exact_matches = [r for r in self.results if r.metrics.n_clusters == target]
        if exact_matches:
            valid = [r for r in exact_matches if r.metrics.silhouette_score is not None]
            if valid:

                def get_silhouette(r: SweepResult) -> float:
                    return r.metrics.silhouette_score or 0.0

                return max(valid, key=get_silhouette)
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

    def __init__(
        self,
        candidates: Sequence[CandidateSpec] | None = None,
        random_state: int = 42,
        max_workers: int | None = None,
    ) -> None:
        """Initialize ParameterSweep.

        Args:
            candidates: Sequence of candidate specifications to evaluate.
                       If None, uses default K-Means candidates.
            random_state: Random seed for reproducibility (default: 42)
            max_workers: Maximum number of parallel workers. If None, runs sequentially.
                         If 1, runs sequentially. If > 1, runs that many in parallel.
        """
        self.candidates = (
            list(candidates) if candidates is not None else make_kmeans_candidates()
        )
        self.random_state = random_state
        self.max_workers = max_workers

    def run(
        self,
        embeddings: np.ndarray,
        verbose: bool = False,
    ) -> SweepResults:
        """Run parameter sweep over candidates.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            verbose: Print progress (default: False)

        Returns:
            SweepResults containing all evaluated configurations
        """
        if not self.candidates:
            return SweepResults()

        return self._run_tasks(embeddings, self.candidates, verbose)

    def _run_tasks(
        self,
        embeddings: np.ndarray,
        candidates: Sequence[CandidateSpec],
        verbose: bool,
    ) -> SweepResults:
        """Execute sweep tasks sequentially or in parallel."""
        if not candidates:
            return SweepResults()

        if self.max_workers is None or self.max_workers == 1:
            return self._run_sequential(embeddings, candidates, verbose)
        return self._run_parallel(embeddings, candidates, verbose)

    def _run_sequential(
        self,
        embeddings: np.ndarray,
        candidates: Sequence[CandidateSpec],
        verbose: bool,
    ) -> SweepResults:
        """Run tasks sequentially."""
        results = SweepResults()

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        iterator = candidates
        if tqdm is not None and verbose:
            iterator = tqdm(
                candidates,
                desc="Clustering sweep",
                unit="candidate",
                disable=False,
            )

        for spec in iterator:
            if verbose and tqdm is None:
                print(f"Running {spec.name} with {spec.params_dict()}")
            try:
                result = self._evaluate_candidate(embeddings, spec)
                results.results.append(result)
            except Exception as e:
                logger.warning(
                    "Failed %s with %s: %s", spec.name, spec.params_dict(), e
                )
                if verbose and tqdm is None:
                    print(f"  Failed: {e}")
        return results

    def _run_parallel(
        self,
        embeddings: np.ndarray,
        candidates: Sequence[CandidateSpec],
        verbose: bool,
    ) -> SweepResults:
        """Run tasks in parallel using joblib with loky backend.

        Uses process-based parallelism via joblib's loky backend to avoid the
        OpenBLAS/OpenMP threading issues that occur with ThreadPoolExecutor
        when running many BLAS-heavy clustering operations concurrently.
        """
        results = SweepResults()

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        def evaluate_task(spec: CandidateSpec) -> SweepResult | None:
            try:
                return _evaluate_spec_worker(embeddings, spec, self.random_state)
            except Exception as e:
                logger.warning(
                    "Failed %s with %s: %s", spec.name, spec.params_dict(), e
                )
                return None

        with TemporaryDirectory(prefix="nordlys-sweep-") as temp_dir:
            if tqdm is not None and verbose:
                parallel_iter = Parallel(
                    n_jobs=self.max_workers,
                    backend="loky",
                    temp_folder=temp_dir,
                    mmap_mode="r",
                )(
                    delayed(evaluate_task)(spec)
                    for spec in tqdm(
                        candidates, desc="Clustering sweep", unit="candidate"
                    )
                )

                for result in parallel_iter:
                    if result is not None:
                        results.results.append(result)
            else:
                if verbose:
                    for spec in candidates:
                        print(f"Running {spec.name} with {spec.params_dict()}")

                parallel_results = Parallel(
                    n_jobs=self.max_workers,
                    backend="loky",
                    temp_folder=temp_dir,
                    mmap_mode="r",
                )(delayed(evaluate_task)(spec) for spec in candidates)

                for result in parallel_results:
                    if result is not None:
                        results.results.append(result)

        return results

    def _evaluate_candidate(
        self,
        embeddings: np.ndarray,
        spec: CandidateSpec,
    ) -> SweepResult:
        """Evaluate a single clustering candidate."""
        clusterer = spec.build(self.random_state)
        clusterer.fit(embeddings)

        labels = clusterer.labels_
        centroids = clusterer.cluster_centers_
        inertia = clusterer.inertia_

        metrics = compute_cluster_metrics(embeddings, labels, inertia)

        return SweepResult(
            algorithm=spec.name,
            params=spec.params_dict(),
            metrics=metrics,
            labels=labels,
            centroids=np.asarray(centroids, dtype=np.float32),
            clusterer=clusterer,
        )

    def __repr__(self) -> str:
        names = list(set(c.name for c in self.candidates))
        return f"ParameterSweep(candidates={names})"


def _evaluate_spec_worker(
    embeddings: np.ndarray,
    spec: CandidateSpec,
    random_state: int,
) -> SweepResult:
    """Worker function for parallel evaluation of clustering candidates.

    This is a module-level function to ensure it can be pickled for joblib.
    """
    import numpy as np

    from nordlys.clustering.metrics import compute_cluster_metrics

    clusterer = spec.build(random_state)
    clusterer.fit(embeddings)

    labels = clusterer.labels_
    centroids = clusterer.cluster_centers_
    inertia = clusterer.inertia_

    metrics = compute_cluster_metrics(embeddings, labels, inertia)

    return SweepResult(
        algorithm=spec.name,
        params=spec.params_dict(),
        metrics=metrics,
        labels=labels,
        centroids=np.asarray(centroids, dtype=np.float32),
        clusterer=clusterer,
    )


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
