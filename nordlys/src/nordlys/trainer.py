"""Dataset-native trainer that compiles router checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from nordlys.checkpoint import build_checkpoint
from nordlys.checkpoint_types import (
    CheckpointMetrics,
    CheckpointModelEntry,
    ClusteringConfig,
    EmbeddingConfig,
)
from nordlys.clustering import Clusterer, KMeansClusterer, compute_cluster_metrics
from nordlys.dataset import Dataset
from nordlys.embeddings import Embedder, SentenceTransformers
from nordlys.reduction import Reducer, restore_reducer
from nordlys.reduction.base import ReductionPayload
from nordlys.router import Router
from nordlys.search import ParameterSweep, SweepResult, SweepScorer, SweepConstraint


@dataclass(frozen=True)
class FittedStructure:
    """Compiled prototype routing structure learned from data.

    Attributes:
        cluster_centers: Routing prototypes of shape (n_clusters, n_features)
        n_clusters: Number of compiled routing prototypes
        embedding_config: Embedding model configuration
        clustering_config: Clustering algorithm configuration
        reduction: Optional dimensionality reduction payload
        metrics: Metrics computed against the compiled prototype routing labels
    """

    cluster_centers: np.ndarray
    n_clusters: int
    embedding_config: EmbeddingConfig
    clustering_config: ClusteringConfig
    reduction: ReductionPayload | None
    metrics: CheckpointMetrics

    @property
    def centroids(self) -> np.ndarray:
        """Alias for cluster_centers for backwards compatibility."""
        return self.cluster_centers

    @property
    def routing_prototypes(self) -> np.ndarray:
        """Prototype vectors used by runtime nearest-prototype routing."""
        return self.cluster_centers


@dataclass(frozen=True)
class RoutingPolicy:
    """Result of calibrating model quality per cluster.

    Attributes:
        scores: Dict mapping model_id to list of per-cluster scores (0-1, higher is better)
        sample_counts: Number of calibration samples per cluster
    """

    scores: dict[str, list[float]]
    sample_counts: list[int]

    def model_scores(self, model_id: str) -> list[float]:
        """Get scores for a specific model."""
        return self.scores.get(model_id, [])

    def best_model_per_cluster(self) -> list[str]:
        """Get the best model (highest score) for each cluster."""
        best = []
        for cluster_idx in range(len(self.sample_counts)):
            best_model = None
            best_score = -1.0
            for model_id, model_scores in self.scores.items():
                if cluster_idx < len(model_scores):
                    if model_scores[cluster_idx] > best_score:
                        best_score = model_scores[cluster_idx]
                        best_model = model_id
            best.append(best_model if best_model is not None else "")
        return best


@dataclass(frozen=True)
class EvaluationReport:
    """Result of evaluating a router on held-out data.

    Attributes:
        routing_decisions: List of route results for each evaluation sample
        metrics: Evaluation metrics such as accuracy
    """

    routing_decisions: list
    metrics: dict


@dataclass(frozen=True)
class Trainer:
    """Train router checkpoints from Dataset objects.

    Canonical path: ``Trainer -> fit_clusters() / score_clusters() / build() -> Router``.

    Example:
        >>> from nordlys import Dataset, Trainer
        >>> from nordlys.clustering import KMeansClusterer
        >>> train_ds = Dataset.from_list([
        ...     {"id": "1", "input": "Fix the parser"},
        ... ])
        >>> val_ds = Dataset.from_list([
        ...     {"id": "2", "input": "Write tests", "targets": {"gpt-4": 1}},
        ... ])
        >>> # Pass your own clusterer
        >>> trainer = Trainer(
        ...     models=["gpt-4"],
        ...     clusterer=KMeansClusterer(n_clusters=10),
        ... )
        >>> fitted = trainer.fit_clusters(train_ds)
        >>> scored = trainer.score_clusters(fitted, val_ds)
        >>> router = trainer.build(fitted, scored)
        >>> result = router.route("Explain quantum computing")
    """

    models: list[str]
    input_col: str = "input"
    target_col: str = "targets"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_normalize: bool = True
    embedder: Embedder | None = None

    clusterer: Clusterer | None = None
    n_clusters: int = 20

    reducer: Reducer | None = None

    random_state: int = 42
    allow_trust_remote_code: bool = False
    device: Literal["cpu", "cuda"] = "cpu"

    def fit_structure(self, dataset: Dataset) -> FittedStructure:
        """Fit clustering and compile it into prototype routing.

        The requested clusterer is used to learn structure on the training data,
        then the result is compiled into the nearest-prototype representation used
        by runtime routing. This keeps all algorithms on the same routing path.

        Args:
            dataset: Training dataset with 'id' and 'input' columns

        Returns:
            FittedStructure containing the compiled prototype routing state
        """
        self._validate_fit_clusters(dataset)

        embedder = self._make_embedder()
        embeddings = embedder.encode(dataset.column(self.input_col))
        cluster_input, reduction_payload = self._reduce_or_pass(embeddings)

        clusterer = self._make_clusterer()
        clusterer.fit(cluster_input)

        raw_centroids = clusterer.cluster_centers_
        raw_labels = clusterer.labels_

        inertia: float | None = None
        if isinstance(clusterer, KMeansClusterer):
            inertia = clusterer.inertia_

        centroids, labels, compact_inertia = self._compile_prototype_routing_structure(
            cluster_input, raw_centroids, raw_labels, inertia
        )
        n_clusters = len(centroids)

        metrics = compute_cluster_metrics(
            cluster_input,
            labels,
            compact_inertia,
        )

        return FittedStructure(
            cluster_centers=centroids,
            n_clusters=n_clusters,
            embedding_config=embedder.checkpoint_config(),
            clustering_config=ClusteringConfig(
                n_clusters=n_clusters,
                random_state=self.random_state,
                max_iter=300,
                n_init=10,
                algorithm="lloyd",
                normalization="l2" if self.embedding_normalize else "none",
            ),
            reduction=reduction_payload,
            metrics=CheckpointMetrics(
                n_samples=metrics.n_samples,
                cluster_sizes=metrics.cluster_sizes,
                silhouette_score=metrics.silhouette_score,
                inertia=metrics.inertia,
            ),
        )

    def select_structure(
        self,
        dataset: Dataset,
        sweep: ParameterSweep,
        scorer: SweepScorer,
        constraints: list[SweepConstraint] | None = None,
    ) -> FittedStructure:
        """Select the best structure and compile it into prototype routing.

        This is the advanced path for structure selection. It embeds the dataset once,
        evaluates all clustering candidates from the sweep, and selects the best one
        according to the provided scorer and constraints.

        Args:
            dataset: Training dataset with 'id' and 'input' columns
            sweep: ParameterSweep configured with candidates to evaluate
            scorer: A callable that scores each SweepResult (higher is better)
            constraints: Optional list of constraints that results must satisfy

        Returns:
            FittedStructure containing the selected clustering

        Example:
            >>> from nordlys.search import ParameterSweep, silhouette_scorer
            >>> sweep = ParameterSweep(param_grids={
            ...     "kmeans": {"n_clusters": [8, 12, 16]},
            ...     "minibatch_kmeans": {"n_clusters": [8, 12, 16]},
            ... })
            >>> fitted = trainer.select_structure(
            ...     dataset=train_ds,
            ...     sweep=sweep,
            ...     scorer=silhouette_scorer(),
            ... )
        """
        self._validate_fit_clusters(dataset)

        embedder = self._make_embedder()
        embeddings = embedder.encode(dataset.column(self.input_col))
        cluster_input, reduction_payload = self._reduce_or_pass(embeddings)

        results = sweep.run(cluster_input)

        selected = results.select(scorer=scorer, constraints=constraints)
        if selected is None:
            raise ValueError(
                "No clustering candidate satisfied the provided constraints. "
                "Try relaxing constraints or using a different sweep configuration."
            )

        raw_centroids = selected.centroids
        raw_labels = selected.labels
        inertia = selected.metrics.inertia

        centroids, labels, compact_inertia = self._compile_prototype_routing_structure(
            cluster_input, raw_centroids, raw_labels, inertia
        )

        return self._fitted_structure_from_sweep_result(
            selected,
            embedder.checkpoint_config(),
            reduction_payload,
            centroids,
            labels,
            compact_inertia,
        )

    def _fitted_structure_from_sweep_result(
        self,
        result: SweepResult,
        embedding_config: EmbeddingConfig,
        reduction_payload: ReductionPayload | None,
        centroids: np.ndarray,
        labels: np.ndarray,
        compact_inertia: float | None,
    ) -> FittedStructure:
        """Build a FittedStructure from a selected SweepResult."""
        seed_raw = result.params.get("random_state", self.random_state)
        seed = int(seed_raw) if seed_raw is not None else self.random_state
        n_clusters = len(centroids)
        return FittedStructure(
            cluster_centers=centroids,
            n_clusters=n_clusters,
            embedding_config=embedding_config,
            clustering_config=ClusteringConfig(
                n_clusters=n_clusters,
                random_state=seed,
                max_iter=300,
                n_init=10,
                algorithm=result.algorithm,
                normalization="l2" if self.embedding_normalize else "none",
            ),
            reduction=reduction_payload,
            metrics=CheckpointMetrics(
                n_samples=result.metrics.n_samples,
                cluster_sizes=result.metrics.cluster_sizes,
                silhouette_score=result.metrics.silhouette_score,
                inertia=compact_inertia,
            ),
        )

    def assign_clusters(self, fitted: FittedStructure, dataset: Dataset) -> np.ndarray:
        """Assign dataset instances to compiled routing prototypes.

        Uses nearest-prototype assignment, which is the runtime routing rule for
        all compiled structures regardless of the original clustering algorithm.

        Args:
            fitted: Fitted clustering from fit_clusters()
            dataset: Dataset with 'id' and 'input' columns

        Returns:
            Array of cluster assignments of shape (n_samples,)
        """
        embedder = self._make_embedder()
        embeddings = embedder.encode(dataset.column(self.input_col))

        # Apply the same reduction as used during fitting
        reducer = restore_reducer(fitted.reduction)
        if reducer is not None:
            embeddings = reducer.transform(embeddings)

        # Compute distances to centroids
        # embeddings: (n_samples, n_features), centroids: (n_clusters, n_features)
        # result: (n_samples, n_clusters)
        centroids = fitted.cluster_centers
        distances = np.sqrt(
            np.sum(
                (embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                axis=2,
            )
        )
        labels = np.argmin(distances, axis=1)

        return labels

    def calibrate(self, fitted: FittedStructure, dataset: Dataset) -> RoutingPolicy:
        """Calibrate routing policy using fitted structure and validation data.

        This computes per-prototype model scores from calibration data using the
        same nearest-prototype assignment rule used at runtime.

        Args:
            fitted: Fitted structure from fit_structure()
            dataset: Calibration dataset with 'id', 'input', and 'targets' columns

        Returns:
            RoutingPolicy object with per-cluster model scores
        """
        self._validate_score_clusters(dataset)

        # Assign calibration data to clusters
        labels = self.assign_clusters(fitted, dataset)

        # Compute scores from assigned labels
        targets = dataset.column(self.target_col)
        n_clusters = fitted.n_clusters

        scores = self._calc_scores(
            labels,
            targets,
            self.models,
            n_clusters,
        )

        # Compute sample counts per cluster
        sample_counts = [0] * n_clusters
        for label in labels:
            if 0 <= label < n_clusters:
                sample_counts[label] += 1

        return RoutingPolicy(
            scores=scores,
            sample_counts=sample_counts,
        )

    def compile(self, fitted: FittedStructure, policy: RoutingPolicy) -> Router:
        """Compile a prototype router from structure and policy.

        Args:
            fitted: Fitted structure from fit_structure()
            policy: Routing policy from calibrate()

        Returns:
            Router ready for nearest-prototype routing
        """
        models_payload = [
            CheckpointModelEntry(
                model_id=model_id,
                scores=policy.scores[model_id],
            )
            for model_id in self.models
        ]

        checkpoint = build_checkpoint(
            cluster_centers=fitted.cluster_centers,
            models=models_payload,
            embedding=fitted.embedding_config,
            clustering=fitted.clustering_config,
            reduction=fitted.reduction,
            metrics=fitted.metrics,
        )

        return Router(checkpoint=checkpoint, device=self.device)

    def fit_router(
        self, structure_dataset: Dataset, calibration_dataset: Dataset
    ) -> Router:
        """Convenience method: fit structure, calibrate, compile.

        Args:
            structure_dataset: Dataset for learning task structure
            calibration_dataset: Dataset for calibrating model quality per cluster

        Returns:
            Router ready for routing
        """
        fitted = self.fit_structure(structure_dataset)
        policy = self.calibrate(fitted, calibration_dataset)
        return self.compile(fitted, policy)

    def _validate_fit_clusters(self, dataset: Dataset) -> None:
        """Validate dataset for fit_clusters step."""
        if not self.models:
            raise ValueError("At least one model is required")

        errors = dataset.validate_schema(
            required_columns=["id", self.input_col],
            check_unique_ids=True,
        )
        if errors:
            raise ValueError(f"Dataset validation failed: {errors}")

    def _validate_score_clusters(self, dataset: Dataset) -> None:
        """Validate dataset for score_clusters step."""
        allowed_model_ids = set(self.models)

        errors = dataset.validate_schema(
            required_columns=["id", self.input_col, self.target_col],
            check_unique_ids=True,
        )
        if errors:
            raise ValueError(f"Dataset validation failed: {errors}")

        targets = dataset.column(self.target_col)
        target_errors: list[str] = []
        for idx, value in enumerate(targets):
            if value is None:
                continue
            if not isinstance(value, dict):
                target_errors.append(f"Row {idx}: targets must be a dict")
                continue
            for model_id, score in value.items():
                if model_id not in allowed_model_ids:
                    target_errors.append(
                        f"Row {idx}: unknown model_id '{model_id}'. "
                        f"Allowed model IDs: {sorted(allowed_model_ids)}"
                    )
                if not isinstance(model_id, str):
                    target_errors.append(f"Row {idx}: model_id must be str")
                if not isinstance(score, (int, float)):
                    target_errors.append(
                        f"Row {idx}: target for {model_id} must be numeric"
                    )

        if target_errors:
            raise ValueError(f"Dataset validation failed: {target_errors}")

    def _validate(self, dataset: Dataset) -> None:
        if not self.models:
            raise ValueError("At least one model is required")

        allowed_model_ids = set(self.models)

        errors = dataset.validate_schema(
            required_columns=["id", self.input_col, self.target_col],
            check_unique_ids=True,
        )
        if errors:
            raise ValueError(f"Dataset validation failed: {errors}")

        targets = dataset.column(self.target_col)
        errors.extend(
            [
                f"Row {idx}: targets must be a dict"
                for idx, value in enumerate(targets)
                if value is not None and not isinstance(value, dict)
            ]
        )
        if errors:
            raise ValueError(f"Dataset validation failed: {errors}")

        target_errors: list[str] = []
        for idx, value in enumerate(targets):
            if value is None:
                continue
            if not isinstance(value, dict):
                continue  # Already caught above
            for model_id, score in value.items():
                if model_id not in allowed_model_ids:
                    target_errors.append(
                        f"Row {idx}: unknown model_id '{model_id}'. "
                        f"Allowed model IDs: {sorted(allowed_model_ids)}"
                    )
                if not isinstance(model_id, str):
                    target_errors.append(f"Row {idx}: model_id must be str")
                if score not in (0, 1):
                    target_errors.append(
                        f"Row {idx}: target for {model_id} must be 0 or 1"
                    )

        if target_errors:
            raise ValueError(f"Dataset validation failed: {target_errors}")

    def _make_embedder(self) -> Embedder:
        if self.embedder is not None:
            return self.embedder

        return SentenceTransformers(
            model=self.embedding_model,
            batch_size=self.embedding_batch_size,
            normalize=self.embedding_normalize,
            trust_remote_code=self.allow_trust_remote_code,
            device=self.device,
        )

    def _reduce_or_pass(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, ReductionPayload | None]:
        if self.reducer is None:
            return embeddings, None

        reduced = self.reducer.fit_transform(embeddings)
        return reduced, self.reducer.checkpoint_payload()

    def _make_clusterer(self) -> Clusterer:
        if self.clusterer is None:
            return KMeansClusterer(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )

        return self.clusterer

    def _compile_prototype_routing_structure(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        raw_labels: np.ndarray,
        inertia: float | None,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Compile raw clustering output into runtime prototype routing.

        All algorithms are served through nearest-prototype routing. This step
        takes the clusterer's fitted output, keeps the prototype ordering that
        corresponds to the fitted labels, and then reassigns the training samples
        using the runtime nearest-prototype rule. Any prototype that would never
        be selected by runtime routing is removed and the resulting labels are
        compacted to a dense 0..k-1 range.

        Returns:
            Tuple of (routing_prototypes, routing_labels, routing_inertia).
            routing_inertia is None when the clusterer is not a KMeans variant.
        """
        raw_labels = np.asarray(raw_labels)
        centroids = np.asarray(centroids, dtype=np.float32)

        valid_mask = raw_labels >= 0
        valid_raw_labels = raw_labels[valid_mask]
        valid_embeddings = embeddings[valid_mask]

        if len(valid_raw_labels) == 0 or len(centroids) == 0:
            return (
                np.empty((0, embeddings.shape[1]), dtype=np.float32),
                np.full(len(embeddings), -1, dtype=np.intp),
                None,
            )

        unique_raw = np.unique(valid_raw_labels)
        compact_centroids_list: list[np.ndarray] = []
        for raw_label in unique_raw:
            if raw_label < len(centroids):
                compact_centroids_list.append(centroids[raw_label])

        compact_centroids_arr = np.asarray(compact_centroids_list, dtype=np.float32)

        while True:
            distances = np.sqrt(
                np.sum(
                    (
                        valid_embeddings[:, np.newaxis, :]
                        - compact_centroids_arr[np.newaxis, :, :]
                    )
                    ** 2,
                    axis=2,
                )
            )
            nearest = distances.argmin(axis=1)

            used_mask = np.zeros(len(compact_centroids_arr), dtype=bool)
            used_mask[np.unique(nearest)] = True
            if bool(np.all(used_mask)):
                break

            compact_centroids_arr = compact_centroids_arr[used_mask]

        compact_labels = np.full(len(embeddings), -1, dtype=np.intp)
        compact_labels[valid_mask] = nearest

        compact_inertia: float | None = None
        if inertia is not None:
            closest_dists = distances[np.arange(len(valid_embeddings)), nearest]
            compact_inertia = float((closest_dists**2).sum())

        return compact_centroids_arr, compact_labels, compact_inertia

    def _calc_scores(
        self,
        labels: np.ndarray,
        targets: list[dict[str, int] | None],
        model_ids: list[str],
        n_clusters: int,
    ) -> dict[str, list[float]]:
        allowed_model_ids = set(model_ids)

        # Validate all targets are from allowed model IDs
        for row_targets in targets:
            if row_targets is None:
                continue
            for model_id in row_targets.keys():
                if model_id not in allowed_model_ids:
                    raise ValueError(
                        f"Unknown model_id '{model_id}' in targets. "
                        f"Allowed: {sorted(allowed_model_ids)}"
                    )

        # Use numpy for faster computation
        valid_mask = labels >= 0
        valid_labels = np.unique(labels[valid_mask])

        label_map = (
            {int(label): i for i, label in enumerate(valid_labels)}
            if len(valid_labels) == n_clusters
            else {i: i for i in range(n_clusters)}
        )

        # Pre-allocate accumulator arrays
        n_models = len(model_ids)
        model_to_idx = {mid: i for i, mid in enumerate(model_ids)}
        sums = np.zeros((n_models, n_clusters), dtype=np.float64)
        counts = np.zeros((n_models, n_clusters), dtype=np.int32)

        # Vectorized accumulation
        for row_idx, (label, row_targets) in enumerate(zip(labels, targets)):
            if label < 0:
                continue
            cluster_idx = label_map.get(int(label))
            if cluster_idx is None:
                continue
            if row_targets is None:
                continue
            for model_id, score in row_targets.items():
                model_idx = model_to_idx.get(model_id)
                if model_idx is not None:
                    sums[model_idx, cluster_idx] += float(score)
                    counts[model_idx, cluster_idx] += 1

        # Raise if any cluster has no data
        if np.any(counts == 0):
            zero_count = int(np.sum(counts == 0))
            raise ValueError(
                f"No training data for {zero_count} cluster-model combination(s). "
                "Consider increasing cluster size or reducing n_clusters."
            )

        # Return scores (higher is better) - already in 0-1 range
        scores = sums / counts

        return {mid: scores[i].tolist() for i, mid in enumerate(model_ids)}
