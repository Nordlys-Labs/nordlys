"""Dataset-native trainer that compiles router checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nordlys.clustering import Clusterer, KMeansClusterer, compute_cluster_metrics
from nordlys.dataset import Dataset
from nordlys.embeddings import Embedder, SentenceTransformers
from nordlys.router import ModelConfig
from nordlys.reduction import Reducer
from nordlys_core import NordlysCheckpoint


@dataclass(frozen=True)
class Trainer:
    """Train router checkpoints from Dataset objects.

    Canonical path: ``Dataset -> Trainer.fit() -> NordlysCheckpoint``.

    Example:
        >>> from nordlys import Dataset, Trainer, ModelConfig
        >>> from nordlys.clustering import KMeansClusterer
        >>> dataset = Dataset.from_list([
        ...     {"id": "1", "input": "Fix the parser", "targets": {"gpt-4": 1}},
        ... ])
        >>> # Pass your own clusterer
        >>> trainer = Trainer(
        ...     models=[ModelConfig(id="gpt-4", cost_input=30.0, cost_output=60.0)],
        ...     clusterer=KMeansClusterer(n_clusters=10),
        ... )
        >>> checkpoint = trainer.fit(dataset)
    """

    models: list[ModelConfig]
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

    def fit(self, dataset: Dataset) -> NordlysCheckpoint:
        """Fit clustering pipeline and return a checkpoint."""
        self._validate(dataset)

        model_ids = [m.id for m in self.models]
        embedder = self._make_embedder()
        embeddings = embedder.encode(dataset.column(self.input_col))
        cluster_input = self._reduce_or_pass(embeddings)

        clusterer = self._make_clusterer()
        clusterer.fit(cluster_input)

        centroids = clusterer.cluster_centers_
        labels = clusterer.labels_
        n_clusters = len(centroids)

        error_rates = self._calc_error_rates(
            labels,
            dataset.column(self.target_col),
            model_ids,
            n_clusters,
        )

        inertia: float | None = None
        if isinstance(clusterer, KMeansClusterer):
            inertia = clusterer.inertia_

        metrics = compute_cluster_metrics(
            cluster_input,
            labels,
            inertia,
        )

        payload = {
            "version": "2.0",
            "cluster_centers": np.asarray(centroids, dtype=np.float32).tolist(),
            "models": [
                {
                    "model_id": m.id,
                    "cost_per_1m_input_tokens": m.cost_input,
                    "cost_per_1m_output_tokens": m.cost_output,
                    "error_rates": error_rates[m.id],
                }
                for m in self.models
            ],
            "embedding": {
                **embedder.checkpoint_config(),
            },
            "clustering": {
                "n_clusters": n_clusters,
                "random_state": self.random_state,
                "max_iter": 300,
                "n_init": 10,
                "algorithm": "lloyd",
                "normalization": "l2" if self.embedding_normalize else "none",
            },
            "metrics": {
                "n_samples": metrics.n_samples,
                "cluster_sizes": metrics.cluster_sizes,
                "silhouette_score": metrics.silhouette_score,
                "inertia": metrics.inertia,
            },
        }
        return NordlysCheckpoint.from_json_string(json.dumps(payload))

    def _validate(self, dataset: Dataset) -> None:
        if not self.models:
            raise ValueError("At least one model is required")

        allowed_model_ids = {m.id for m in self.models}

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

    def _reduce_or_pass(self, embeddings: np.ndarray) -> np.ndarray:
        if self.reducer is not None:
            raise ValueError(
                "Reducer is not supported for checkpoints. "
                "The checkpoint stores centroids in reduced space, but "
                "Router._from_checkpoint cannot restore the reducer, causing "
                "mismatched comparisons during inference. "
                "Set reducer=None to use full embedding space."
            )
        return embeddings

    def _make_clusterer(self) -> Clusterer:
        if self.clusterer is None:
            return KMeansClusterer(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )

        return self.clusterer

    def _calc_error_rates(
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

        # Compute error rates - raise if any cluster has no data
        if np.any(counts == 0):
            zero_count = int(np.sum(counts == 0))
            raise ValueError(
                f"No training data for {zero_count} cluster-model combination(s). "
                "Consider increasing cluster size or reducing n_clusters."
            )

        accuracies = sums / counts
        error_rates = 1.0 - accuracies

        return {mid: error_rates[i].tolist() for i, mid in enumerate(model_ids)}
