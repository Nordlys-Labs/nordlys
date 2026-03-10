"""Dataset-native trainer that compiles router checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from nordlys.clustering import Clusterer, KMeansClusterer, compute_cluster_metrics
from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer
from nordlys.dataset import Dataset
from nordlys.nordlys import ModelConfig
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

    clusterer: Clusterer | str | None = None
    n_clusters: int = 20

    reducer: Reducer | None = None

    random_state: int = 42
    allow_trust_remote_code: bool = False
    device: Literal["cpu", "cuda"] = "cpu"

    kmeans_max_iter: int = 300
    kmeans_n_init: int = 10
    kmeans_algorithm: str = "lloyd"

    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_cluster_selection_method: str = "eom"

    default_error_rate: float = 0.5

    def fit(self, dataset: Dataset) -> NordlysCheckpoint:
        """Fit clustering pipeline and return a checkpoint."""
        self._validate(dataset)

        model_ids = [m.id for m in self.models]
        embeddings = self._embed(dataset.column(self.input_col))
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

        metrics = compute_cluster_metrics(
            cluster_input,
            labels,
            getattr(clusterer, "inertia_", None),
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
                "model": self.embedding_model,
                "trust_remote_code": self.allow_trust_remote_code,
            },
            "clustering": {
                "n_clusters": n_clusters,
                "random_state": self.random_state,
                "max_iter": self.kmeans_max_iter,
                "n_init": self.kmeans_n_init,
                "algorithm": self.kmeans_algorithm,
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

        for idx, value in enumerate(targets):
            if value is None:
                continue
            for model_id, score in value.items():
                if not isinstance(model_id, str):
                    raise ValueError(f"Row {idx}: model_id must be str")
                if score not in (0, 1):
                    raise ValueError(f"Row {idx}: target for {model_id} must be 0 or 1")

    def _embed(self, texts: list[str]) -> np.ndarray:
        encoder = SentenceTransformer(
            self.embedding_model,
            device=self.device,
            trust_remote_code=self.allow_trust_remote_code,
        )
        encoder.tokenizer.clean_up_tokenization_spaces = False

        return encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.embedding_batch_size,
            normalize_embeddings=self.embedding_normalize,
        )

    def _reduce_or_pass(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(embeddings) if self.reducer else embeddings

    def _make_clusterer(self) -> Clusterer:
        if self.clusterer is None:
            return KMeansClusterer(
                n_clusters=self.n_clusters,
                max_iter=self.kmeans_max_iter,
                n_init=self.kmeans_n_init,
                random_state=self.random_state,
                algorithm=self.kmeans_algorithm,
            )

        if isinstance(self.clusterer, Clusterer):
            return self.clusterer

        clusterer_creators = {
            "kmeans": lambda: KMeansClusterer(
                n_clusters=self.n_clusters,
                max_iter=self.kmeans_max_iter,
                n_init=self.kmeans_n_init,
                random_state=self.random_state,
                algorithm=self.kmeans_algorithm,
            ),
            "hdbscan": lambda: HDBSCANClusterer(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric=self.hdbscan_metric,
                cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
                cluster_selection_method=self.hdbscan_cluster_selection_method,
            ),
        }

        creator = clusterer_creators.get(self.clusterer)
        if creator is None:
            raise ValueError(f"Unknown clusterer: {self.clusterer}")
        return creator()

    def _calc_error_rates(
        self,
        labels: np.ndarray,
        targets: list[dict[str, int] | None],
        model_ids: list[str],
        n_clusters: int,
    ) -> dict[str, list[float]]:
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

        # Compute error rates with default fallback
        default = self.default_error_rate
        with np.errstate(divide="ignore", invalid="ignore"):
            accuracies = np.where(counts > 0, sums / counts, default)
            error_rates = 1.0 - accuracies

        # Handle zero-count clusters
        error_rates = np.where(counts == 0, default, error_rates)

        return {mid: error_rates[i].tolist() for i, mid in enumerate(model_ids)}
