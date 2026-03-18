"""TypedDict schemas for checkpoint payloads."""

from __future__ import annotations

from typing import TypedDict


class CheckpointModelEntry(TypedDict):
    """Model entry in the checkpoint models list."""

    model_id: str
    scores: list[float]


class EmbeddingConfig(TypedDict, total=False):
    """Embedding configuration stored in checkpoints."""

    model: str
    trust_remote_code: bool
    embedding_prompt_name: str | None
    embedding_prompt: str | None
    max_seq_length: int | None
    truncate_dim: int | None
    revision: str | None


class ClusteringConfig(TypedDict):
    """Clustering configuration stored in checkpoints."""

    n_clusters: int
    random_state: int
    max_iter: int
    n_init: int
    algorithm: str
    normalization: str


class CheckpointMetrics(TypedDict, total=False):
    """Clustering metrics stored in checkpoints."""

    n_samples: int | None
    cluster_sizes: list[int] | None
    silhouette_score: float | None
    inertia: float | None
