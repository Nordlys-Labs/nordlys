"""Checkpoint construction helpers.

This module centralizes checkpoint payload creation so Trainer and Router
emit the same schema.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from nordlys.reduction.base import ReductionPayload
from nordlys_core import NordlysCheckpoint


def build_checkpoint(
    *,
    cluster_centers: np.ndarray,
    models: list[dict[str, Any]],
    embedding: dict[str, Any],
    clustering: dict[str, Any],
    reduction: ReductionPayload | None,
    metrics: dict[str, Any],
) -> NordlysCheckpoint:
    """Build a validated ``NordlysCheckpoint`` from normalized sections."""
    n_clusters = int(cluster_centers.shape[0])
    _validate_models(models=models, n_clusters=n_clusters)
    _validate_embedding(embedding)

    payload = {
        "cluster_centers": np.asarray(cluster_centers, dtype=np.float32).tolist(),
        "models": models,
        "embedding": embedding,
        "clustering": clustering,
        "reduction": None if reduction is None else reduction.model_dump(mode="json"),
        "metrics": metrics,
    }
    return NordlysCheckpoint.from_json_string(json.dumps(payload))


def _validate_models(*, models: list[dict[str, Any]], n_clusters: int) -> None:
    if not models:
        raise ValueError("Checkpoint must contain at least one model")

    for model in models:
        model_id = str(model.get("model_id", ""))
        scores = model.get("scores")
        if not isinstance(scores, list):
            raise ValueError(f"Model '{model_id}' missing scores list")
        if len(scores) != n_clusters:
            raise ValueError(
                f"Model '{model_id}' has {len(scores)} scores, expected {n_clusters}"
            )


def _validate_embedding(embedding: dict[str, Any]) -> None:
    model = embedding.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Checkpoint embedding must include a non-empty model")

    max_seq_length = embedding.get("max_seq_length", 0)
    if max_seq_length in (None, 0):
        return
    if not isinstance(max_seq_length, int) or max_seq_length <= 0:
        raise ValueError(
            "Checkpoint embedding max_seq_length must be a positive integer"
        )
