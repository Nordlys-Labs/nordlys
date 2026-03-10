"""Checkpoint construction helpers.

This module centralizes checkpoint payload creation so Trainer and Router
emit the same schema.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from nordlys_core import NordlysCheckpoint


def build_checkpoint(
    *,
    cluster_centers: np.ndarray,
    models: list[dict[str, Any]],
    embedding: dict[str, Any],
    clustering: dict[str, Any],
    metrics: dict[str, Any],
    version: str = "2.0",
) -> NordlysCheckpoint:
    """Build a validated ``NordlysCheckpoint`` from normalized sections."""
    n_clusters = int(cluster_centers.shape[0])
    _validate_models(models=models, n_clusters=n_clusters)

    payload = {
        "version": version,
        "cluster_centers": np.asarray(cluster_centers, dtype=np.float32).tolist(),
        "models": models,
        "embedding": embedding,
        "clustering": clustering,
        "metrics": metrics,
    }
    return NordlysCheckpoint.from_json_string(json.dumps(payload))


def _validate_models(*, models: list[dict[str, Any]], n_clusters: int) -> None:
    if not models:
        raise ValueError("Checkpoint must contain at least one model")

    for model in models:
        model_id = str(model.get("model_id", ""))
        error_rates = model.get("error_rates")
        if not isinstance(error_rates, list):
            raise ValueError(f"Model '{model_id}' missing error_rates list")
        if len(error_rates) != n_clusters:
            raise ValueError(
                f"Model '{model_id}' has {len(error_rates)} error rates, "
                f"expected {n_clusters}"
            )
