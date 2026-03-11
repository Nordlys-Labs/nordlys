"""Unit tests for checkpoint construction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys.checkpoint import build_checkpoint


def test_build_checkpoint_success() -> None:
    checkpoint = build_checkpoint(
        cluster_centers=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        models=[
            {
                "model_id": "gpt-4",
                "cost_per_1m_input_tokens": 1.0,
                "cost_per_1m_output_tokens": 2.0,
                "error_rates": [0.1, 0.2],
            }
        ],
        embedding={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "trust_remote_code": False,
        },
        clustering={
            "n_clusters": 2,
            "random_state": 42,
            "max_iter": 300,
            "n_init": 10,
            "algorithm": "lloyd",
            "normalization": "l2",
        },
        reduction=None,
        metrics={
            "n_samples": 2,
            "cluster_sizes": [1, 1],
            "silhouette_score": None,
            "inertia": None,
        },
    )

    assert checkpoint.version == "3.0"
    assert checkpoint.reduction is None
    assert checkpoint.clustering.n_clusters == 2


def test_build_checkpoint_rejects_missing_models() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        build_checkpoint(
            cluster_centers=np.asarray([[0.0, 1.0]], dtype=np.float32),
            models=[],
            embedding={"model": "x", "trust_remote_code": False},
            clustering={
                "n_clusters": 1,
                "random_state": 42,
                "max_iter": 300,
                "n_init": 10,
                "algorithm": "lloyd",
                "normalization": "l2",
            },
            reduction=None,
            metrics={
                "n_samples": 1,
                "cluster_sizes": [1],
                "silhouette_score": None,
                "inertia": None,
            },
        )


def test_build_checkpoint_rejects_error_rate_length_mismatch() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        build_checkpoint(
            cluster_centers=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            models=[
                {
                    "model_id": "gpt-4",
                    "cost_per_1m_input_tokens": 1.0,
                    "cost_per_1m_output_tokens": 2.0,
                    "error_rates": [0.1],
                }
            ],
            embedding={"model": "x", "trust_remote_code": False},
            clustering={
                "n_clusters": 2,
                "random_state": 42,
                "max_iter": 300,
                "n_init": 10,
                "algorithm": "lloyd",
                "normalization": "l2",
            },
            reduction=None,
            metrics={
                "n_samples": 2,
                "cluster_sizes": [1, 1],
                "silhouette_score": None,
                "inertia": None,
            },
        )


def test_build_checkpoint_rejects_missing_error_rates() -> None:
    with pytest.raises(ValueError, match="missing error_rates"):
        build_checkpoint(
            cluster_centers=np.asarray([[0.0, 1.0]], dtype=np.float32),
            models=[
                {
                    "model_id": "gpt-4",
                    "cost_per_1m_input_tokens": 1.0,
                    "cost_per_1m_output_tokens": 2.0,
                }
            ],
            embedding={"model": "x", "trust_remote_code": False},
            clustering={
                "n_clusters": 1,
                "random_state": 42,
                "max_iter": 300,
                "n_init": 10,
                "algorithm": "lloyd",
                "normalization": "l2",
            },
            reduction=None,
            metrics={
                "n_samples": 1,
                "cluster_sizes": [1],
                "silhouette_score": None,
                "inertia": None,
            },
        )
