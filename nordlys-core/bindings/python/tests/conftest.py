"""Pytest fixtures for Python binding tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest


# Sample checkpoint for testing
SAMPLE_CHECKPOINT = {
    "metadata": {
        "n_clusters": 3,
        "embedding_model": "test-model",
        "silhouette_score": 0.85,
        "clustering": {"n_init": 10, "algorithm": "lloyd"},
        "routing": {
            "lambda_min": 0.0,
            "lambda_max": 2.0,
            "max_alternatives": 2,
            "default_cost_preference": 0.5,
        },
    },
    "cluster_centers": {
        "n_clusters": 3,
        "feature_dim": 4,
        "cluster_centers": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    },
    "models": [
        {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
            "cost_per_1m_output_tokens": 60.0,
            "error_rates": [0.01, 0.02, 0.015],
        },
        {
            "provider": "anthropic",
            "model_name": "claude-3",
            "cost_per_1m_input_tokens": 15.0,
            "cost_per_1m_output_tokens": 75.0,
            "error_rates": [0.02, 0.01, 0.025],
        },
    ],
}


@pytest.fixture
def sample_checkpoint_json() -> str:
    """Return sample checkpoint as JSON string."""
    return json.dumps(SAMPLE_CHECKPOINT)


@pytest.fixture
def sample_checkpoint_path(tmp_path: Path) -> Path:
    """Create a temporary checkpoint file and return its path."""
    checkpoint_path = tmp_path / "test_checkpoint.json"
    checkpoint_path.write_text(json.dumps(SAMPLE_CHECKPOINT))
    return checkpoint_path


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Return a sample 4-dim embedding matching the test checkpoint."""
    return np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_checkpoint_json_float64() -> str:
    """Return sample checkpoint with float64 dtype as JSON string."""
    checkpoint = copy.deepcopy(SAMPLE_CHECKPOINT)
    checkpoint["metadata"]["dtype"] = "float64"
    return json.dumps(checkpoint)


@pytest.fixture
def nordlys32(sample_checkpoint_json: str):
    """Create a Nordlys32 instance from sample checkpoint."""
    from nordlys_core_ext import Nordlys32, NordlysCheckpoint

    checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
    return Nordlys32.from_checkpoint(checkpoint)


@pytest.fixture
def nordlys64(sample_checkpoint_json_float64: str):
    """Create a Nordlys64 instance from sample checkpoint."""
    from nordlys_core_ext import Nordlys64, NordlysCheckpoint

    checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json_float64)
    return Nordlys64.from_checkpoint(checkpoint)
