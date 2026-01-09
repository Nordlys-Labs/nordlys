"""Pytest fixtures for Python binding tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest


# Sample checkpoint for testing (v2.0 format)
SAMPLE_CHECKPOINT = {
    "version": "2.0",
    "cluster_centers": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    "models": [
        {
            "model_id": "openai/gpt-4",
            "cost_per_1m_input_tokens": 30.0,
            "cost_per_1m_output_tokens": 60.0,
            "error_rates": [0.01, 0.02, 0.015],
        },
        {
            "model_id": "anthropic/claude-3",
            "cost_per_1m_input_tokens": 15.0,
            "cost_per_1m_output_tokens": 75.0,
            "error_rates": [0.02, 0.01, 0.025],
        },
    ],
    "embedding": {
        "model": "test-model",
        "dtype": "float32",
        "trust_remote_code": False,
    },
    "clustering": {
        "n_clusters": 3,
        "random_state": 42,
        "max_iter": 300,
        "n_init": 10,
        "algorithm": "lloyd",
        "normalization": "l2",
    },
    "routing": {
        "cost_bias_min": 0.0,
        "cost_bias_max": 1.0,
    },
    "metrics": {
        "silhouette_score": 0.85,
    },
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
    checkpoint["embedding"]["dtype"] = "float64"
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
