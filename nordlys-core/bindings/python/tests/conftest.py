"""Pytest fixtures for Python binding tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest


# Sample checkpoint for testing
SAMPLE_CHECKPOINT = {
    "cluster_centers": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    "models": [
        {
            "model_id": "openai/gpt-4",
            "scores": [0.99, 0.98, 0.985],
        },
        {
            "model_id": "anthropic/claude-3",
            "scores": [0.98, 0.99, 0.975],
        },
    ],
    "embedding": {
        "model": "test-model",
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
    "reduction": None,
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
def nordlys(sample_checkpoint_json: str):
    """Create a Nordlys instance from sample checkpoint."""
    from nordlys_core import Nordlys, NordlysCheckpoint

    checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
    return Nordlys.from_checkpoint(checkpoint)
