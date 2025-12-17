"""Pytest fixtures for Python binding tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest


# Sample profile for testing
SAMPLE_PROFILE = {
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
def sample_profile_json() -> str:
    """Return sample profile as JSON string."""
    return json.dumps(SAMPLE_PROFILE)


@pytest.fixture
def sample_profile_path(tmp_path: Path) -> Path:
    """Create a temporary profile file and return its path."""
    profile_path = tmp_path / "test_profile.json"
    profile_path.write_text(json.dumps(SAMPLE_PROFILE))
    return profile_path


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Return a sample 4-dim embedding matching the test profile."""
    return np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_profile_json_float64() -> str:
    """Return sample profile with float64 dtype as JSON string."""
    profile = copy.deepcopy(SAMPLE_PROFILE)
    profile["metadata"]["dtype"] = "float64"
    return json.dumps(profile)


@pytest.fixture
def router(sample_profile_json: str):
    """Create a Router instance from sample profile."""
    from adaptive_core_ext import Router

    return Router.from_json_string(sample_profile_json)


@pytest.fixture
def router_float64(sample_profile_json_float64: str):
    """Create a float64 Router instance from sample profile."""
    from adaptive_core_ext import Router

    return Router.from_json_string(sample_profile_json_float64)
