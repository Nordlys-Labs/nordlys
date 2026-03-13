"""Pytest fixtures for nordlys tests."""

from nordlys_core import NordlysCheckpoint

import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_models() -> list[str]:
    """Return sample model IDs for testing."""
    return [
        "openai/gpt-4",
        "anthropic/claude-3-sonnet",
        "openai/gpt-3.5-turbo",
    ]


@pytest.fixture
def sample_model() -> str:
    """Return a single sample model ID."""
    return "openai/gpt-4"


@pytest.fixture
def small_training_data(sample_models: list[str]) -> pd.DataFrame:
    """Create minimal training DataFrame (20 samples) for fast tests."""
    np.random.seed(42)
    n_samples = 20

    questions = [
        "What is 2+2?",
        "Explain quantum mechanics briefly",
        "Write a Python hello world",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Write a sorting algorithm",
        "What is machine learning?",
        "Explain the theory of relativity",
        "Write a REST API endpoint",
        "What causes earthquakes?",
        "How do vaccines work?",
        "Write a binary search function",
        "What is artificial intelligence?",
        "Explain DNA replication",
        "Write a web scraper in Python",
        "What is the speed of light?",
        "How does encryption work?",
        "Write a recursive function",
        "What is blockchain?",
        "Explain neural networks",
    ]

    data: dict[str, list[str] | list[float]] = {"questions": questions}
    for model in sample_models:
        data[model] = np.random.uniform(0.5, 1.0, n_samples).tolist()

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_embeddings() -> np.ndarray:
    """Generate synthetic embeddings (50 samples, 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(50, 384).astype(np.float32)


@pytest.fixture
def sample_checkpoint(sample_models: list[str]) -> NordlysCheckpoint:
    """Return a small valid checkpoint for Router runtime tests."""
    model_payload = []
    for model in sample_models:
        model_payload.append(
            {
                "model_id": model,
                "scores": [0.8, 0.6],
            }
        )

    payload = {
        "cluster_centers": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "models": model_payload,
        "embedding": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "trust_remote_code": False,
            "embedding_prompt_name": "",
            "embedding_prompt": "",
            "max_seq_length": 0,
        },
        "clustering": {
            "n_clusters": 2,
            "random_state": 42,
            "max_iter": 300,
            "n_init": 10,
            "algorithm": "lloyd",
            "normalization": "l2",
        },
        "reduction": None,
        "metrics": {
            "n_samples": 2,
            "cluster_sizes": [1, 1],
            "silhouette_score": None,
            "inertia": None,
        },
    }
    return NordlysCheckpoint.from_json_string(json.dumps(payload))
