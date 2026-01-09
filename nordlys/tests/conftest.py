"""Pytest fixtures for nordlys tests."""

import numpy as np
import pandas as pd
import pytest

from nordlys import ModelConfig


@pytest.fixture
def sample_models() -> list[ModelConfig]:
    """Return sample model configurations for testing."""
    return [
        ModelConfig(
            id="openai/gpt-4",
            cost_input=30.0,
            cost_output=60.0,
        ),
        ModelConfig(
            id="anthropic/claude-3-sonnet",
            cost_input=15.0,
            cost_output=75.0,
        ),
        ModelConfig(
            id="openai/gpt-3.5-turbo",
            cost_input=0.5,
            cost_output=1.5,
        ),
    ]


@pytest.fixture
def sample_model() -> ModelConfig:
    """Return a single sample model configuration."""
    return ModelConfig(
        id="openai/gpt-4",
        cost_input=30.0,
        cost_output=60.0,
    )


@pytest.fixture
def small_training_data(sample_models: list[ModelConfig]) -> pd.DataFrame:
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
        data[model.id] = np.random.uniform(0.5, 1.0, n_samples).tolist()

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_embeddings() -> np.ndarray:
    """Generate synthetic embeddings (50 samples, 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(50, 384).astype(np.float32)
