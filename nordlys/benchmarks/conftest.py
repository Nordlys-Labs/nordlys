"""Shared pytest fixtures for benchmarks."""

import numpy as np
import pandas as pd
import pytest

from nordlys import Dataset, ModelConfig, Router, Trainer
from nordlys.clustering import KMeansClusterer
from tests.utils.fake_embedder import FakeEmbedder, _to_dataset


@pytest.fixture
def benchmark_models() -> list[ModelConfig]:
    """Model configurations for benchmarking."""
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
        ModelConfig(
            id="anthropic/claude-3-haiku",
            cost_input=0.25,
            cost_output=1.25,
        ),
        ModelConfig(
            id="openai/gpt-4-turbo",
            cost_input=10.0,
            cost_output=30.0,
        ),
    ]


@pytest.fixture
def small_training_data(benchmark_models: list[ModelConfig]) -> pd.DataFrame:
    """Create small training DataFrame (~100 samples) for benchmarks."""
    np.random.seed(42)
    n_samples = 100

    # Generate diverse questions
    base_questions = [
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

    # Repeat and vary questions to reach ~100 samples
    questions = []
    for i in range(n_samples):
        base_q = base_questions[i % len(base_questions)]
        questions.append(f"{base_q} (variant {i})")

    data: dict[str, list[str] | list[float]] = {"questions": questions}
    for model in benchmark_models:
        data[model.id] = np.random.uniform(0.5, 1.0, n_samples).tolist()

    return pd.DataFrame(data)


@pytest.fixture
def medium_training_data(benchmark_models: list[ModelConfig]) -> pd.DataFrame:
    """Create medium training DataFrame (~1000 samples) for benchmarks."""
    np.random.seed(42)
    n_samples = 1000

    # Generate diverse questions
    base_questions = [
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

    # Repeat and vary questions to reach ~1000 samples
    questions = []
    for i in range(n_samples):
        base_q = base_questions[i % len(base_questions)]
        questions.append(f"{base_q} (variant {i})")

    data: dict[str, list[str] | list[float]] = {"questions": questions}
    for model in benchmark_models:
        data[model.id] = np.random.uniform(0.5, 1.0, n_samples).tolist()

    return pd.DataFrame(data)


@pytest.fixture
def large_training_data(benchmark_models: list[ModelConfig]) -> pd.DataFrame:
    """Create large training DataFrame (~10000 samples) for benchmarks."""
    np.random.seed(42)
    n_samples = 10000

    # Generate diverse questions
    base_questions = [
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

    # Repeat and vary questions to reach ~10000 samples
    questions = []
    for i in range(n_samples):
        base_q = base_questions[i % len(base_questions)]
        questions.append(f"{base_q} (variant {i})")

    data: dict[str, list[str] | list[float]] = {"questions": questions}
    for model in benchmark_models:
        data[model.id] = np.random.uniform(0.5, 1.0, n_samples).tolist()

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_embeddings() -> np.ndarray:
    """Generate synthetic embeddings for clustering/reduction benchmarks."""
    np.random.seed(42)
    # Generate realistic embedding dimensions (384 is common for sentence-transformers)
    n_samples = 1000
    n_features = 384
    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def fitted_nordlys(
    benchmark_models: list[ModelConfig],
    small_training_data: pd.DataFrame,
) -> Router:
    """Pre-fitted Router instance for routing benchmarks."""
    trainer = Trainer(
        models=benchmark_models,
        embedder=FakeEmbedder(),
        clusterer=KMeansClusterer(n_clusters=10, random_state=42),
    )
    checkpoint = trainer.fit(_to_dataset(small_training_data, benchmark_models))
    return Router(checkpoint=checkpoint)


@pytest.fixture
def small_dataset(
    benchmark_models: list[ModelConfig], small_training_data: pd.DataFrame
) -> Dataset:
    return _to_dataset(small_training_data, benchmark_models)


@pytest.fixture
def medium_dataset(
    benchmark_models: list[ModelConfig], medium_training_data: pd.DataFrame
) -> Dataset:
    return _to_dataset(medium_training_data, benchmark_models)


@pytest.fixture
def large_dataset(
    benchmark_models: list[ModelConfig], large_training_data: pd.DataFrame
) -> Dataset:
    return _to_dataset(large_training_data, benchmark_models)
