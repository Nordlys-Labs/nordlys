"""Fixtures for integration tests."""

import numpy as np
import pandas as pd
import pytest

from nordlys import ModelConfig, Nordlys


@pytest.fixture
def three_models() -> list[ModelConfig]:
    """Standard 3-model setup for integration tests."""
    return [
        ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
        ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
        ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ]


@pytest.fixture
def training_data_100(three_models: list[ModelConfig]) -> pd.DataFrame:
    """100 realistic questions with 3 model accuracy scores."""
    np.random.seed(42)
    n_samples = 100

    # Generate diverse question types
    questions = []

    # Math questions (simple - cheap model good)
    math_questions = [
        "What is 2+2?",
        "Calculate 15 * 23",
        "What is the square root of 144?",
        "Is 17 a prime number?",
        "What is 25% of 80?",
    ]

    # Coding questions (medium - varies by complexity)
    code_questions = [
        "Write a Python hello world",
        "How to sort a list in Python?",
        "Write a function to reverse a string",
        "Implement binary search",
        "Create a REST API endpoint",
        "Write a decorator in Python",
        "Implement a linked list",
        "Write a recursive fibonacci",
        "Create a class hierarchy",
        "Implement depth-first search",
    ]

    # Reasoning questions (complex - expensive model good)
    reasoning_questions = [
        "Explain quantum mechanics",
        "What is the theory of relativity?",
        "Explain consciousness",
        "How does the brain work?",
        "What is the meaning of existence?",
    ]

    # Factual questions (simple/medium - varies)
    factual_questions = [
        "What is the capital of France?",
        "When was the internet invented?",
        "Who wrote Hamlet?",
        "What causes earthquakes?",
        "How does photosynthesis work?",
        "What is DNA?",
        "Explain gravity",
        "What are black holes?",
        "How do vaccines work?",
        "What is climate change?",
    ]

    # Build full question list
    all_questions = (
        math_questions * 6
        + code_questions * 4
        + reasoning_questions * 6
        + factual_questions * 3
    )

    questions = all_questions[:n_samples]

    # Generate accuracy scores based on question type
    # GPT-4: expensive, good at everything (especially complex)
    # GPT-3.5: cheap, good at simple/medium tasks
    # Claude-3-Sonnet: mid-tier pricing, balanced

    gpt4_scores = []
    gpt35_scores = []
    claude_scores = []

    for q in questions:
        if any(word in q.lower() for word in ["what is", "calculate", "is ", "square"]):
            # Simple factual/math - all models good, cheap model excels
            gpt4_scores.append(np.random.uniform(0.85, 0.95))
            gpt35_scores.append(np.random.uniform(0.90, 0.98))
            claude_scores.append(np.random.uniform(0.88, 0.96))
        elif any(
            word in q.lower() for word in ["write", "implement", "create", "function"]
        ):
            # Coding - medium complexity, gpt-4 best
            gpt4_scores.append(np.random.uniform(0.88, 0.96))
            gpt35_scores.append(np.random.uniform(0.70, 0.85))
            claude_scores.append(np.random.uniform(0.82, 0.92))
        elif any(
            word in q.lower()
            for word in ["explain", "theory", "consciousness", "meaning"]
        ):
            # Complex reasoning - expensive models much better
            gpt4_scores.append(np.random.uniform(0.90, 0.98))
            gpt35_scores.append(np.random.uniform(0.50, 0.70))
            claude_scores.append(np.random.uniform(0.85, 0.94))
        else:
            # Default - balanced
            gpt4_scores.append(np.random.uniform(0.80, 0.95))
            gpt35_scores.append(np.random.uniform(0.75, 0.90))
            claude_scores.append(np.random.uniform(0.78, 0.92))

    return pd.DataFrame(
        {
            "questions": questions,
            "openai/gpt-4": gpt4_scores,
            "openai/gpt-3.5-turbo": gpt35_scores,
            "anthropic/claude-3-sonnet": claude_scores,
        }
    )


@pytest.fixture
def fitted_nordlys(
    three_models: list[ModelConfig], training_data_100: pd.DataFrame
) -> Nordlys:
    """Pre-fitted Nordlys instance for testing."""
    nordlys = Nordlys(models=three_models, nr_clusters=10)
    nordlys.fit(training_data_100)
    return nordlys
