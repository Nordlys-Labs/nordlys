"""Pytest fixtures for nordlys tests."""

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
