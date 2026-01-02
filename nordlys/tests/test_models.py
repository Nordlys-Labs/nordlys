"""Tests for ModelConfig class."""

import pytest
from pydantic import ValidationError

from nordlys import ModelConfig


class TestModelConfigCreation:
    """Test ModelConfig initialization and validation."""

    def test_create_valid_model(self, sample_model):
        """Test creating a valid model configuration."""
        assert sample_model.id == "openai/gpt-4"
        assert sample_model.cost_input == 30.0
        assert sample_model.cost_output == 60.0

    def test_create_model_with_zero_costs(self):
        """Test creating a model with zero costs."""
        model = ModelConfig(id="test/free-model", cost_input=0.0, cost_output=0.0)
        assert model.cost_input == 0.0
        assert model.cost_output == 0.0

    def test_negative_cost_input_fails(self):
        """Test that negative cost_input raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(id="test/model", cost_input=-1.0, cost_output=1.0)

    def test_negative_cost_output_fails(self):
        """Test that negative cost_output raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(id="test/model", cost_input=1.0, cost_output=-1.0)

    def test_empty_id_fails(self):
        """Test that empty model ID raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(id="", cost_input=1.0, cost_output=1.0)


class TestModelConfigProperties:
    """Test ModelConfig computed properties."""

    def test_cost_average(self, sample_model):
        """Test cost_average property calculation."""
        assert sample_model.cost_average == 45.0  # (30 + 60) / 2

    def test_provider_extraction(self, sample_model):
        """Test provider extraction from model ID."""
        assert sample_model.provider == "openai"

    def test_model_name_extraction(self, sample_model):
        """Test model name extraction from model ID."""
        assert sample_model.model_name == "gpt-4"

    def test_provider_without_slash(self):
        """Test provider extraction when ID has no slash."""
        model = ModelConfig(id="model-name", cost_input=1.0, cost_output=1.0)
        assert model.provider == ""
        assert model.model_name == "model-name"

    def test_model_frozen(self, sample_model):
        """Test that ModelConfig is immutable."""
        with pytest.raises(ValidationError):
            sample_model.id = "new/model"


class TestModelConfigComparison:
    """Test ModelConfig with different configurations."""

    def test_different_models_not_equal(self, sample_models):
        """Test that different models are not equal."""
        assert sample_models[0] != sample_models[1]

    def test_same_config_equal(self):
        """Test that identical configurations are equal."""
        model1 = ModelConfig(id="test/model", cost_input=1.0, cost_output=2.0)
        model2 = ModelConfig(id="test/model", cost_input=1.0, cost_output=2.0)
        assert model1 == model2
