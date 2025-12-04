"""Tests for model_resolver utilities."""

import pytest

from adaptive_router.models.api import Model
from adaptive_router_app.utils.model_resolver import resolve_models


def _make_model(provider: str, name: str) -> Model:
    return Model(
        provider=provider,
        model_name=name,
        cost_per_1m_input_tokens=1_000.0,
        cost_per_1m_output_tokens=2_000.0,
    )


class TestResolveModels:
    """Test resolve_models function."""

    def test_resolves_valid_models(self) -> None:
        models = [
            _make_model("openai", "gpt-4"),
            _make_model("anthropic", "claude-3.5-sonnet"),
        ]

        result = resolve_models(["openai/gpt-4"], models)

        assert len(result) == 1
        assert result[0].provider == "openai"
        assert result[0].model_name == "gpt-4"

    def test_resolves_models_with_variants(self) -> None:
        models = [
            _make_model("google", "gemini-2.0-flash-exp:free"),
            _make_model("google", "gemini-2.0-flash-001"),
        ]

        variant = resolve_models(["google/gemini-2.0-flash-exp:free"], models)
        assert variant[0].model_name == "gemini-2.0-flash-exp:free"

        base = resolve_models(["google/gemini-2.0-flash-001"], models)
        assert base[0].model_name == "gemini-2.0-flash-001"

    def test_raises_error_when_only_unknown_models(self) -> None:
        models = [_make_model("openai", "gpt-4")]

        with pytest.raises(ValueError, match="No requested models"):
            resolve_models(["anthropic/claude-3", "openai/gpt-5"], models)

    def test_raises_error_for_invalid_format(self) -> None:
        models = [_make_model("openai", "gpt-4")]

        with pytest.raises(ValueError, match="expected format"):
            resolve_models(["openai-gpt-4"], models)

    def test_raises_when_no_models_are_loaded(self) -> None:
        with pytest.raises(ValueError, match="No models loaded"):
            resolve_models(["openai/gpt-4"], [])
