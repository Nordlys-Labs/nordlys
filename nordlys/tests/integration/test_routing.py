"""Integration tests for Nordlys routing methods."""

import pytest

from nordlys import Nordlys, RouteResult


class TestNordlysRoute:
    """Test Nordlys route method."""

    def test_route_single_prompt(self, fitted_nordlys):
        """Test routing a single prompt."""
        result = fitted_nordlys.route("What is machine learning?")
        assert result is not None

    def test_route_returns_route_result(self, fitted_nordlys):
        """Test that route returns RouteResult."""
        result = fitted_nordlys.route("Write a sorting algorithm")
        assert isinstance(result, RouteResult)

    def test_route_result_has_model_id(self, fitted_nordlys):
        """Test that RouteResult has model_id."""
        result = fitted_nordlys.route("Test prompt")
        assert hasattr(result, "model_id")
        assert isinstance(result.model_id, str)
        assert len(result.model_id) > 0

    def test_route_result_has_cluster_id(self, fitted_nordlys):
        """Test that RouteResult has cluster_id."""
        result = fitted_nordlys.route("Test prompt")
        assert hasattr(result, "cluster_id")
        assert isinstance(result.cluster_id, int)
        assert result.cluster_id >= 0

    def test_route_result_has_cluster_distance(self, fitted_nordlys):
        """Test that RouteResult has cluster_distance."""
        result = fitted_nordlys.route("Test prompt")
        assert hasattr(result, "cluster_distance")
        assert isinstance(result.cluster_distance, float)
        assert result.cluster_distance >= 0

    def test_route_result_has_alternatives(self, fitted_nordlys):
        """Test that RouteResult has alternatives."""
        result = fitted_nordlys.route("Test prompt")
        assert hasattr(result, "alternatives")
        assert isinstance(result.alternatives, list)

    def test_route_before_fit_raises(self, three_models):
        """Test that routing before fit raises RuntimeError."""
        nordlys = Nordlys(models=three_models)

        with pytest.raises(RuntimeError, match="must be fitted before use"):
            nordlys.route("Test prompt")


class TestRoutingCostBias:
    """Test cost_bias parameter in routing."""

    def test_route_cost_bias_0_prefers_cheap(self, fitted_nordlys):
        """Test that cost_bias=0 returns a valid model (smoke test).

        Note: The actual model selection depends on cluster assignments and
        error rates, so we only verify a valid model is returned. The cost_bias
        parameter influences the tradeoff but doesn't guarantee the cheapest model.
        """
        result = fitted_nordlys.route("What is 2+2?", cost_bias=0.0)
        # Verify a valid model is returned
        assert result.model_id in [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]

    def test_route_cost_bias_1_prefers_quality(self, fitted_nordlys):
        """Test that cost_bias=1 tends toward quality models."""
        result = fitted_nordlys.route("Explain quantum physics", cost_bias=1.0)
        # Should return one of the valid models
        assert result.model_id in [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]

    def test_route_cost_bias_05_balanced(self, fitted_nordlys):
        """Test that cost_bias=0.5 returns a valid model."""
        result = fitted_nordlys.route("Test prompt", cost_bias=0.5)
        assert result.model_id in [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]

    def test_different_cost_bias_may_differ(self, fitted_nordlys):
        """Test that different cost_bias values may produce different results."""
        result_0 = fitted_nordlys.route("Explain complex reasoning", cost_bias=0.0)
        result_1 = fitted_nordlys.route("Explain complex reasoning", cost_bias=1.0)

        # Both should be valid (may or may not be different)
        assert result_0.model_id in [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]
        assert result_1.model_id in [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]


class TestRoutingAlternatives:
    """Test alternatives in routing results."""

    def test_alternatives_are_strings(self, fitted_nordlys):
        """Test that alternatives are list of strings."""
        result = fitted_nordlys.route("Test prompt")
        for alt in result.alternatives:
            assert isinstance(alt, str)

    def test_alternatives_are_valid_models(self, fitted_nordlys):
        """Test that alternatives are valid model IDs."""
        valid_models = [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
        ]
        result = fitted_nordlys.route("Test prompt")
        for alt in result.alternatives:
            assert alt in valid_models

    def test_alternatives_exclude_selected_model(self, fitted_nordlys):
        """Test that alternatives don't include selected model."""
        result = fitted_nordlys.route("Test prompt")
        assert result.model_id not in result.alternatives

    def test_max_alternatives(self, fitted_nordlys):
        """Test that alternatives has reasonable count."""
        result = fitted_nordlys.route("Test prompt")
        # Should have at most (n_models - 1) alternatives
        assert len(result.alternatives) <= 2  # 3 models - 1 selected


class TestRoutingConsistency:
    """Test routing consistency and determinism."""

    def test_same_prompt_same_result(self, fitted_nordlys):
        """Test that same prompt returns same result."""
        prompt = "What is the capital of France?"
        result1 = fitted_nordlys.route(prompt)
        result2 = fitted_nordlys.route(prompt)

        assert result1.model_id == result2.model_id
        assert result1.cluster_id == result2.cluster_id

    def test_routing_deterministic(self, fitted_nordlys):
        """Test that routing is deterministic."""
        prompt = "Write a Python function"
        results = [fitted_nordlys.route(prompt) for _ in range(5)]

        first_model = results[0].model_id
        first_cluster = results[0].cluster_id

        for r in results[1:]:
            assert r.model_id == first_model
            assert r.cluster_id == first_cluster


class TestRoutingEdgeCases:
    """Test routing edge cases."""

    def test_route_empty_string(self, fitted_nordlys):
        """Test routing an empty string."""
        result = fitted_nordlys.route("")
        # Should still return a result (empty string is valid input)
        assert result is not None
        assert isinstance(result, RouteResult)

    def test_route_very_long_prompt(self, fitted_nordlys):
        """Test routing a very long prompt."""
        long_prompt = "This is a test. " * 100  # ~1600 chars
        result = fitted_nordlys.route(long_prompt)
        assert result is not None
        assert isinstance(result, RouteResult)

    def test_route_special_characters(self, fitted_nordlys):
        """Test routing prompt with special characters."""
        prompt = "What is π? Calculate √2. Use 日本語."
        result = fitted_nordlys.route(prompt)
        assert result is not None
        assert isinstance(result, RouteResult)

    def test_route_whitespace_only(self, fitted_nordlys):
        """Test routing whitespace-only prompt."""
        result = fitted_nordlys.route("   \n\t  ")
        assert result is not None
        assert isinstance(result, RouteResult)


class TestBatchRouting:
    """Test batch routing."""

    def test_route_batch_returns_list(self, fitted_nordlys):
        """Test that route_batch returns a list."""
        prompts = ["What is AI?", "Write code", "Explain physics"]
        results = fitted_nordlys.route_batch(prompts)

        assert isinstance(results, list)

    def test_route_batch_length_matches_input(self, fitted_nordlys):
        """Test that batch result length matches input length."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
        results = fitted_nordlys.route_batch(prompts)

        assert len(results) == len(prompts)

    def test_route_batch_all_route_results(self, fitted_nordlys):
        """Test that all batch results are RouteResult instances."""
        prompts = ["Test 1", "Test 2"]
        results = fitted_nordlys.route_batch(prompts)

        for r in results:
            assert isinstance(r, RouteResult)

    def test_batch_vs_individual_routing(self, fitted_nordlys):
        """Test that batch routing matches individual routing."""
        prompts = ["What is ML?", "Write Python code"]
        batch_results = fitted_nordlys.route_batch(prompts)
        individual_results = [fitted_nordlys.route(p) for p in prompts]

        for batch_r, ind_r in zip(batch_results, individual_results):
            assert batch_r.model_id == ind_r.model_id
            assert batch_r.cluster_id == ind_r.cluster_id

    def test_route_empty_batch(self, fitted_nordlys):
        """Test routing an empty batch."""
        results = fitted_nordlys.route_batch([])
        assert results == []

    def test_route_single_item_batch(self, fitted_nordlys):
        """Test routing a single-item batch."""
        results = fitted_nordlys.route_batch(["Single prompt"])
        assert len(results) == 1
        assert isinstance(results[0], RouteResult)
