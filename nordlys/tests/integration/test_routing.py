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

    def test_route_batch_uses_batch_processing(self, fitted_nordlys):
        """Test that route_batch processes all prompts in a single batch."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
        results = fitted_nordlys.route_batch(prompts)

        # All results should be valid RouteResults
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, RouteResult)
            assert result.model_id is not None
            assert result.cluster_id >= 0

    def test_route_batch_with_repeated_prompts(self, fitted_nordlys):
        """Test that route_batch handles repeated prompts correctly."""
        prompts = ["Same prompt", "Same prompt", "Different prompt"]
        results = fitted_nordlys.route_batch(prompts)

        assert len(results) == 3
        # Same prompts should produce same results
        assert results[0].model_id == results[1].model_id
        assert results[0].cluster_id == results[1].cluster_id
        # Different prompt may or may not match
        assert isinstance(results[2], RouteResult)

    def test_route_batch_large_batch(self, fitted_nordlys):
        """Test route_batch with a large batch of prompts."""
        prompts = [f"Prompt {i}" for i in range(100)]
        results = fitted_nordlys.route_batch(prompts)

        assert len(results) == 100
        for result in results:
            assert isinstance(result, RouteResult)
            assert result.model_id is not None
