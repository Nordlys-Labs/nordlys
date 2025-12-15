"""Unit tests for ModelRouter service."""

from unittest.mock import Mock

import pytest

from adaptive_router.models.api import ModelSelectionRequest
from adaptive_router.models.api import Model
from adaptive_router.core.router import ModelRouter


@pytest.fixture
def mock_router():
    """Create a mock ModelRouter with mocked C++ core and embedding model."""
    # Mock the C++ CoreRouter
    mock_core_router = Mock()
    mock_route_response = Mock()
    mock_route_response.selected_model = "openai/gpt-4"
    mock_route_response.alternatives = [
        "anthropic/claude-3-sonnet-20240229",
        "openai/gpt-3.5-turbo",
    ]
    mock_route_response.cluster_id = 5
    mock_route_response.cluster_distance = 0.15
    mock_core_router.route.return_value = mock_route_response
    mock_core_router.get_supported_models.return_value = [
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-sonnet-20240229",
    ]
    mock_core_router.get_n_clusters.return_value = 10
    mock_core_router.get_embedding_dim.return_value = 384

    # Mock the embedding model
    mock_embedding_model = Mock()
    import numpy as np

    mock_embedding_model.encode.return_value = np.zeros(384)  # Return fake embedding

    # Mock profile metadata
    mock_metadata = Mock()
    mock_metadata.routing = Mock()
    mock_metadata.routing.default_cost_preference = 0.5
    mock_metadata.embedding_model = "all-MiniLM-L6-v2"

    # Create router with mocked components
    router = ModelRouter(
        core_router=mock_core_router,
        embedding_model=mock_embedding_model,
        profile_metadata=mock_metadata,
    )

    return router


class TestModelRouter:
    """Test ModelRouter class logic without external dependencies."""

    def test_initialization(self, mock_router: ModelRouter) -> None:
        """Test router initialization creates a functional instance."""
        # Test that the router can perform its main function
        request = ModelSelectionRequest(
            prompt="Write a simple hello world function",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Verify the router produces valid output
        assert response.model_id
        assert response.model_id
        assert isinstance(response.alternatives, list)

    def test_initialization_without_params(self, mock_router: ModelRouter) -> None:
        """Test router works with default config."""
        # Test that the router works
        request = ModelSelectionRequest(
            prompt="Calculate the factorial of 10",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Verify valid response
        assert response.model_id
        assert response.model_id

    def test_select_model_with_full_models(self, mock_router: ModelRouter) -> None:
        """Test model selection when full models are provided."""
        sample_models = [
            Model(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=27.0,
            ),
        ]

        request = ModelSelectionRequest(
            prompt="Write a Python function to implement quicksort",
            models=sample_models,
            cost_bias=0.9,
        )
        response = mock_router.select_model(request)

        # Verify response structure
        assert response.model_id
        assert response.model_id
        assert isinstance(response.alternatives, list)

    def test_select_model_cost_bias_low(self, mock_router: ModelRouter) -> None:
        """Test that low cost bias works correctly."""
        # Low cost bias (0.1)
        request = ModelSelectionRequest(
            prompt="Write a simple hello world program",
            cost_bias=0.1,
        )
        response = mock_router.select_model(request)

        assert response.model_id
        assert response.model_id

    def test_select_model_cost_bias_high(self, mock_router: ModelRouter) -> None:
        """Test that high cost bias works correctly."""
        # High cost bias (0.9)
        request = ModelSelectionRequest(
            prompt="Design a distributed system architecture for real-time data processing",
            cost_bias=0.9,
        )
        response = mock_router.select_model(request)

        assert response.model_id
        assert response.model_id

    def test_select_model_empty_input(self, mock_router: ModelRouter) -> None:
        """Test selecting models when no models are provided."""
        request = ModelSelectionRequest(
            prompt="Explain quantum computing",
            models=None,
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Should work with internal model list
        assert response.model_id
        assert response.model_id
        assert isinstance(response.alternatives, list)

    def test_partial_model_filtering(self, mock_router: ModelRouter) -> None:
        """Test filtering with partial Model (minimal fields)."""
        partial_models = [
            Model(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=27.0,
            )
        ]

        request = ModelSelectionRequest(
            prompt="Generate a creative story",
            models=partial_models,
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Verify that model filter was passed to C++ core
        mock_router._core_router.route.assert_called_once()
        call_args = mock_router._core_router.route.call_args
        # Check positional arguments: (embedding, cost_bias, models)
        args = call_args[0]
        assert len(args) >= 3  # embedding, cost_bias, models
        model_filter = args[2]
        assert isinstance(model_filter, list)
        assert "openai/gpt-4" in model_filter

        # Should work with filtered models
        assert response.model_id
        assert response.model_id

    def test_model_selection_code_task(self, mock_router: ModelRouter) -> None:
        """Test model selection for code generation tasks."""
        request = ModelSelectionRequest(
            prompt="Write a Python function to implement binary search",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        assert response.model_id
        assert response.model_id

    def test_model_selection_creative_task(self, mock_router: ModelRouter) -> None:
        """Test model selection for creative writing tasks."""
        request = ModelSelectionRequest(
            prompt="Write a short poem about nature",
            cost_bias=0.3,
        )
        response = mock_router.select_model(request)

        assert response.model_id
        assert response.model_id


class TestModelRouterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_cost_bias_raises_error(self) -> None:
        """Test that invalid cost bias values raise validation errors."""

        models = [
            Model(
                provider="openai",
                model_name="gpt-5",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=27.0,
            )
        ]

        # Test cost_bias > 1.0 raises ValidationError
        with pytest.raises(Exception) as exc_info:
            ModelSelectionRequest(
                prompt="Simple task",
                models=models,
                cost_bias=2.0,
            )
        assert (
            "cost_bias" in str(exc_info.value).lower()
            or "validation" in str(exc_info.value).lower()
        )

        # Test cost_bias < 0.0 raises ValidationError
        with pytest.raises(Exception) as exc_info:
            ModelSelectionRequest(
                prompt="Simple task",
                models=models,
                cost_bias=-1.0,
            )
        assert (
            "cost_bias" in str(exc_info.value).lower()
            or "validation" in str(exc_info.value).lower()
        )

    def test_valid_cost_bias_boundary_values(self, mock_router: ModelRouter) -> None:
        """Test that boundary values 0.0 and 1.0 are accepted."""
        # Test cost_bias = 0.0 (minimum)
        request_min = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=0.0,
        )
        response_min = mock_router.select_model(request_min)
        assert response_min.model_id
        assert response_min.model_id

        # Test cost_bias = 1.0 (maximum)
        request_max = ModelSelectionRequest(
            prompt="Simple task",
            cost_bias=1.0,
        )
        response_max = mock_router.select_model(request_max)
        assert response_max.model_id
        assert response_max.model_id

    def test_complex_prompt_handling(self, mock_router: ModelRouter) -> None:
        """Test handling of very complex prompts."""
        # Very long and complex prompt
        complex_prompt = """
        Design and implement a distributed microservices architecture with the following requirements:
        1. Real-time data processing with sub-second latency
        2. Horizontal scalability to handle 1M+ requests per second
        3. Fault tolerance with automatic failover
        4. Multi-region deployment with active-active replication
        5. End-to-end encryption and compliance with GDPR
        Include implementation details, technology stack recommendations, and deployment strategies.
        """

        request = ModelSelectionRequest(
            prompt=complex_prompt,
            cost_bias=0.9,
        )
        response = mock_router.select_model(request)

        assert response.model_id
        assert response.model_id

    def test_simple_prompt_handling(self, mock_router: ModelRouter) -> None:
        """Test handling of very simple prompts."""
        request = ModelSelectionRequest(
            prompt="Hello, how are you?",
            cost_bias=0.1,
        )
        response = mock_router.select_model(request)

        # Should successfully select a model
        assert response.model_id
        assert response.model_id
        assert isinstance(response.alternatives, list)

    def test_alternatives_generation(self, mock_router: ModelRouter) -> None:
        """Test that alternatives are properly generated."""
        request = ModelSelectionRequest(
            prompt="Write a complex algorithm",
            cost_bias=0.5,
        )
        response = mock_router.select_model(request)

        # Should successfully select a model
        assert response.model_id
        assert response.model_id
        # Should have alternatives
        assert isinstance(response.alternatives, list)

    def test_no_models_raises_error(self, mock_router: ModelRouter) -> None:
        """Test that providing empty models list is handled."""
        # Empty models list should be handled by service
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=[],
            cost_bias=0.5,
        )

        # Should not raise error
        response = mock_router.select_model(request)
        assert response.model_id
        assert response.model_id

        # Empty list should be passed as empty filter (no filtering)
        mock_router._core_router.route.assert_called_once()
        call_args = mock_router._core_router.route.call_args
        args = call_args[0]
        assert len(args) >= 3
        model_filter = args[2]
        assert model_filter == []


class TestModelFiltering:
    """Test model filtering functionality."""

    @pytest.fixture
    def filtering_mock_router(self):
        """Create a mock router that tracks filtering calls."""
        from unittest.mock import Mock
        import numpy as np

        mock_core_router = Mock()
        mock_route_response = Mock()
        mock_route_response.selected_model = "openai/gpt-4"
        mock_route_response.alternatives = ["anthropic/claude-3-sonnet-20240229"]
        mock_route_response.cluster_id = 5
        mock_route_response.cluster_distance = 0.15

        # Make route() return different responses based on filter
        def route_side_effect(embedding, cost_bias, models=None):
            response = Mock()
            if models and len(models) > 0:
                # If filtering, return first filtered model
                response.selected_model = models[0]
                response.alternatives = models[1:] if len(models) > 1 else []
            else:
                # No filter - return default
                response.selected_model = "openai/gpt-4"
                response.alternatives = ["anthropic/claude-3-sonnet-20240229"]
            response.cluster_id = 5
            response.cluster_distance = 0.15
            return response

        mock_core_router.route.side_effect = route_side_effect
        mock_core_router.get_supported_models.return_value = [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet-20240229",
        ]
        mock_core_router.get_n_clusters.return_value = 10
        mock_core_router.get_embedding_dim.return_value = 384

        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = np.zeros(384)

        mock_metadata = Mock()
        mock_metadata.routing = Mock()
        mock_metadata.routing.default_cost_preference = 0.5
        mock_metadata.embedding_model = "all-MiniLM-L6-v2"

        router = ModelRouter(
            core_router=mock_core_router,
            embedding_model=mock_embedding_model,
            profile_metadata=mock_metadata,
        )

        return router

    def test_filtering_with_single_model(
        self, filtering_mock_router: ModelRouter
    ) -> None:
        """Test filtering with single model - should only return that model."""
        single_model = [
            Model(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=27.0,
            )
        ]

        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=single_model,
            cost_bias=0.5,
        )
        response = filtering_mock_router.select_model(request)

        # Verify filter was passed correctly
        call_args = filtering_mock_router._core_router.route.call_args
        args = call_args[0]
        assert len(args) >= 3
        model_filter = args[2]
        assert model_filter == ["openai/gpt-4"]
        assert response.model_id == "openai/gpt-4"

    def test_filtering_with_multiple_models(
        self, filtering_mock_router: ModelRouter
    ) -> None:
        """Test filtering with multiple models - should only return from filtered set."""
        multiple_models = [
            Model(
                provider="openai",
                model_name="gpt-4",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=27.0,
            ),
            Model(
                provider="anthropic",
                model_name="claude-3-sonnet-20240229",
                cost_per_1m_input_tokens=3.0,
                cost_per_1m_output_tokens=15.0,
            ),
        ]

        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=multiple_models,
            cost_bias=0.5,
        )
        response = filtering_mock_router.select_model(request)

        # Verify filter contains both models
        call_args = filtering_mock_router._core_router.route.call_args
        args = call_args[0]
        assert len(args) >= 3
        model_filter = args[2]
        assert len(model_filter) == 2
        assert "openai/gpt-4" in model_filter
        assert "anthropic/claude-3-sonnet-20240229" in model_filter
        # Selected model should be from filtered set
        assert response.model_id in model_filter

    def test_filtering_with_empty_list(
        self, filtering_mock_router: ModelRouter
    ) -> None:
        """Test filtering with empty list - should use all models (no filter)."""
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=[],
            cost_bias=0.5,
        )
        response = filtering_mock_router.select_model(request)

        # Verify empty filter was passed (no filtering)
        call_args = filtering_mock_router._core_router.route.call_args
        args = call_args[0]
        assert len(args) >= 3
        model_filter = args[2]
        assert model_filter == []
        # Should still return a valid response
        assert response.model_id

    def test_filtering_with_none_models(
        self, filtering_mock_router: ModelRouter
    ) -> None:
        """Test filtering with None models - should use all models (no filter)."""
        request = ModelSelectionRequest(
            prompt="Test prompt",
            models=None,
            cost_bias=0.5,
        )
        response = filtering_mock_router.select_model(request)

        # Verify empty filter was passed (no filtering)
        call_args = filtering_mock_router._core_router.route.call_args
        args = call_args[0]
        assert len(args) >= 3
        model_filter = args[2]
        assert model_filter == []
        # Should still return a valid response
        assert response.model_id

    def test_model_unique_id_format(self) -> None:
        """Test that Model.unique_id() produces correct format for filtering."""
        model = Model(
            provider="openai",
            model_name="gpt-4",
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=27.0,
        )

        model_id = model.unique_id()
        assert model_id == "openai/gpt-4"
        assert "/" in model_id  # Should have provider/model format

    def test_model_unique_id_case_insensitive(self) -> None:
        """Test that Model.unique_id() normalizes case correctly."""
        model_upper = Model(
            provider="OpenAI",
            model_name="GPT-4",
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=27.0,
        )

        model_id = model_upper.unique_id()
        assert model_id == "openai/gpt-4"  # Should be lowercase
