"""Tests for RouteResult dataclass."""

from nordlys import RouteResult


class TestRouteResult:
    """Test RouteResult dataclass."""

    def test_create_route_result(self):
        """Test creating a RouteResult instance."""
        result = RouteResult(
            model_id="openai/gpt-4",
            cluster_id=5,
            cluster_distance=0.15,
            alternatives=["anthropic/claude-3"],
        )
        assert result.model_id == "openai/gpt-4"
        assert result.cluster_id == 5
        assert result.cluster_distance == 0.15
        assert len(result.alternatives) == 1

    def test_route_result_default_alternatives(self):
        """Test RouteResult with default empty alternatives."""
        result = RouteResult(
            model_id="openai/gpt-4",
            cluster_id=5,
            cluster_distance=0.15,
        )
        assert result.alternatives == []

    def test_route_result_with_multiple_alternatives(self):
        """Test RouteResult with multiple alternatives."""
        alternatives = ["model1", "model2", "model3"]
        result = RouteResult(
            model_id="openai/gpt-4",
            cluster_id=2,
            cluster_distance=0.1,
            alternatives=alternatives,
        )
        assert len(result.alternatives) == 3
        assert result.alternatives[0] == "model1"
