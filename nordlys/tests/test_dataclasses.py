"""Tests for Alternative and RouteResult dataclasses."""

from nordlys import Alternative, RouteResult


class TestAlternative:
    """Test Alternative dataclass."""

    def test_create_alternative(self):
        """Test creating an Alternative instance."""
        alt = Alternative(model_id="openai/gpt-4", score=0.85)
        assert alt.model_id == "openai/gpt-4"
        assert alt.score == 0.85

    def test_alternative_comparison(self):
        """Test Alternative comparison."""
        alt1 = Alternative(model_id="model1", score=0.9)
        alt2 = Alternative(model_id="model2", score=0.8)
        assert alt1.score > alt2.score


class TestRouteResult:
    """Test RouteResult dataclass."""

    def test_create_route_result(self):
        """Test creating a RouteResult instance."""
        result = RouteResult(
            model_id="openai/gpt-4",
            cluster_id=5,
            cluster_distance=0.15,
            alternatives=[
                Alternative(model_id="anthropic/claude-3", score=0.85),
            ],
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
        alternatives = [
            Alternative(model_id="model1", score=0.9),
            Alternative(model_id="model2", score=0.8),
            Alternative(model_id="model3", score=0.7),
        ]
        result = RouteResult(
            model_id="openai/gpt-4",
            cluster_id=2,
            cluster_distance=0.1,
            alternatives=alternatives,
        )
        assert len(result.alternatives) == 3
        assert result.alternatives[0].score == 0.9
