"""Performance simulation for on-device inference."""

import random
from typing import Any


class PerformanceSimulator:
    """Simulates on-device inference performance based on model metadata."""

    def __init__(self, profile_data: dict[str, Any], seed: int = 42):
        """Initialize simulator with profile data.

        Args:
            profile_data: Loaded production_profile.json
            seed: Random seed for reproducibility
        """
        self.profile_data = profile_data
        self.models = {m["model_id"]: m for m in profile_data["models"]}
        self.cluster_centers = profile_data["cluster_centers"]
        self.rng = random.Random(seed)

    def simulate_inference(
        self,
        model_id: str,
        prompt: str,
        cluster_id: int | None = None
    ) -> dict[str, Any]:
        """Simulate inference metrics for a model.

        Args:
            model_id: Model identifier
            prompt: User prompt
            cluster_id: Cluster assignment (if available)

        Returns:
            Dict with simulated latency, tokens, memory, error_rate
        """
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Unknown model: {model_id}")

        # Estimate output tokens (simple heuristic)
        # Mobile prompts tend to have shorter responses
        prompt_words = len(prompt.split())
        if prompt_words < 10:  # Quick query
            output_tokens = self.rng.randint(10, 50)
        elif prompt_words < 30:  # Medium query
            output_tokens = self.rng.randint(30, 150)
        else:  # Complex query
            output_tokens = self.rng.randint(50, 300)

        # Get model's tokens/sec
        tokens_per_sec = model.get("avg_tokens_per_sec", 10)

        # Time to First Token (TTFT)
        # Smaller models: faster startup
        # Larger models: slower startup
        size_mb = model.get("size_mb", 500)
        base_ttft_ms = 20 + (size_mb / 50)  # ~20-50ms range
        ttft_ms = base_ttft_ms + self.rng.uniform(-5, 10)

        # Intertoken latency (ITL)
        itl_ms = (1000 / tokens_per_sec) if tokens_per_sec > 0 else 100

        # Total generation time
        generation_time_ms = ttft_ms + (output_tokens * itl_ms)

        # End-to-end latency (includes overhead)
        overhead_ms = self.rng.uniform(5, 15)
        e2e_latency_ms = generation_time_ms + overhead_ms

        # Get error rate from profile
        error_rate = self._get_error_rate(model_id, cluster_id)

        # Memory usage (model size + context)
        context_mb = self.rng.uniform(50, 150)  # Context/activations
        memory_mb = size_mb + context_mb

        return {
            "model_id": model_id,
            "ttft_ms": ttft_ms,
            "itl_ms": itl_ms,
            "generation_time_ms": generation_time_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "output_tokens": output_tokens,
            "throughput_tokens_per_sec": tokens_per_sec,
            "memory_mb": memory_mb,
            "error_rate": error_rate,
            "cluster_id": cluster_id,
        }

    def _get_error_rate(
        self,
        model_id: str,
        cluster_id: int | None
    ) -> float:
        """Get error rate from profile.

        Args:
            model_id: Model identifier
            cluster_id: Cluster assignment

        Returns:
            Error rate (0.0 to 1.0)
        """
        model = self.models.get(model_id)
        if not model:
            return 0.1  # Default 10% error

        error_rates = model.get("error_rates", [])

        if cluster_id is not None and 0 <= cluster_id < len(error_rates):
            # Use cluster-specific error rate
            return error_rates[cluster_id]
        elif error_rates:
            # Use average error rate
            return sum(error_rates) / len(error_rates)
        else:
            # Fallback: estimate from model size
            # Smaller models generally have higher error rates
            size_mb = model.get("size_mb", 500)
            # Simple heuristic: 15% for 270MB, 5% for 1.7GB
            return max(0.03, 0.20 - (size_mb / 10000))

    def compare_routes(
        self,
        prompt: str,
        profile_route: dict[str, Any],
        claude_route: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare performance of two routing decisions.

        Args:
            prompt: User prompt
            profile_route: Result from CactusProfileRouter
            claude_route: Result from ClaudeOracleRouter

        Returns:
            Dict comparing both routes
        """
        # Simulate inference for both choices
        profile_perf = self.simulate_inference(
            profile_route["selected_model"],
            prompt,
            profile_route.get("cluster")
        )

        claude_perf = self.simulate_inference(
            claude_route["selected_model"],
            prompt,
            claude_route.get("cluster")
        )

        # Calculate differences
        latency_diff_ms = (
            claude_perf["e2e_latency_ms"] - profile_perf["e2e_latency_ms"]
        )
        error_diff = (
            claude_perf["error_rate"] - profile_perf["error_rate"]
        )
        memory_diff_mb = (
            claude_perf["memory_mb"] - profile_perf["memory_mb"]
        )

        return {
            "prompt": prompt,
            "agreement": (
                profile_route["selected_model"] ==
                claude_route["selected_model"]
            ),
            "profile_router": {
                "model": profile_route["selected_model"],
                "routing_time_ms": profile_route["routing_time_ms"],
                **profile_perf
            },
            "claude_router": {
                "model": claude_route["selected_model"],
                "routing_time_ms": claude_route["routing_time_ms"],
                "api_cost_usd": claude_route.get("api_cost_usd", 0),
                **claude_perf
            },
            "comparison": {
                "latency_diff_ms": latency_diff_ms,
                "latency_diff_pct": (
                    (latency_diff_ms / profile_perf["e2e_latency_ms"]) * 100
                    if profile_perf["e2e_latency_ms"] > 0 else 0
                ),
                "error_diff": error_diff,
                "error_diff_pct": (
                    (error_diff / profile_perf["error_rate"]) * 100
                    if profile_perf["error_rate"] > 0 else 0
                ),
                "memory_diff_mb": memory_diff_mb,
                "claude_faster": latency_diff_ms < 0,
                "claude_more_accurate": error_diff < 0,
                "claude_lighter": memory_diff_mb < 0,
            }
        }
