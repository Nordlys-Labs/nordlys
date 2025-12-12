"""Router implementations for benchmarking."""

import json
import time
from pathlib import Path
from typing import Any

import anthropic
from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import ModelSelectionRequest


class CactusProfileRouter:
    """Router using trained Cactus profile with adaptive_router package."""

    def __init__(self, profile_path: str | Path):
        """Initialize router with Cactus profile.

        Args:
            profile_path: Path to production_profile.json
        """
        self.profile_path = Path(profile_path)

        # Load profile to get model info
        with open(self.profile_path) as f:
            self.profile_data = json.load(f)

        # Use adaptive_router ModelRouter
        self.router = ModelRouter.from_json_file(self.profile_path)

        # Extract model metadata
        self.models = {
            model["model_id"]: model
            for model in self.profile_data["models"]
        }

    def route(
        self,
        prompt: str,
        cost_bias: float = 0.5
    ) -> dict[str, Any]:
        """Select model for given prompt.

        Args:
            prompt: User prompt to route
            cost_bias: 0.0 = prefer speed, 1.0 = prefer quality

        Returns:
            Dict with selected_model, cluster, routing_time_ms, metadata
        """
        start = time.perf_counter()

        # Create selection request
        request = ModelSelectionRequest(prompt=prompt)

        # Route using adaptive_router
        response = self.router.select_model(request, cost_bias=cost_bias)

        routing_time_ms = (time.perf_counter() - start) * 1000

        # Get model metadata
        model_meta = self.models.get(response.model_id, {})

        return {
            "selected_model": response.model_id,
            "cluster": None,  # Not exposed in API response
            "routing_time_ms": routing_time_ms,
            "model_size_mb": model_meta.get("size_mb", 0),
            "avg_tokens_per_sec": model_meta.get("avg_tokens_per_sec", 0),
            "error_rate": None,  # Will be simulated later
            "alternatives": [alt.model_id for alt in response.alternatives],
        }

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get metadata for a model."""
        return self.models.get(model_id, {})


class ClaudeOracleRouter:
    """Router using Claude Opus 4.5 as an oracle for routing decisions."""

    def __init__(
        self,
        api_key: str,
        available_models: list[dict[str, Any]]
    ):
        """Initialize Claude oracle router.

        Args:
            api_key: Anthropic API key
            available_models: List of Cactus models with metadata
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.models = {m["model_id"]: m for m in available_models}
        self.model_list_str = self._format_model_list()

    def _format_model_list(self) -> str:
        """Format model list for Claude's system prompt."""
        lines = []
        for model_id, meta in sorted(
            self.models.items(),
            key=lambda x: x[1].get("size_mb", 0)
        ):
            size_mb = meta.get("size_mb", 0)
            tokens_sec = meta.get("avg_tokens_per_sec", 0)
            caps = ", ".join(meta.get("capabilities", []))
            desc = meta.get("description", "")

            lines.append(
                f"- {model_id}: {size_mb}MB, "
                f"~{tokens_sec} tokens/sec, {caps or 'text'}"
            )
            if desc:
                lines.append(f"  ({desc})")

        return "\n".join(lines)

    def route(
        self,
        prompt: str,
        cost_bias: float = 0.5
    ) -> dict[str, Any]:
        """Ask Claude to select best model for prompt.

        Args:
            prompt: User prompt to route
            cost_bias: 0.0 = prefer speed, 1.0 = prefer quality

        Returns:
            Dict with selected_model, reasoning, routing_time_ms, metadata
        """
        import time

        # Interpret cost_bias
        if cost_bias <= 0.2:
            preference = "strongly prefer the FASTEST (smallest) model"
        elif cost_bias <= 0.4:
            preference = "prefer faster (smaller) models"
        elif cost_bias <= 0.6:
            preference = "balance speed and quality"
        elif cost_bias <= 0.8:
            preference = "prefer higher quality (larger) models"
        else:
            preference = "strongly prefer the HIGHEST QUALITY (largest) model"

        system_prompt = f"""You are an expert model router for on-device AI systems.

You must select the BEST model from this list of local Cactus models:

{self.model_list_str}

ROUTING STRATEGY (cost_bias={cost_bias:.2f}):
{preference}

IMPORTANT RULES:
1. Respond with ONLY the model_id (e.g., "gemma-270m")
2. NO explanations, NO markdown, NO extra text
3. Choose from the models listed above ONLY
4. Consider prompt complexity, required capabilities, and speed/quality tradeoff

USER PROMPT TO ROUTE:"""

        start = time.perf_counter()

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4
                max_tokens=50,
                temperature=0.0,  # Deterministic
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            routing_time_ms = (time.perf_counter() - start) * 1000

            # Extract model_id from response
            selected_model = response.content[0].text.strip()

            # Clean up response (remove markdown, quotes, etc.)
            selected_model = (
                selected_model
                .replace("`", "")
                .replace('"', "")
                .replace("'", "")
                .strip()
            )

            # Validate model exists
            if selected_model not in self.models:
                # Try to find partial match
                for model_id in self.models:
                    if model_id in selected_model:
                        selected_model = model_id
                        break
                else:
                    # Fallback to smallest model
                    selected_model = min(
                        self.models.keys(),
                        key=lambda m: self.models[m].get("size_mb", 0)
                    )

            # Get model metadata
            model_meta = self.models[selected_model]

            return {
                "selected_model": selected_model,
                "cluster": None,  # Claude doesn't use clusters
                "routing_time_ms": routing_time_ms,
                "model_size_mb": model_meta.get("size_mb", 0),
                "avg_tokens_per_sec": model_meta.get("avg_tokens_per_sec", 0),
                "error_rate": None,  # Will be simulated later
                "alternatives": [],
                "api_cost_usd": self._estimate_cost(response),
            }

        except Exception as e:
            # Fallback on error
            print(f"Claude API error: {e}, using fallback")
            fallback_model = "gemma-270m"  # Safest fallback
            return {
                "selected_model": fallback_model,
                "cluster": None,
                "routing_time_ms": 0,
                "model_size_mb": self.models[fallback_model].get("size_mb", 0),
                "avg_tokens_per_sec": self.models[fallback_model].get("avg_tokens_per_sec", 0),
                "error_rate": None,
                "alternatives": [],
                "error": str(e),
            }

    def _estimate_cost(self, response: Any) -> float:
        """Estimate API cost for routing call.

        Claude Opus 4.5 pricing (as of 2025):
        - Input: $15 per 1M tokens
        - Output: $75 per 1M tokens
        """
        if hasattr(response, "usage"):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            input_cost = (input_tokens / 1_000_000) * 15.0
            output_cost = (output_tokens / 1_000_000) * 75.0

            return input_cost + output_cost

        return 0.0

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get metadata for a model."""
        return self.models.get(model_id, {})
