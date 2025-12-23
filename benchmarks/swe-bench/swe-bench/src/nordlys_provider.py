"""Custom LiteLLM provider for Nordlys Router.

This module implements a custom LiteLLM provider that wraps the Nordlys
Anthropic-compatible API, allowing mini-swe-agent to use models with
the 'nordlys/' prefix.
"""

import os
from typing import Any

import litellm
from anthropic import Anthropic
from litellm import CustomLLM


class NordlysProvider(CustomLLM):
    """Custom LiteLLM provider that wraps Nordlys Anthropic-compatible API.

    This provider allows using Nordlys models through LiteLLM with the
    'nordlys/' prefix, e.g., 'nordlys/Nordlys-singularity'.
    """

    def __init__(self) -> None:
        """Initialize the Nordlys provider with Anthropic client."""
        api_key = os.environ.get("NORDLYS_API_KEY")
        if not api_key:
            raise ValueError(
                "NORDLYS_API_KEY environment variable is required. "
                "Set it in your .env file or export it."
            )

        self.client = Anthropic(
            api_key=api_key,
            base_url=os.environ.get("NORDLYS_API_BASE", "https://api.llmadaptive.uk"),
        )

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> litellm.ModelResponse:
        """Handle completion request via Nordlys API.

        Args:
            model: Model identifier (e.g., 'nordlys/Nordlys-singularity')
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            LiteLLM ModelResponse with the completion result
        """
        # Extract model name (e.g., 'nordlys/Nordlys-singularity' -> 'Nordlys-singularity')
        model_name = model.split("/", 1)[-1] if "/" in model else model

        # Separate system message from user/assistant messages
        system_msg: str | None = None
        user_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        # Call Nordlys API via Anthropic client
        response = self.client.messages.create(
            model=model_name,
            messages=user_messages,
            system=system_msg or "",
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0),
        )

        # Convert to LiteLLM ModelResponse format
        return litellm.ModelResponse(
            id=response.id,
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text,
                    },
                    "finish_reason": response.stop_reason,
                }
            ],
            model=model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )


def register_nordlys_provider() -> None:
    """Register Nordlys provider with LiteLLM.

    This function must be called before using 'nordlys/' prefixed models
    with mini-swe-agent or any LiteLLM-based application.
    """
    nordlys = NordlysProvider()
    litellm.custom_provider_map = [{"provider": "nordlys", "custom_handler": nordlys}]
