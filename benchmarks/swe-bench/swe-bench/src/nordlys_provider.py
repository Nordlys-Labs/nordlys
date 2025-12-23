"""Custom LiteLLM provider for Nordlys Router (for direct Python usage).

NOTE: This custom provider is NOT needed for mini-swe-agent!
      Use `anthropic/nordlys/nordlys-code` with mini-swe-agent instead.

This module provides two ways to use Nordlys with LiteLLM:

1. For mini-swe-agent (recommended):
   - Use model: `anthropic/nordlys/nordlys-code`
   - Configure: mini-extra config set ANTHROPIC_API_KEY "$NORDLYS_API_KEY"
   - Configure: mini-extra config set ANTHROPIC_API_BASE "$NORDLYS_API_BASE"

2. For direct LiteLLM usage (this module):
   - Register provider: register_nordlys_provider()
   - Use model: `nordlys/anything` (sends empty model to API)
   - Useful when you need empty model name for intelligent routing
"""

import os
from typing import Any

import litellm
from anthropic import Anthropic
from litellm import CustomLLM


class NordlysProvider(CustomLLM):
    """Custom LiteLLM provider that wraps Nordlys Anthropic-compatible API.

    This provider sends empty model name to Nordlys for intelligent routing.
    Use this when you need direct control over the model name sent to API.
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
            model: Model identifier (e.g., 'nordlys/anything' - ignored, sends empty)
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (max_tokens, temperature, etc.)

        Returns:
            LiteLLM ModelResponse with the completion result
        """
        # Separate system message from user/assistant messages
        system_msg: str | None = None
        user_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        # Call Nordlys API via Anthropic client
        # Always send empty model for Nordlys intelligent routing
        response = self.client.messages.create(
            model="",  # Empty for Nordlys intelligent routing
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
    with LiteLLM in your Python code (not for mini-swe-agent).

    Example:
        from nordlys_provider import register_nordlys_provider
        import litellm

        register_nordlys_provider()
        response = litellm.completion(
            model="nordlys/code",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    nordlys = NordlysProvider()
    litellm.custom_provider_map = [{"provider": "nordlys", "custom_handler": nordlys}]
