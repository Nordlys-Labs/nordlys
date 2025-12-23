#!/usr/bin/env python3
"""Test script for Nordlys API with Anthropic-compatible endpoint."""

import os
from dotenv import load_dotenv

load_dotenv()

NORDLYS_API_KEY = os.getenv("NORDLYS_API_KEY")
NORDLYS_API_BASE = os.getenv("NORDLYS_API_BASE", "https://api.llmadaptive.uk")

print(f"API Key: {NORDLYS_API_KEY[:20]}..." if NORDLYS_API_KEY else "API Key: NOT SET")
print(f"API Base: {NORDLYS_API_BASE}")
print()

# Test: Using Anthropic client with Nordlys endpoint
print("=" * 60)
print("Test: Using Anthropic client with Nordlys endpoint")
print("=" * 60)

try:
    from anthropic import Anthropic

    # Remove trailing slash if present
    base = NORDLYS_API_BASE.rstrip("/")

    client = Anthropic(
        api_key=NORDLYS_API_KEY,
        base_url=base,
    )
    print(f"Using base URL: {base}")

    print("\nTrying with empty model name (Nordlys handles routing)")
    try:
        response = client.messages.create(
            model="",  # Empty - Nordlys handles routing
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello"}],
        )
        print(f"  SUCCESS: {response.content[0].text[:50]}...")
        print(f"  Model used: {response.model}")
    except Exception as e:
        print(f"  ERROR: {e}")

except Exception as e:
    print(f"Anthropic client error: {e}")
