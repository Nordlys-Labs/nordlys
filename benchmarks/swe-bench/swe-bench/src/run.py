#!/usr/bin/env python3
"""Run SWE-bench with Nordlys Router via mini-swe-agent.

Uses LiteLLM's anthropic/ provider with Nordlys API endpoint for intelligent routing.

Requirements:
    - ANTHROPIC_API_KEY and ANTHROPIC_API_BASE must be set in mini-swe-agent config:
        mini-extra config set ANTHROPIC_API_KEY "$NORDLYS_API_KEY"
        mini-extra config set ANTHROPIC_API_BASE "$NORDLYS_API_BASE"
    - MSWEA_COST_TRACKING='ignore_errors' (for non-standard model names)

Usage:
    # From benchmarks/ directory:
    uv run python swe-bench/swe-bench/src/run.py

    # Quick test (1 instance)
    uv run python swe-bench/swe-bench/src/run.py --slice :1

    # Custom configuration
    uv run python swe-bench/swe-bench/src/run.py --workers 8 --output results/my-run
"""

import os
import subprocess
import sys


def main() -> int:
    """Run SWE-bench benchmark with Nordlys Router via Anthropic endpoint.

    Returns:
        Exit code from mini-swe-agent command
    """
    # Ensure cost tracking is set to ignore errors for custom model names
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

    # Default arguments if none provided
    # Model format: anthropic/nordlys/nordlys-code
    # - LiteLLM uses 'anthropic/' as the provider
    # - Strips prefix and sends 'nordlys/nordlys-code' to API
    # - Nordlys accepts nordlys/* models for intelligent routing
    args = sys.argv[1:] or [
        "--model",
        "anthropic/nordlys/nordlys-code",
        "--subset",
        "verified",
        "--split",
        "test",
        "--workers",
        "4",
        "--output",
        "results/nordlys-run",
    ]

    # Ensure model is specified if not in args
    if "--model" not in args:
        args = ["--model", "anthropic/nordlys/nordlys-code"] + args

    # Run mini-extra swebench command
    cmd = ["mini-extra", "swebench"] + args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
