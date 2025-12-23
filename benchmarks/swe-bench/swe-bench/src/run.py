#!/usr/bin/env python3
"""Run SWE-bench with Nordlys Router via mini-swe-agent.

This script registers the custom Nordlys LiteLLM provider and runs
mini-swe-agent's swebench command with the appropriate configuration.

Usage:
    # From benchmarks/ directory:
    uv run python swe-bench/swe-bench/src/run.py

    # Quick test (5 instances)
    uv run python swe-bench/swe-bench/src/run.py --slice :5

    # Custom configuration
    uv run python swe-bench/swe-bench/src/run.py --workers 8 --output results/my-run
"""

import subprocess
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nordlys_provider import register_nordlys_provider


def main() -> int:
    """Run SWE-bench benchmark with Nordlys Router.

    Returns:
        Exit code from mini-swe-agent command
    """
    # Register Nordlys provider before running mini-swe-agent
    register_nordlys_provider()

    # Default arguments if none provided
    args = sys.argv[1:] or [
        "--model",
        "nordlys/Nordlys-singularity",
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
        args = ["--model", "nordlys/Nordlys-singularity"] + args

    # Run mini-extra swebench command
    cmd = ["mini-extra", "swebench"] + args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
