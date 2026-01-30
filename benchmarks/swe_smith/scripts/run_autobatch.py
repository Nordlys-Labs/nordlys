#!/usr/bin/env python3
"""Run SWE-smith evaluation using the Doubleword autobatcher SDK.

This script replaces the manual batch workflow (prepare -> submit -> poll -> collect)
with a single async pipeline. The autobatcher SDK automatically groups requests
into OpenAI-compatible batches, uploads them, polls for completion, and resolves
each caller's asyncio.Future with a standard ChatCompletion response.

Usage:
    # Full run
    uv run python -m swe_smith.scripts.run_autobatch \
        --sampled-ids results/gpt4o/sampled_ids.json \
        --model gpt-4o \
        --output results/gpt4o/predictions.jsonl

    # Test with small batch
    uv run python -m swe_smith.scripts.run_autobatch \
        --sampled-ids results/gpt4o/sampled_ids.json \
        --model gpt-4o \
        --output results/gpt4o/predictions_test.jsonl \
        --limit 10

    # With pre-fetched file contents
    uv run python -m swe_smith.scripts.run_autobatch \
        --sampled-ids results/gpt4o/sampled_ids.json \
        --file-contents results/gpt4o/file_contents.jsonl \
        --model gpt-4o \
        --output results/gpt4o/predictions.jsonl

Environment:
    DOUBLEWORD_API_KEY: API key for Doubleword (required)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from swe_smith.batch_runner import AutobatchRunner
from swe_smith.config import (
    DOUBLEWORD_API_BASE,
    DOUBLEWORD_COMPLETION_WINDOW,
    RESULTS_DIR,
    SWE_SMITH_DATASET,
    SWE_SMITH_SPLIT,
)


def get_api_key() -> str:
    """Get Doubleword API key from environment."""
    key = os.environ.get("DOUBLEWORD_API_KEY")
    if not key:
        print("Error: DOUBLEWORD_API_KEY environment variable not set")
        print("Set it with: export DOUBLEWORD_API_KEY='your-key'")
        sys.exit(1)
    return key


def load_sampled_ids(path: Path) -> list[str]:
    """Load sampled instance IDs from JSON file.

    Args:
        path: Path to the sampled IDs JSON file.

    Returns:
        List of instance ID strings.
    """
    with open(path) as f:
        data = json.load(f)
    return data["instance_ids"]


def load_file_data(path: Path) -> dict[str, dict]:
    """Load pre-fetched file data from JSONL.

    Args:
        path: Path to the file contents JSONL.

    Returns:
        Dict mapping instance_id to file data dict.
    """
    file_map: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                file_map[data["instance_id"]] = data
    return file_map


def load_swe_smith_instances(instance_ids: set[str]) -> list[dict]:
    """Load SWE-smith instances from HuggingFace for the given IDs.

    Args:
        instance_ids: Set of instance IDs to load.

    Returns:
        List of instance dicts from the dataset.
    """
    from datasets import load_dataset

    print(f"Loading SWE-smith dataset from {SWE_SMITH_DATASET}...")
    ds = load_dataset(SWE_SMITH_DATASET, split=SWE_SMITH_SPLIT)
    print(f"Loaded {len(ds)} total instances")

    instances = []
    for item in ds:
        if item["instance_id"] in instance_ids:
            instances.append(item)

    print(f"Found {len(instances)} instances matching sampled IDs")

    missing = instance_ids - {i["instance_id"] for i in instances}
    if missing:
        print(f"Warning: {len(missing)} sampled IDs not found in dataset")

    return instances


async def run(args: argparse.Namespace) -> None:
    """Main async entry point."""
    api_key = get_api_key()

    # Load sampled IDs
    print(f"Loading sampled IDs from: {args.sampled_ids}")
    instance_ids = load_sampled_ids(args.sampled_ids)
    print(f"Total sampled IDs: {len(instance_ids)}")

    # Apply limit if specified
    if args.limit:
        instance_ids = instance_ids[: args.limit]
        print(f"Limited to {args.limit} instances")

    # Load SWE-smith instances from HuggingFace
    instances = load_swe_smith_instances(set(instance_ids))

    # Load file contents if provided
    file_data_map: dict[str, dict] = {}
    if args.file_contents and args.file_contents.exists():
        print(f"Loading file contents from: {args.file_contents}")
        file_data_map = load_file_data(args.file_contents)
        print(f"Loaded file data for {len(file_data_map)} instances")

    # Create autobatch runner
    runner = AutobatchRunner(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        completion_window=args.completion_window,
    )

    print(f"\n{'='*60}")
    print("Starting autobatch inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Instances: {len(instances)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Completion window: {args.completion_window}")
    print(f"Output: {args.output}")
    print()

    try:
        predictions = await runner.generate_all_patches(
            instances=instances,
            file_data_map=file_data_map,
            output_path=args.output,
        )
    finally:
        await runner.close()

    # Summary
    empty_patches = sum(1 for p in predictions if not p.get("model_patch"))
    non_empty = len(predictions) - empty_patches

    print(f"\n{'='*60}")
    print("Autobatch inference complete!")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"With patches: {non_empty}")
    print(f"Empty patches: {empty_patches}")
    print(f"Saved to: {args.output}")

    # Next steps
    print("\nNext steps:")
    print("1. Run swesmith evaluation:")
    print("   uv run python -m swesmith.harness.eval \\")
    print(f"       -p {args.output} \\")
    print(f'       --run_id "{args.model}-swe-smith-v1" \\')
    print("       -w 10")
    print("\n2. Build Nordlys profile from results:")
    print("   uv run python -m swe_smith.scripts.build_profile \\")
    print(f"       --report logs/run_evaluation/{args.model}-swe-smith-v1/report.json \\")
    print("       --cluster-map results/gpt4o/cluster_map.json \\")
    print(f'       --model "{args.model}"')


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SWE-smith evaluation using Doubleword autobatcher SDK"
    )

    parser.add_argument(
        "--sampled-ids",
        type=Path,
        default=RESULTS_DIR / "gpt4o" / "sampled_ids.json",
        help="Path to sampled IDs JSON",
    )
    parser.add_argument(
        "--file-contents",
        type=Path,
        help="Path to pre-fetched file contents JSONL (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for inference (default: gpt-4o)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=RESULTS_DIR / "gpt4o" / "predictions.jsonl",
        help="Output path for predictions JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of instances (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Autobatcher batch size (default: 100)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DOUBLEWORD_API_BASE,
        help=f"API base URL (default: {DOUBLEWORD_API_BASE})",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default=DOUBLEWORD_COMPLETION_WINDOW,
        choices=["24h", "1h"],
        help=f"Batch completion window (default: {DOUBLEWORD_COMPLETION_WINDOW})",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.sampled_ids.exists():
        print(f"Error: Sampled IDs file not found: {args.sampled_ids}")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
