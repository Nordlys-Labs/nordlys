#!/usr/bin/env python3
"""Prepare batch JSONL for Doubleword API submission.

Creates a batch input file combining SWE-smith problem statements
with optional pre-fetched file contents for one-shot patch generation.

Usage:
    # With file contents (if available)
    python -m swe_smith.scripts.prepare_batch \
        --file-contents results/gpt4o/file_contents.jsonl \
        --model "gpt-4o" \
        --output results/gpt4o/batch_input.jsonl

    # Without file contents (problem statement only)
    python -m swe_smith.scripts.prepare_batch \
        --sampled-ids results/gpt4o/sampled_ids.json \
        --model "gpt-4o" \
        --output results/gpt4o/batch_input.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from swe_smith.batch_runner import DoublewordBatchRunner
from swe_smith.config import (
    RESULTS_DIR,
    SWE_SMITH_DATASET,
    SWE_SMITH_SPLIT,
)


def load_file_data_safe(path: Path) -> dict[str, dict]:
    """Load pre-fetched file data if available.

    Args:
        path: Path to the JSONL file.

    Returns:
        Dict mapping instance_id to file data, or empty dict if not found.
    """
    if not path.exists():
        return {}

    file_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                file_map[data["instance_id"]] = data
    return file_map


def load_swe_smith_instances(instance_ids: set[str]) -> dict[str, dict]:
    """Load SWE-smith instances for given IDs.

    Args:
        instance_ids: Set of instance IDs to load.

    Returns:
        Dict mapping instance_id to instance dict.
    """
    print(f"Loading SWE-smith dataset from {SWE_SMITH_DATASET}...")
    ds = load_dataset(SWE_SMITH_DATASET, split=SWE_SMITH_SPLIT)
    print(f"Loaded {len(ds)} total instances")

    # Filter to sampled IDs
    instances = {}
    for item in ds:
        if item["instance_id"] in instance_ids:
            instances[item["instance_id"]] = item

    print(f"Found {len(instances)} instances matching sampled IDs")
    return instances


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare batch JSONL for Doubleword submission"
    )

    parser.add_argument(
        "--file-contents",
        type=Path,
        help="Path to pre-fetched file contents JSONL (optional)",
    )
    parser.add_argument(
        "--sampled-ids",
        type=Path,
        default=RESULTS_DIR / "gpt4o" / "sampled_ids.json",
        help="Path to sampled IDs JSON",
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
        default=RESULTS_DIR / "gpt4o" / "batch_input.jsonl",
        help="Output path for batch JSONL file",
    )

    args = parser.parse_args()

    # Load file contents if available
    file_data_map = {}
    if args.file_contents and args.file_contents.exists():
        print(f"Loading file contents from: {args.file_contents}")
        file_data_map = load_file_data_safe(args.file_contents)
        print(f"Loaded file data for {len(file_data_map)} instances")
    else:
        print("No file contents provided - using problem statements only")

    # Load sampled IDs
    if not args.sampled_ids.exists():
        print(f"Error: Sampled IDs file not found: {args.sampled_ids}")
        return

    with open(args.sampled_ids) as f:
        sampled_data = json.load(f)
    instance_ids = set(sampled_data["instance_ids"])
    print(f"Using {len(instance_ids)} instance IDs from: {args.sampled_ids}")

    # Load SWE-smith instances for problem statements
    instances_map = load_swe_smith_instances(instance_ids)

    # Convert to list
    instances = list(instances_map.values())

    # Check for missing instances
    missing = instance_ids - set(instances_map.keys())
    if missing:
        print(f"Warning: {len(missing)} instances not found in dataset")

    if file_data_map:
        missing_files = set(instances_map.keys()) - set(file_data_map.keys())
        if missing_files:
            print(f"Note: {len(missing_files)} instances without file context")

    # Create batch runner (doesn't need API key for JSONL creation)
    runner = DoublewordBatchRunner(
        api_key="",  # Not needed for create_batch_jsonl
        model=args.model,
    )

    # Create batch JSONL
    print("\nCreating batch JSONL...")
    print(f"Model: {args.model}")
    print(f"Instances: {len(instances)}")
    print(f"With file context: {len(file_data_map)}")
    print(f"Output: {args.output}")

    count = runner.create_batch_jsonl(
        instances=instances,
        file_data_map=file_data_map,
        output_path=args.output,
    )

    # Verify file
    file_size = args.output.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print("Batch preparation complete!")
    print(f"{'='*60}")
    print(f"Requests written: {count}")
    print(f"Output file: {args.output}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Print sample request
    print("\nSample request (first instance):")
    with open(args.output) as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"  custom_id: {sample['custom_id']}")
        print(f"  model: {sample['body']['model']}")
        print(f"  messages: {len(sample['body']['messages'])} messages")
        user_msg = sample["body"]["messages"][1]["content"]
        print(f"  user message length: {len(user_msg)} chars")


if __name__ == "__main__":
    main()
