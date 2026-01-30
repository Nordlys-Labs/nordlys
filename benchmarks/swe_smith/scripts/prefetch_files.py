#!/usr/bin/env python3
"""Pre-fetch file contents for sampled SWE-smith instances.

Uses Docker with Rosetta emulation on macOS to fetch relevant source files
from SWE-smith containers. The fetched files provide context for one-shot
patch generation.

Prerequisites:
    - Docker Desktop with Rosetta enabled (Settings → General →
      "Use Rosetta for x86_64/amd64 emulation on Apple Silicon")
    - SWE-smith containers available (will be pulled automatically)

Usage:
    python -m benchmarks.swe_smith.scripts.prefetch_files \
        --input results/gpt4o/sampled_ids.json \
        --output results/gpt4o/file_contents.jsonl \
        --workers 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datasets import load_dataset

from swe_smith.config import (
    DEFAULT_WORKERS,
    RESULTS_DIR,
    SWE_SMITH_DATASET,
    SWE_SMITH_SPLIT,
)
from swe_smith.file_fetcher import FileFetcher


def load_sampled_ids(path: Path) -> list[str]:
    """Load sampled instance IDs from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["instance_ids"]


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


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    # Load sampled instance IDs
    print(f"Loading sampled IDs from: {args.input}")
    sampled_ids = load_sampled_ids(args.input)
    print(f"Loaded {len(sampled_ids)} sampled instance IDs")

    # Load SWE-smith instances
    instances_map = load_swe_smith_instances(set(sampled_ids))

    # Convert to list in same order as sampled_ids
    instances = []
    missing = []
    for inst_id in sampled_ids:
        if inst_id in instances_map:
            instances.append(instances_map[inst_id])
        else:
            missing.append(inst_id)

    if missing:
        print(f"Warning: {len(missing)} sampled IDs not found in dataset")
        print(f"  First few missing: {missing[:5]}")

    # Initialize fetcher
    fetcher = FileFetcher(
        startup_timeout=args.timeout,
        command_timeout=30,
        use_rosetta=not args.no_rosetta,
    )

    # Fetch files
    print(f"\n{'='*60}")
    print("Pre-fetching files from SWE-smith containers")
    print(f"{'='*60}")
    print(f"Instances: {len(instances)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print(f"Rosetta: {'disabled' if args.no_rosetta else 'enabled'}")

    results = await fetcher.fetch_batch(
        instances=instances,
        output_path=args.output,
        workers=args.workers,
        resume=not args.no_resume,
    )

    # Summary
    success_count = sum(1 for r in results if "error" not in r)
    error_count = sum(1 for r in results if "error" in r)

    print(f"\n{'='*60}")
    print("Pre-fetch complete!")
    print(f"{'='*60}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output saved to: {args.output}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-fetch file contents from SWE-smith containers"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=RESULTS_DIR / "sampled_ids.json",
        help="Input path for sampled instance IDs JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=RESULTS_DIR / "file_contents.jsonl",
        help="Output path for file contents JSONL",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Container startup timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--no-rosetta",
        action="store_true",
        help="Disable Rosetta emulation (use native platform)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing output, start fresh",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
