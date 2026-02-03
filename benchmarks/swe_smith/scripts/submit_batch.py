#!/usr/bin/env python3
"""Submit batch JSONL to Doubleword API.

Uploads the batch input file and creates a batch job for processing.

Usage:
    # Submit a new batch
    python -m benchmarks.swe_smith.scripts.submit_batch \
        --input results/gpt4o/batch_input.jsonl

    # Check status of existing batch
    python -m benchmarks.swe_smith.scripts.submit_batch \
        --status batch_abc123

    # Wait for batch completion
    python -m benchmarks.swe_smith.scripts.submit_batch \
        --wait batch_abc123

Environment:
    DOUBLEWORD_API_KEY: API key for Doubleword (required)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from swe_smith.batch_runner import DoublewordBatchRunner
from swe_smith.config import (
    DOUBLEWORD_COMPLETION_WINDOW,
    RESULTS_DIR,
)


def get_api_key() -> str:
    """Get Doubleword API key from environment."""
    key = os.environ.get("DOUBLEWORD_API_KEY")
    if not key:
        print("Error: DOUBLEWORD_API_KEY environment variable not set")
        print("Set it with: export DOUBLEWORD_API_KEY='your-key'")
        sys.exit(1)
    return key


def submit_batch(args: argparse.Namespace, runner: DoublewordBatchRunner) -> None:
    """Submit a new batch job."""
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Submitting batch from: {args.input}")
    print(f"Model: {args.model}")
    print(f"Completion window: {args.completion_window}")

    batch_id = runner.submit_batch(args.input)

    print(f"\n{'='*60}")
    print("Batch submitted successfully!")
    print(f"{'='*60}")
    print(f"Batch ID: {batch_id}")
    print("\nTo check status:")
    print(f"  python -m benchmarks.swe_smith.scripts.submit_batch --status {batch_id}")
    print("\nTo wait for completion:")
    print(f"  python -m benchmarks.swe_smith.scripts.submit_batch --wait {batch_id}")
    print("\nTo download results when complete:")
    print(f"  python -m benchmarks.swe_smith.scripts.collect_results --batch-id {batch_id}")


def check_status(args: argparse.Namespace, runner: DoublewordBatchRunner) -> None:
    """Check batch status."""
    batch_id = args.status

    print(f"Checking status for batch: {batch_id}")

    status = runner.check_status(batch_id)

    print(f"\n{'='*60}")
    print(f"Batch Status: {status.status.upper()}")
    print(f"{'='*60}")
    print(f"ID: {status.id}")
    print(f"Total requests: {status.total_requests}")
    print(f"Completed: {status.completed_requests}")
    print(f"Failed: {status.failed_requests}")
    print(f"Progress: {status.progress_pct:.1f}%")

    if status.output_file_id:
        print(f"Output file: {status.output_file_id}")
    if status.error_file_id:
        print(f"Error file: {status.error_file_id}")

    if status.is_complete:
        print("\nBatch is complete! Download results with:")
        print(f"  python -m benchmarks.swe_smith.scripts.collect_results --batch-id {batch_id}")
    elif status.is_failed:
        print(f"\nBatch failed with status: {status.status}")
    else:
        print("\nBatch is still processing...")


def wait_for_completion(args: argparse.Namespace, runner: DoublewordBatchRunner) -> None:
    """Wait for batch to complete."""
    batch_id = args.wait

    print(f"Waiting for batch: {batch_id}")
    print(f"Poll interval: {args.poll_interval}s")
    if args.timeout:
        print(f"Timeout: {args.timeout}s")

    try:
        status = runner.wait_for_completion(
            batch_id=batch_id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )

        print(f"\n{'='*60}")
        print("Batch completed!")
        print(f"{'='*60}")
        print(f"Total requests: {status.total_requests}")
        print(f"Completed: {status.completed_requests}")
        print(f"Failed: {status.failed_requests}")

        print("\nDownload results with:")
        print(f"  python -m benchmarks.swe_smith.scripts.collect_results --batch-id {batch_id}")

    except TimeoutError as e:
        print(f"\n{e}")
        print("The batch is still processing. You can continue waiting later with:")
        print(f"  python -m benchmarks.swe_smith.scripts.submit_batch --wait {batch_id}")
        sys.exit(1)

    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Submit batch to Doubleword API or check status"
    )

    # Submission arguments
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=RESULTS_DIR / "batch_input.jsonl",
        help="Input path for batch JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for inference (default: gpt-4o)",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default=DOUBLEWORD_COMPLETION_WINDOW,
        choices=["24h", "1h"],
        help=f"Batch completion window (default: {DOUBLEWORD_COMPLETION_WINDOW})",
    )

    # Status/wait arguments
    parser.add_argument(
        "--status",
        type=str,
        metavar="BATCH_ID",
        help="Check status of existing batch",
    )
    parser.add_argument(
        "--wait",
        type=str,
        metavar="BATCH_ID",
        help="Wait for batch to complete",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks when waiting (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum seconds to wait for completion",
    )

    args = parser.parse_args()

    # Get API key and create runner
    api_key = get_api_key()
    runner = DoublewordBatchRunner(
        api_key=api_key,
        model=args.model,
        completion_window=args.completion_window,
    )

    # Determine action
    if args.status:
        check_status(args, runner)
    elif args.wait:
        wait_for_completion(args, runner)
    else:
        submit_batch(args, runner)


if __name__ == "__main__":
    main()
