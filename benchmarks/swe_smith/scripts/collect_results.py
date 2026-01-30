#!/usr/bin/env python3
"""Download results from completed Doubleword batch and format for swesmith.

Downloads the batch results and converts to the prediction format expected
by swesmith.harness.eval.

Usage:
    # Check status only
    python -m benchmarks.swe_smith.scripts.collect_results \
        --batch-id batch_abc123 \
        --status-only

    # Download results
    python -m benchmarks.swe_smith.scripts.collect_results \
        --batch-id batch_abc123 \
        --output results/gpt4o/predictions.jsonl

    # Also download errors
    python -m benchmarks.swe_smith.scripts.collect_results \
        --batch-id batch_abc123 \
        --output results/gpt4o/predictions.jsonl \
        --errors results/gpt4o/errors.jsonl

Environment:
    DOUBLEWORD_API_KEY: API key for Doubleword (required)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from swe_smith.batch_runner import (
    DoublewordBatchRunner,
    save_predictions_jsonl,
)
from swe_smith.config import RESULTS_DIR


def get_api_key() -> str:
    """Get Doubleword API key from environment."""
    key = os.environ.get("DOUBLEWORD_API_KEY")
    if not key:
        print("Error: DOUBLEWORD_API_KEY environment variable not set")
        print("Set it with: export DOUBLEWORD_API_KEY='your-key'")
        sys.exit(1)
    return key


def print_status(runner: DoublewordBatchRunner, batch_id: str) -> None:
    """Print batch status."""
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
        print("\nBatch is complete and ready for download!")
    elif status.is_failed:
        print(f"\nBatch failed with status: {status.status}")
    else:
        print("\nBatch is still processing. Check back later.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download results from completed Doubleword batch"
    )

    parser.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Batch ID to download results for",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=RESULTS_DIR / "predictions.jsonl",
        help="Output path for predictions JSONL",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        help="Output path for errors JSONL (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for predictions (default: gpt-4o)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check status, don't download",
    )

    args = parser.parse_args()

    # Get API key and create runner
    api_key = get_api_key()
    runner = DoublewordBatchRunner(
        api_key=api_key,
        model=args.model,
    )

    # Check status
    status = runner.check_status(args.batch_id)

    if args.status_only:
        print_status(runner, args.batch_id)
        return

    # Validate batch is complete
    if not status.is_complete:
        print(f"Error: Batch is not complete (status: {status.status})")
        print(f"Progress: {status.completed_requests}/{status.total_requests}")
        print("\nWait for completion with:")
        print(f"  python -m benchmarks.swe_smith.scripts.submit_batch --wait {args.batch_id}")
        sys.exit(1)

    # Download predictions
    print(f"Downloading results for batch: {args.batch_id}")
    predictions = runner.download_results(args.batch_id)

    # Save predictions
    save_predictions_jsonl(predictions, args.output)

    # Count patches
    empty_patches = sum(1 for p in predictions if not p.get("model_patch"))
    non_empty = len(predictions) - empty_patches

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"With patches: {non_empty}")
    print(f"Empty patches: {empty_patches}")
    print(f"Saved to: {args.output}")

    # Download errors if requested
    if args.errors:
        errors = runner.download_errors(args.batch_id)
        if errors:
            args.errors.parent.mkdir(parents=True, exist_ok=True)
            with open(args.errors, "w") as f:
                for error in errors:
                    f.write(json.dumps(error) + "\n")
            print(f"Errors saved to: {args.errors} ({len(errors)} errors)")
        else:
            print("No errors to download")

    # Show sample prediction
    if predictions:
        sample = predictions[0]
        print("\nSample prediction:")
        print(f"  instance_id: {sample['instance_id']}")
        print(f"  model_name_or_path: {sample['model_name_or_path']}")
        patch_preview = sample.get("model_patch", "")[:200]
        if len(sample.get("model_patch", "")) > 200:
            patch_preview += "..."
        print(f"  model_patch: {patch_preview}")

    # Print next steps
    print("\nNext steps:")
    print("1. Run swesmith evaluation (requires Ubuntu/Docker):")
    print("   python -m swesmith.harness.eval \\")
    print(f"       -p {args.output} \\")
    print("       --run_id \"gpt4o-swe-smith-eval\" \\")
    print("       -w 10")
    print("\n2. Build Nordlys profile from results:")
    print("   python -m benchmarks.swe_smith.scripts.build_profile \\")
    print("       --report logs/run_evaluation/gpt4o-swe-smith-eval/report.json \\")
    print("       --cluster-map results/gpt4o/cluster_map.json")


if __name__ == "__main__":
    main()
