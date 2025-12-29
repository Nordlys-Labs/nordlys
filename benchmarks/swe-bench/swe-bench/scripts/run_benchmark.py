#!/usr/bin/env python3
"""CLI entry point for SWE-bench Nordlys routing benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add parent directory to path for imports (handles hyphenated directory names)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark import BenchmarkConfig, SWEBenchBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Nordlys routing benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="Output directory",
    )
    parser.add_argument(
        "--aws-region",
        type=str,
        default=None,
        help="AWS region (uses AWS_DEFAULT_REGION env var if not set)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        output_dir=args.output,
        aws_region=args.aws_region,
    )

    logger.info(f"Output: {config.output_dir}")

    benchmark = SWEBenchBenchmark(config)
    result = benchmark.run()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Instances processed: {result.total_instances}")
    print(f"  Patches found: {result.patches_found}")
    print(f"  Patches missing: {result.patches_missing}")
    print(f"  Total cost: ${result.total_cost:.2f}")
    print(f"\nRouting distribution:")
    for model_id, count in sorted(
        result.routing_distribution.items(), key=lambda x: -x[1]
    ):
        pct = 100 * count / result.total_instances
        cost = result.cost_by_model.get(model_id, 0)
        print(f"    {model_id}: {count} ({pct:.1f}%) - ${cost:.2f}")
    print(f"\nOutputs:")
    print(f"  Predictions: {result.predictions_file}")
    print(f"  Summary: {result.summary_file}")
    print("=" * 60)

    # Print submission command
    print("\nTo submit to SWE-bench:")
    print(f"  sb-cli submit swe-bench_verified test \\")
    print(f"    --predictions_path {result.predictions_file} \\")
    print(f"    --run_id nordlys-routing-v1")


if __name__ == "__main__":
    main()
