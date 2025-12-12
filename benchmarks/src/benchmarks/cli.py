"""Command-line interface for benchmarks."""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

from benchmarks.core.runner import BenchmarkRunner
from benchmarks.visualizations.charts import generate_all_charts


console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Cactus router vs Claude Opus 4.5 oracle"
    )

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Path to Cactus production profile JSON"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom dataset JSON (default: bundled dataset)"
    )

    parser.add_argument(
        "--mmlu-size",
        type=int,
        default=250,
        help="Number of MMLU samples to include (default: 250)"
    )

    parser.add_argument(
        "--cost-bias",
        type=float,
        default=0.5,
        help="Routing cost bias 0.0=speed, 1.0=quality (default: 0.5)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization charts"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: Anthropic API key required[/red]")
        console.print("Set via --api-key or ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Validate profile path
    profile_path = Path(args.profile)
    if not profile_path.exists():
        console.print(f"[red]Error: Profile not found: {profile_path}[/red]")
        sys.exit(1)

    # Initialize runner
    try:
        runner = BenchmarkRunner(
            profile_path=profile_path,
            anthropic_api_key=api_key,
            cost_bias=args.cost_bias,
            seed=args.seed
        )
    except Exception as e:
        console.print(f"[red]Error initializing runner: {e}[/red]")
        sys.exit(1)

    # Run benchmark
    try:
        results = runner.run_benchmark(
            dataset_path=args.dataset,
            mmlu_subset_size=args.mmlu_size,
            output_dir=args.output
        )
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print summary
    runner.print_summary(results)

    # Generate visualizations
    if args.visualize:
        console.print("\n[cyan]Generating visualizations...[/cyan]")
        try:
            output_dir = Path(args.output)
            charts_dir = output_dir / "charts"
            charts_dir.mkdir(exist_ok=True)

            generate_all_charts(results, charts_dir)

            console.print(f"[green]OK - Charts saved to {charts_dir}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate charts: {e}[/yellow]")

    console.print(f"\n[bold green]Benchmark complete! Results saved to {args.output}[/bold green]")


if __name__ == "__main__":
    main()
