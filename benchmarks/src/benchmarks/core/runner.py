"""Main benchmark runner for comparing routers."""

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from benchmarks.core.routers import CactusProfileRouter, ClaudeOracleRouter
from benchmarks.core.simulator import PerformanceSimulator
from benchmarks.core.metrics import BenchmarkMetrics
from benchmarks.datasets import load_combined


console = Console()


class BenchmarkRunner:
    """Runs head-to-head comparison between routers."""

    def __init__(
        self,
        profile_path: str | Path,
        anthropic_api_key: str,
        cost_bias: float = 0.5,
        seed: int = 42,
    ):
        """Initialize benchmark runner.

        Args:
            profile_path: Path to Cactus production profile JSON
            anthropic_api_key: Anthropic API key for Claude router
            cost_bias: Routing cost bias (0.0 = speed, 1.0 = quality)
            seed: Random seed for reproducibility
        """
        self.profile_path = Path(profile_path)
        self.cost_bias = cost_bias
        self.seed = seed

        # Load profile data
        with open(self.profile_path) as f:
            self.profile_data = json.load(f)

        # Initialize routers
        console.print("[cyan]Initializing routers...[/cyan]")
        self.profile_router = CactusProfileRouter(profile_path)
        self.claude_router = ClaudeOracleRouter(
            api_key=anthropic_api_key,
            available_models=self.profile_data["models"]
        )

        # Initialize simulator
        self.simulator = PerformanceSimulator(self.profile_data, seed=seed)

        console.print("[green]OK - Routers initialized[/green]")

    def run_benchmark(
        self,
        dataset_path: str | Path | None = None,
        mmlu_subset_size: int = 250,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run full benchmark comparison.

        Args:
            dataset_path: Path to custom dataset (None for default)
            mmlu_subset_size: Number of MMLU samples to include
            output_dir: Directory to save results (None for no save)

        Returns:
            Dict with full benchmark results
        """
        # Load dataset
        console.print("[cyan]Loading dataset...[/cyan]")
        prompts = load_combined(
            custom_path=dataset_path,
            mmlu_subset_size=mmlu_subset_size,
            shuffle=True,
            seed=self.seed
        )
        console.print(f"[green]OK - Loaded {len(prompts)} prompts[/green]")

        # Run comparisons
        console.print(f"[cyan]Running benchmark (cost_bias={self.cost_bias})...[/cyan]")
        results = []

        for prompt_data in track(
            prompts,
            description="Comparing routers",
            console=console
        ):
            prompt = prompt_data["input"]
            category = prompt_data.get("category", "unknown")

            # Route with profile router
            profile_route = self.profile_router.route(prompt, self.cost_bias)

            # Route with Claude oracle
            claude_route = self.claude_router.route(prompt, self.cost_bias)

            # Compare performance
            comparison = self.simulator.compare_routes(
                prompt, profile_route, claude_route
            )

            # Add metadata
            comparison["prompt_id"] = prompt_data.get("id", "unknown")
            comparison["prompt_category"] = category
            comparison["prompt_complexity"] = prompt_data.get("complexity", "unknown")

            results.append(comparison)

            # Brief pause to avoid API rate limits
            time.sleep(0.1)

        console.print(f"[green]OK - Completed {len(results)} comparisons[/green]")

        # Compute metrics
        console.print("[cyan]Computing metrics...[/cyan]")
        metrics_computer = BenchmarkMetrics(results)
        metrics = metrics_computer.compute_all()
        console.print("[green]OK - Metrics computed[/green]")

        # Package results
        benchmark_results = {
            "metadata": {
                "profile_path": str(self.profile_path),
                "cost_bias": self.cost_bias,
                "total_prompts": len(prompts),
                "mmlu_subset_size": mmlu_subset_size,
                "seed": self.seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "metrics": metrics,
            "results": results,
        }

        # Save results
        if output_dir:
            self._save_results(benchmark_results, output_dir)

        return benchmark_results

    def _save_results(
        self,
        results: dict[str, Any],
        output_dir: str | Path
    ) -> None:
        """Save benchmark results to disk.

        Args:
            results: Benchmark results dict
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"[green]OK - Results saved to {results_path}[/green]")

        # Save metrics summary
        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)

        console.print(f"[green]OK - Metrics saved to {metrics_path}[/green]")

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print summary of benchmark results.

        Args:
            results: Benchmark results dict
        """
        metrics = results["metrics"]
        summary = metrics["summary"]
        routing = metrics["routing"]
        performance = metrics["performance"]
        quality = metrics["quality"]
        cost = metrics["cost"]

        console.print("\n" + "=" * 80)
        console.print("[bold cyan]CACTUS ROUTER vs CLAUDE OPUS 4.5 ORACLE BENCHMARK[/bold cyan]")
        console.print("=" * 80)

        # Metadata
        meta = results["metadata"]
        console.print(f"\nProfile: {meta['profile_path']}")
        console.print(f"Total prompts: {meta['total_prompts']}")
        console.print(f"Cost bias: {meta['cost_bias']} (0.0=speed, 1.0=quality)")

        # Agreement
        console.print("\n[bold]ROUTING AGREEMENT:[/bold]")
        console.print(f"  Overall agreement: {summary['agreement_rate']:.1%} "
                     f"({summary['agreement_count']}/{summary['total_prompts']})")

        # Model distribution
        console.print("\n[bold]MODEL DISTRIBUTION:[/bold]")
        console.print("  [underline]Profile Router:[/underline]")
        for model, stats in list(routing["profile_router"]["model_distribution"].items())[:5]:
            console.print(f"    {model:20s} {stats['percentage']:5.1f}% ({stats['count']} prompts)")

        console.print("  [underline]Claude Oracle:[/underline]")
        for model, stats in list(routing["claude_router"]["model_distribution"].items())[:5]:
            console.print(f"    {model:20s} {stats['percentage']:5.1f}% ({stats['count']} prompts)")

        # Performance
        console.print("\n[bold]PERFORMANCE COMPARISON:[/bold]")
        console.print("  [underline]Profile Router:[/underline]")
        console.print(f"    Avg latency: {performance['profile_router']['avg_latency_ms']:.1f}ms "
                     f"(P95: {performance['profile_router']['p95_latency_ms']:.1f}ms, "
                     f"P99: {performance['profile_router']['p99_latency_ms']:.1f}ms)")
        console.print(f"    Avg memory: {performance['profile_router']['avg_memory_mb']:.0f}MB")

        console.print("  [underline]Claude Oracle:[/underline]")
        console.print(f"    Avg latency: {performance['claude_router']['avg_latency_ms']:.1f}ms "
                     f"(P95: {performance['claude_router']['p95_latency_ms']:.1f}ms, "
                     f"P99: {performance['claude_router']['p99_latency_ms']:.1f}ms)")
        console.print(f"    Avg memory: {performance['claude_router']['avg_memory_mb']:.0f}MB")

        console.print(f"  [bold]Latency difference:[/bold] "
                     f"{performance['comparison']['avg_latency_diff_ms']:+.1f}ms "
                     f"({performance['comparison']['avg_latency_diff_pct']:+.1f}%)")

        # Quality
        console.print("\n[bold]QUALITY COMPARISON:[/bold]")
        console.print(f"  Profile router avg error: {quality['profile_router']['avg_error_rate']:.2%}")
        console.print(f"  Claude router avg error: {quality['claude_router']['avg_error_rate']:.2%}")
        console.print(f"  Error difference: {quality['comparison']['avg_error_diff']:+.2%}")

        # Cost
        console.print("\n[bold]COST ANALYSIS:[/bold]")
        console.print(f"  Profile router: $0.00 (after training)")
        console.print(f"  Claude routing API: ${cost['claude_router_total_cost_usd']:.2f} total "
                     f"(${cost['claude_router_avg_cost_per_routing_usd']:.4f} per routing)")

        # Top categories
        console.print("\n[bold]TOP CATEGORIES BY AGREEMENT:[/bold]")
        by_cat = sorted(
            metrics["by_category"].items(),
            key=lambda x: x[1]["agreement_rate"],
            reverse=True
        )[:5]
        for cat, cat_metrics in by_cat:
            console.print(f"  {cat:20s} {cat_metrics['agreement_rate']:.1%} "
                         f"({cat_metrics['count']} prompts)")

        console.print("\n" + "=" * 80)
