"""Metrics computation for benchmark results."""

import numpy as np
from typing import Any


class BenchmarkMetrics:
    """Computes and aggregates benchmark metrics."""

    def __init__(self, results: list[dict[str, Any]]):
        """Initialize with benchmark results.

        Args:
            results: List of comparison results from runner
        """
        self.results = results

    def compute_all(self) -> dict[str, Any]:
        """Compute all metrics.

        Returns:
            Dict with comprehensive metrics
        """
        return {
            "summary": self.compute_summary(),
            "routing": self.compute_routing_metrics(),
            "performance": self.compute_performance_metrics(),
            "quality": self.compute_quality_metrics(),
            "cost": self.compute_cost_metrics(),
            "by_category": self.compute_category_metrics(),
        }

    def compute_summary(self) -> dict[str, Any]:
        """Compute high-level summary metrics."""
        total = len(self.results)
        agreements = sum(1 for r in self.results if r["agreement"])

        return {
            "total_prompts": total,
            "agreement_rate": agreements / total if total > 0 else 0,
            "agreement_count": agreements,
            "disagreement_count": total - agreements,
        }

    def compute_routing_metrics(self) -> dict[str, Any]:
        """Compute routing-specific metrics."""
        profile_times = [
            r["profile_router"]["routing_time_ms"] for r in self.results
        ]
        claude_times = [
            r["claude_router"]["routing_time_ms"] for r in self.results
        ]

        # Model distribution
        profile_models = [r["profile_router"]["model"] for r in self.results]
        claude_models = [r["claude_router"]["model"] for r in self.results]

        profile_dist = self._distribution(profile_models)
        claude_dist = self._distribution(claude_models)

        return {
            "profile_router": {
                "avg_routing_time_ms": np.mean(profile_times),
                "p95_routing_time_ms": np.percentile(profile_times, 95),
                "model_distribution": profile_dist,
            },
            "claude_router": {
                "avg_routing_time_ms": np.mean(claude_times),
                "p95_routing_time_ms": np.percentile(claude_times, 95),
                "model_distribution": claude_dist,
            },
        }

    def compute_performance_metrics(self) -> dict[str, Any]:
        """Compute performance metrics (latency, throughput)."""
        profile_latencies = [
            r["profile_router"]["e2e_latency_ms"] for r in self.results
        ]
        claude_latencies = [
            r["claude_router"]["e2e_latency_ms"] for r in self.results
        ]

        profile_memory = [
            r["profile_router"]["memory_mb"] for r in self.results
        ]
        claude_memory = [
            r["claude_router"]["memory_mb"] for r in self.results
        ]

        return {
            "profile_router": {
                "avg_latency_ms": np.mean(profile_latencies),
                "median_latency_ms": np.median(profile_latencies),
                "p95_latency_ms": np.percentile(profile_latencies, 95),
                "p99_latency_ms": np.percentile(profile_latencies, 99),
                "avg_memory_mb": np.mean(profile_memory),
            },
            "claude_router": {
                "avg_latency_ms": np.mean(claude_latencies),
                "median_latency_ms": np.median(claude_latencies),
                "p95_latency_ms": np.percentile(claude_latencies, 95),
                "p99_latency_ms": np.percentile(claude_latencies, 99),
                "avg_memory_mb": np.mean(claude_memory),
            },
            "comparison": {
                "avg_latency_diff_ms": np.mean([
                    r["comparison"]["latency_diff_ms"] for r in self.results
                ]),
                "avg_latency_diff_pct": np.mean([
                    r["comparison"]["latency_diff_pct"] for r in self.results
                ]),
                "claude_faster_count": sum(
                    1 for r in self.results if r["comparison"]["claude_faster"]
                ),
            },
        }

    def compute_quality_metrics(self) -> dict[str, Any]:
        """Compute quality metrics (error rates)."""
        profile_errors = [
            r["profile_router"]["error_rate"] for r in self.results
        ]
        claude_errors = [
            r["claude_router"]["error_rate"] for r in self.results
        ]

        return {
            "profile_router": {
                "avg_error_rate": np.mean(profile_errors),
                "median_error_rate": np.median(profile_errors),
                "min_error_rate": np.min(profile_errors),
                "max_error_rate": np.max(profile_errors),
            },
            "claude_router": {
                "avg_error_rate": np.mean(claude_errors),
                "median_error_rate": np.median(claude_errors),
                "min_error_rate": np.min(claude_errors),
                "max_error_rate": np.max(claude_errors),
            },
            "comparison": {
                "avg_error_diff": np.mean([
                    r["comparison"]["error_diff"] for r in self.results
                ]),
                "claude_more_accurate_count": sum(
                    1 for r in self.results if r["comparison"]["claude_more_accurate"]
                ),
            },
        }

    def compute_cost_metrics(self) -> dict[str, Any]:
        """Compute cost metrics."""
        total_claude_cost = sum(
            r["claude_router"].get("api_cost_usd", 0) for r in self.results
        )

        avg_claude_cost = total_claude_cost / len(self.results) if self.results else 0

        return {
            "profile_router_cost_usd": 0.0,  # Free after training
            "claude_router_total_cost_usd": total_claude_cost,
            "claude_router_avg_cost_per_routing_usd": avg_claude_cost,
            "total_prompts": len(self.results),
        }

    def compute_category_metrics(self) -> dict[str, Any]:
        """Compute metrics broken down by prompt category."""
        # Group by category
        by_category: dict[str, list] = {}
        for r in self.results:
            cat = r.get("prompt_category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        category_metrics = {}
        for cat, cat_results in by_category.items():
            agreements = sum(1 for r in cat_results if r["agreement"])

            category_metrics[cat] = {
                "count": len(cat_results),
                "agreement_rate": agreements / len(cat_results),
                "profile_avg_latency_ms": np.mean([
                    r["profile_router"]["e2e_latency_ms"] for r in cat_results
                ]),
                "claude_avg_latency_ms": np.mean([
                    r["claude_router"]["e2e_latency_ms"] for r in cat_results
                ]),
                "profile_avg_error_rate": np.mean([
                    r["profile_router"]["error_rate"] for r in cat_results
                ]),
                "claude_avg_error_rate": np.mean([
                    r["claude_router"]["error_rate"] for r in cat_results
                ]),
            }

        return category_metrics

    def _distribution(self, items: list[str]) -> dict[str, Any]:
        """Compute distribution of items.

        Args:
            items: List of model IDs

        Returns:
            Dict with counts and percentages
        """
        from collections import Counter

        counts = Counter(items)
        total = len(items)

        return {
            item: {
                "count": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
            for item, count in counts.most_common()
        }
