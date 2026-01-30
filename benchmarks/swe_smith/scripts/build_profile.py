#!/usr/bin/env python3
"""Build Nordlys profile from swesmith evaluation results.

Parses the swesmith harness report and computes per-cluster error rates
for Nordlys model routing.

Usage:
    python -m benchmarks.swe_smith.scripts.build_profile \
        --report logs/run_evaluation/gpt4o-eval/report.json \
        --cluster-map results/gpt4o/cluster_map.json \
        --model "gpt-4o" \
        --provider "openai" \
        --output results/profiles/gpt-4o/profile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_smith.config import N_CLUSTERS, PROFILES_DIR, RESULTS_DIR
from swe_smith.profiler import ClusterProfiler


def load_cluster_map(path: Path) -> dict[str, int]:
    """Load cluster map from JSON file.

    Args:
        path: Path to cluster map JSON.

    Returns:
        Dict mapping instance_id to cluster_id.
    """
    with open(path) as f:
        data = json.load(f)
    return data["cluster_map"]


def parse_swesmith_report(report_path: Path) -> dict[str, bool]:
    """Parse swesmith harness report to extract resolved status.

    Args:
        report_path: Path to swesmith report.json file.

    Returns:
        Dict mapping instance_id to resolved status (True/False).
    """
    with open(report_path) as f:
        report = json.load(f)

    results = {}

    # Try format: {"resolved": ["id1", "id2", ...], "unresolved": [...]}
    if "resolved" in report:
        for inst_id in report.get("resolved", []):
            results[inst_id] = True
        for inst_id in report.get("unresolved", []):
            results[inst_id] = False
        return results

    # Try format: {"instance_id": {"resolved": true/false, ...}}
    for inst_id, info in report.items():
        if isinstance(info, dict) and "resolved" in info:
            results[inst_id] = info["resolved"]
        elif isinstance(info, bool):
            results[inst_id] = info

    return results


def parse_model_info(model: str) -> tuple[str, str]:
    """Extract provider and model name from model identifier.

    Args:
        model: Model identifier (e.g., "anthropic/claude-3-sonnet" or "gpt-4o").

    Returns:
        Tuple of (provider, model_name).
    """
    if "/" in model:
        parts = model.split("/")
        provider = parts[0]
        model_name = parts[-1]
    else:
        model_name = model
        model_lower = model.lower()
        if "claude" in model_lower:
            provider = "anthropic"
        elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            provider = "openai"
        elif "deepseek" in model_lower:
            provider = "deepseek"
        elif "gemini" in model_lower:
            provider = "google"
        else:
            provider = "unknown"

    return provider, model_name


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build Nordlys profile from swesmith evaluation results"
    )

    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        required=True,
        help="Path to swesmith report.json file",
    )
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=RESULTS_DIR / "cluster_map.json",
        help="Path to cluster map JSON (instance_id -> cluster_id)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'gpt-4o', 'claude-3-sonnet')",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Model provider (e.g., 'openai', 'anthropic'). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional run ID for metadata",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for profile JSON (default: profiles/<model>/profile.json)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=N_CLUSTERS,
        help=f"Total number of clusters (default: {N_CLUSTERS})",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.report.exists():
        print(f"Error: Report file not found: {args.report}")
        return

    if not args.cluster_map.exists():
        print(f"Error: Cluster map not found: {args.cluster_map}")
        return

    # Determine provider and model name
    if args.provider:
        provider = args.provider
        model_name = args.model
    else:
        provider, model_name = parse_model_info(args.model)
        print(f"Auto-detected provider: {provider}")

    # Load cluster map
    print(f"Loading cluster map from: {args.cluster_map}")
    cluster_map = load_cluster_map(args.cluster_map)
    print(f"Loaded {len(cluster_map)} instance -> cluster mappings")

    # Parse evaluation results
    print(f"Parsing report from: {args.report}")
    results = parse_swesmith_report(args.report)
    print(f"Parsed {len(results)} evaluation results")

    # Check coverage
    in_both = set(cluster_map.keys()) & set(results.keys())
    missing_results = set(cluster_map.keys()) - set(results.keys())
    extra_results = set(results.keys()) - set(cluster_map.keys())

    print(f"Instances with both cluster and result: {len(in_both)}")
    if missing_results:
        print(f"Warning: {len(missing_results)} instances in cluster map but not in results")
    if extra_results:
        print(f"Warning: {len(extra_results)} instances in results but not in cluster map")

    # Build profile
    print("\nBuilding profile...")
    profiler = ClusterProfiler(
        n_clusters=args.n_clusters,
        output_dir=PROFILES_DIR,
    )

    profile = profiler.build_profile(
        model_name=model_name,
        provider=provider,
        cluster_map=cluster_map,
        results=results,
        run_id=args.run_id,
    )

    # Add metadata
    profile["eval_type"] = "swe-smith"
    profile["inference_method"] = "one-shot-batch"

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = PROFILES_DIR / model_name / "profile.json"

    # Save profile
    profiler.save_profile(profile, output_path)

    # Print summary
    profiler.print_summary(profile)

    print(f"\n{'='*60}")
    print("Profile build complete!")
    print(f"{'='*60}")
    print(f"Profile saved to: {output_path}")

    # Print JSON output format summary
    print("\nOutput format:")
    print(json.dumps(
        {
            "model_name": profile["model_name"],
            "provider": profile["provider"],
            "eval_type": profile["eval_type"],
            "inference_method": profile["inference_method"],
            "overall_error_rate": profile["overall_error_rate"],
            "total_evaluated": profile["total_evaluated"],
            "total_resolved": profile["total_resolved"],
            "error_rates": f"[{len(profile['error_rates'])} cluster error rates...]",
            "cluster_stats": f"{{{len(profile['cluster_stats'])} cluster stats...}}",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
