#!/usr/bin/env python3
"""Sample instances from SWE-smith clusters and save for pre-fetching.

This script samples instances from each semantic cluster and saves:
1. List of sampled instance IDs (for pre-fetching)
2. Cluster map (instance_id -> cluster_id) for profiling

Usage:
    python -m benchmarks.swe_smith.scripts.sample_instances \
        --output results/gpt4o/sampled_ids.json \
        --cluster-map results/gpt4o/cluster_map.json

    # Show stats only (no output files)
    python -m benchmarks.swe_smith.scripts.sample_instances --stats-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_smith.config import (
    MIN_SAMPLES,
    RESULTS_DIR,
    SAMPLE_FRACTION,
)
from swe_smith.sampler import ClusterSampler


def save_json(data: dict, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def print_sampling_stats(sampler: ClusterSampler) -> None:
    """Print detailed sampling statistics."""
    print(f"\n{'='*60}")
    print("SWE-smith Cluster Sampling Statistics")
    print(f"{'='*60}")
    print(f"Total clusters: {sampler.n_clusters}")
    print(f"Total instances (excl. noise): {sampler.total_instances}")
    print(f"Min samples per cluster: {sampler.min_samples}")
    print(f"Sample fraction: {sampler.sample_fraction}")
    print(f"Total samples to draw: {sampler.total_samples}")

    print("\nPer-cluster breakdown:")
    print(f"{'Cluster':<10} {'Size':<10} {'Sample':<10} {'%':<10}")
    print("-" * 40)

    stats = sampler.get_cluster_stats()
    for cluster_id, info in sorted(stats.items()):
        pct = info["sample_size"] / info["size"] * 100
        print(f"{cluster_id:<10} {info['size']:<10} {info['sample_size']:<10} {pct:.1f}%")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sample instances from SWE-smith clusters for evaluation"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=RESULTS_DIR / "sampled_ids.json",
        help="Output path for sampled instance IDs JSON",
    )
    parser.add_argument(
        "--cluster-map",
        type=Path,
        default=RESULTS_DIR / "cluster_map.json",
        help="Output path for cluster map JSON (instance_id -> cluster_id)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help=f"Minimum samples per cluster (default: {MIN_SAMPLES})",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=SAMPLE_FRACTION,
        help=f"Fraction of cluster to sample (default: {SAMPLE_FRACTION})",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't save files",
    )

    args = parser.parse_args()

    # Initialize sampler
    sampler = ClusterSampler(
        min_samples=args.min_samples,
        sample_fraction=args.sample_fraction,
    )

    # Print stats
    print_sampling_stats(sampler)

    if args.stats_only:
        return

    # Sample instances from all clusters
    print(f"\nSampling with random_state={args.random_state}...")
    samples = sampler.sample_all_clusters(random_state=args.random_state)

    # Flatten to list of instance IDs
    all_ids = []
    for cluster_id in sorted(samples.keys()):
        all_ids.extend(samples[cluster_id])

    print(f"Sampled {len(all_ids)} instances from {len(samples)} clusters")

    # Save sampled instance IDs
    sampled_data = {
        "instance_ids": all_ids,
        "random_state": args.random_state,
        "min_samples": args.min_samples,
        "sample_fraction": args.sample_fraction,
        "n_clusters": len(samples),
        "total_sampled": len(all_ids),
    }
    save_json(sampled_data, args.output)

    # Build and save cluster map (instance_id -> cluster_id)
    cluster_map = {}
    for cluster_id, instance_ids in samples.items():
        for inst_id in instance_ids:
            cluster_map[inst_id] = cluster_id

    cluster_map_data = {
        "cluster_map": cluster_map,
        "random_state": args.random_state,
        "n_clusters": len(samples),
        "total_instances": len(cluster_map),
    }
    save_json(cluster_map_data, args.cluster_map)

    print(f"\n{'='*60}")
    print("Sampling complete!")
    print(f"{'='*60}")
    print(f"Instance IDs saved to: {args.output}")
    print(f"Cluster map saved to: {args.cluster_map}")
    print(f"Total instances: {len(all_ids)}")


if __name__ == "__main__":
    main()
