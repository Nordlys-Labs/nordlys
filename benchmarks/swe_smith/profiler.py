"""Per-cluster error rate calculation for Nordlys model routing."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from swe_smith.config import N_CLUSTERS, PROFILES_DIR


def calculate_cluster_stats(
    cluster_map: dict[str, int], results: dict[str, bool]
) -> dict[int, dict[str, int]]:
    """Calculate resolved/total counts per cluster.

    Args:
        cluster_map: Mapping from instance_id to cluster_id.
        results: Mapping from instance_id to resolved status.

    Returns:
        Dictionary mapping cluster_id to {"total": N, "resolved": M}.
    """
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "resolved": 0})

    for inst_id, resolved in results.items():
        if inst_id not in cluster_map:
            continue
        cluster_id = cluster_map[inst_id]
        stats[cluster_id]["total"] += 1
        if resolved:
            stats[cluster_id]["resolved"] += 1

    return dict(stats)


def compute_error_rates(
    cluster_stats: dict[int, dict[str, int]], n_clusters: int = N_CLUSTERS
) -> list[float]:
    """Compute error rate for each cluster.

    Formula: error_rate[i] = 1 - (resolved / total)
    Default 1.0 for empty/missing clusters.

    Args:
        cluster_stats: Per-cluster stats from calculate_cluster_stats.
        n_clusters: Total number of clusters.

    Returns:
        List of error rates indexed by cluster_id.
    """
    error_rates = []
    for i in range(n_clusters):
        if i in cluster_stats and cluster_stats[i]["total"] > 0:
            rate = 1.0 - (cluster_stats[i]["resolved"] / cluster_stats[i]["total"])
            error_rates.append(round(rate, 4))
        else:
            error_rates.append(1.0)
    return error_rates


class ClusterProfiler:
    """Builds model profiles with per-cluster error rates."""

    def __init__(
        self,
        n_clusters: int = N_CLUSTERS,
        output_dir: Path = PROFILES_DIR,
    ):
        """Initialize the profiler.

        Args:
            n_clusters: Total number of clusters.
            output_dir: Directory to save profiles.
        """
        self.n_clusters = n_clusters
        self.output_dir = output_dir

    def build_profile(
        self,
        model_name: str,
        provider: str,
        cluster_map: dict[str, int],
        results: dict[str, bool],
        run_id: str | None = None,
    ) -> dict:
        """Build a model profile with per-cluster error rates.

        Args:
            model_name: Name of the model (e.g., "claude-3-sonnet").
            provider: Model provider (e.g., "anthropic").
            cluster_map: Mapping from instance_id to cluster_id.
            results: Mapping from instance_id to resolved status.
            run_id: Optional run identifier for metadata.

        Returns:
            Profile dict ready for Nordlys routing.
        """
        # Calculate stats
        cluster_stats = calculate_cluster_stats(cluster_map, results)
        error_rates = compute_error_rates(cluster_stats, self.n_clusters)

        # Calculate overall stats
        total_evaluated = sum(s["total"] for s in cluster_stats.values())
        total_resolved = sum(s["resolved"] for s in cluster_stats.values())
        overall_error_rate = (
            1.0 - (total_resolved / total_evaluated) if total_evaluated > 0 else 1.0
        )

        profile = {
            "model_name": model_name,
            "provider": provider,
            "eval_type": "swe-smith",
            "error_rates": error_rates,
            "cluster_stats": {str(k): v for k, v in sorted(cluster_stats.items())},
            "overall_error_rate": round(overall_error_rate, 4),
            "total_evaluated": total_evaluated,
            "total_resolved": total_resolved,
            "created_at": datetime.now().isoformat(),
        }

        if run_id:
            profile["run_id"] = run_id

        return profile

    def save_profile(self, profile: dict, output_path: Path | None = None) -> Path:
        """Save profile to JSON file.

        Args:
            profile: Profile dict to save.
            output_path: Optional specific path. If None, uses output_dir/model_name/profile.json.

        Returns:
            Path where profile was saved.
        """
        if output_path is None:
            model_name = profile["model_name"]
            output_path = self.output_dir / model_name / "profile.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"Saved profile to: {output_path}")
        return output_path

    def load_profile(self, model_name: str) -> dict:
        """Load an existing profile.

        Args:
            model_name: Name of the model.

        Returns:
            Profile dict.

        Raises:
            FileNotFoundError: If profile doesn't exist.
        """
        profile_path = self.output_dir / model_name / "profile.json"
        with open(profile_path) as f:
            return json.load(f)

    def profile_from_results(
        self,
        model_name: str,
        provider: str,
        cluster_map: dict[str, int],
        results: dict[str, bool],
        run_id: str | None = None,
        save: bool = True,
    ) -> dict:
        """Build and optionally save a profile from evaluation results.

        Args:
            model_name: Name of the model.
            provider: Model provider.
            cluster_map: Mapping from instance_id to cluster_id.
            results: Mapping from instance_id to resolved status.
            run_id: Optional run identifier.
            save: Whether to save the profile to disk.

        Returns:
            Profile dict.
        """
        profile = self.build_profile(
            model_name, provider, cluster_map, results, run_id
        )

        if save:
            self.save_profile(profile)

        return profile

    def print_summary(self, profile: dict) -> None:
        """Print a summary of the profile.

        Args:
            profile: Profile dict to summarize.
        """
        print(f"\n{'='*60}")
        print(f"Profile Summary: {profile['model_name']}")
        print(f"{'='*60}")
        print(f"Provider: {profile['provider']}")
        print(f"Eval Type: {profile['eval_type']}")
        print(f"Total Evaluated: {profile['total_evaluated']}")
        print(f"Total Resolved: {profile['total_resolved']}")
        print(f"Overall Error Rate: {profile['overall_error_rate']:.2%}")
        print(f"Clusters with data: {len(profile['cluster_stats'])}/{self.n_clusters}")

        # Error rate distribution
        error_rates = profile["error_rates"]
        print("\nError Rate Distribution:")
        print(f"  Min: {min(error_rates):.2%}")
        print(f"  Max: {max(error_rates):.2%}")
        print(f"  Mean: {sum(error_rates)/len(error_rates):.2%}")

        # Top 5 easiest clusters
        sorted_clusters = sorted(
            enumerate(error_rates), key=lambda x: x[1]
        )
        print("\nTop 5 Easiest Clusters:")
        for cluster_id, rate in sorted_clusters[:5]:
            stats = profile["cluster_stats"].get(str(cluster_id), {})
            resolved = stats.get("resolved", 0)
            total = stats.get("total", 0)
            print(f"  Cluster {cluster_id}: {rate:.2%} ({resolved}/{total})")

        print("\nTop 5 Hardest Clusters:")
        for cluster_id, rate in sorted_clusters[-5:]:
            stats = profile["cluster_stats"].get(str(cluster_id), {})
            resolved = stats.get("resolved", 0)
            total = stats.get("total", 0)
            print(f"  Cluster {cluster_id}: {rate:.2%} ({resolved}/{total})")
