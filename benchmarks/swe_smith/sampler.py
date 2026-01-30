"""Cluster sampling logic for SWE-smith instances."""

import json
import random
from collections import Counter
from pathlib import Path

from swe_smith.config import (
    CLUSTER_ASSIGNMENTS,
    MIN_SAMPLES,
    NOISE_CLUSTER,
    SAMPLE_FRACTION,
)


def load_cluster_assignments(path: Path = CLUSTER_ASSIGNMENTS) -> tuple[list[str], list[int]]:
    """Load instance_ids and cluster_labels from cluster_assignments.json.

    Args:
        path: Path to cluster_assignments.json file.

    Returns:
        Tuple of (instance_ids, cluster_labels).
    """
    with open(path) as f:
        data = json.load(f)
    return data["instance_ids"], data["cluster_labels"]


def compute_sample_sizes(
    cluster_labels: list[int],
    min_samples: int = MIN_SAMPLES,
    sample_fraction: float = SAMPLE_FRACTION,
) -> dict[int, int]:
    """Compute sample sizes for each cluster.

    Formula: n_i = max(min_samples, sample_fraction * cluster_size)
    Skips noise cluster (-1).

    Args:
        cluster_labels: List of cluster labels for all instances.
        min_samples: Minimum number of samples per cluster.
        sample_fraction: Fraction of cluster to sample.

    Returns:
        Dictionary mapping cluster_id to sample size.
    """
    counts = Counter(cluster_labels)
    return {
        cluster_id: max(min_samples, int(sample_fraction * count))
        for cluster_id, count in counts.items()
        if cluster_id != NOISE_CLUSTER
    }


def build_cluster_instance_map(
    instance_ids: list[str], cluster_labels: list[int]
) -> dict[int, list[str]]:
    """Build mapping from cluster_id to list of instance_ids.

    Args:
        instance_ids: List of all instance IDs.
        cluster_labels: List of cluster labels (same order as instance_ids).

    Returns:
        Dictionary mapping cluster_id to list of instance_ids in that cluster.
    """
    cluster_map: dict[int, list[str]] = {}
    for inst_id, cluster_id in zip(instance_ids, cluster_labels):
        if cluster_id == NOISE_CLUSTER:
            continue
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = []
        cluster_map[cluster_id].append(inst_id)
    return cluster_map


class ClusterSampler:
    """Samples instances from SWE-smith clusters for evaluation."""

    def __init__(
        self,
        cluster_assignments_path: Path = CLUSTER_ASSIGNMENTS,
        min_samples: int = MIN_SAMPLES,
        sample_fraction: float = SAMPLE_FRACTION,
    ):
        """Initialize the sampler.

        Args:
            cluster_assignments_path: Path to cluster_assignments.json.
            min_samples: Minimum samples per cluster.
            sample_fraction: Fraction of cluster to sample.
        """
        self.cluster_assignments_path = cluster_assignments_path
        self.min_samples = min_samples
        self.sample_fraction = sample_fraction

        # Load assignments
        self.instance_ids, self.cluster_labels = load_cluster_assignments(
            cluster_assignments_path
        )
        self.cluster_instance_map = build_cluster_instance_map(
            self.instance_ids, self.cluster_labels
        )
        self.sample_sizes = compute_sample_sizes(
            self.cluster_labels, min_samples, sample_fraction
        )

    @property
    def n_clusters(self) -> int:
        """Number of valid clusters (excluding noise)."""
        return len(self.cluster_instance_map)

    @property
    def total_instances(self) -> int:
        """Total number of instances across all valid clusters."""
        return sum(len(ids) for ids in self.cluster_instance_map.values())

    @property
    def total_samples(self) -> int:
        """Total number of samples to be drawn across all clusters."""
        return sum(self.sample_sizes.values())

    def get_cluster_stats(self) -> dict[int, dict[str, int]]:
        """Get statistics for each cluster.

        Returns:
            Dictionary mapping cluster_id to stats dict with 'size' and 'sample_size'.
        """
        return {
            cluster_id: {
                "size": len(self.cluster_instance_map[cluster_id]),
                "sample_size": self.sample_sizes[cluster_id],
            }
            for cluster_id in sorted(self.cluster_instance_map.keys())
        }

    def sample_cluster(
        self, cluster_id: int, random_state: int | None = None
    ) -> list[str]:
        """Sample instances from a single cluster.

        Args:
            cluster_id: The cluster to sample from.
            random_state: Optional random seed for reproducibility.

        Returns:
            List of sampled instance IDs.
        """
        if cluster_id not in self.cluster_instance_map:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        instances = self.cluster_instance_map[cluster_id]
        sample_size = self.sample_sizes[cluster_id]

        if random_state is not None:
            rng = random.Random(random_state)
        else:
            rng = random.Random()

        # If sample size >= cluster size, return all instances
        if sample_size >= len(instances):
            return instances.copy()

        return rng.sample(instances, sample_size)

    def sample_all_clusters(
        self, random_state: int = 42
    ) -> dict[int, list[str]]:
        """Sample instances from all clusters.

        Args:
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary mapping cluster_id to list of sampled instance IDs.
        """
        samples = {}
        for cluster_id in sorted(self.cluster_instance_map.keys()):
            # Use different seed per cluster for reproducibility
            cluster_seed = random_state + cluster_id
            samples[cluster_id] = self.sample_cluster(cluster_id, cluster_seed)
        return samples

    def get_instance_cluster_map(
        self, sampled_ids: set[str] | None = None
    ) -> dict[str, int]:
        """Get mapping from instance_id to cluster_id.

        Args:
            sampled_ids: Optional set of instance IDs to filter to.
                        If None, returns all instances.

        Returns:
            Dictionary mapping instance_id to cluster_id.
        """
        result = {}
        for inst_id, cluster_id in zip(self.instance_ids, self.cluster_labels):
            if cluster_id == NOISE_CLUSTER:
                continue
            if sampled_ids is None or inst_id in sampled_ids:
                result[inst_id] = cluster_id
        return result
