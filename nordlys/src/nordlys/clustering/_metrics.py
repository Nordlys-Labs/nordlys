"""Clustering metrics and info classes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ClusterInfo:
    """Information about a single cluster.

    Attributes:
        cluster_id: Unique identifier for the cluster
        size: Number of samples in the cluster
        centroid: Cluster centroid vector
        model_accuracies: Per-model accuracy scores for this cluster
    """

    cluster_id: int
    size: int
    centroid: np.ndarray
    model_accuracies: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ClusterInfo(id={self.cluster_id}, size={self.size}, "
            f"models={len(self.model_accuracies)})"
        )


@dataclass(frozen=True)
class ClusterMetrics:
    """Overall clustering metrics.

    Attributes:
        silhouette_score: Silhouette score (-1 to 1, higher is better)
        n_clusters: Number of clusters
        n_samples: Total number of samples
        cluster_sizes: List of cluster sizes
        inertia: Sum of squared distances to closest centroid (if applicable)
    """

    silhouette_score: float
    n_clusters: int
    n_samples: int
    cluster_sizes: list[int]
    inertia: float | None = None

    @property
    def min_cluster_size(self) -> int:
        """Minimum cluster size."""
        return min(self.cluster_sizes) if self.cluster_sizes else 0

    @property
    def max_cluster_size(self) -> int:
        """Maximum cluster size."""
        return max(self.cluster_sizes) if self.cluster_sizes else 0

    @property
    def avg_cluster_size(self) -> float:
        """Average cluster size."""
        if not self.cluster_sizes:
            return 0.0
        return sum(self.cluster_sizes) / len(self.cluster_sizes)

    def __repr__(self) -> str:
        return (
            f"ClusterMetrics(n_clusters={self.n_clusters}, "
            f"silhouette={self.silhouette_score:.3f}, "
            f"sizes={self.min_cluster_size}-{self.max_cluster_size})"
        )


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    inertia: float | None = None,
) -> ClusterMetrics:
    """Compute clustering metrics from embeddings and labels.

    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        labels: Cluster labels of shape (n_samples,)
        inertia: Optional inertia value from clustering algorithm

    Returns:
        ClusterMetrics object with computed metrics
    """
    from sklearn.metrics import silhouette_score as sk_silhouette_score

    n_samples = len(labels)
    unique_labels = np.unique(labels)

    # Filter out noise label (-1) for metrics
    valid_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(valid_labels)

    # Compute cluster sizes
    cluster_sizes = []
    for label in valid_labels:
        cluster_sizes.append(int(np.sum(labels == label)))

    # Compute silhouette score (requires at least 2 clusters)
    if n_clusters >= 2:
        # Only use samples with valid labels
        valid_mask = labels >= 0
        if valid_mask.sum() > n_clusters:
            silhouette = float(
                sk_silhouette_score(embeddings[valid_mask], labels[valid_mask])
            )
        else:
            silhouette = 0.0
    else:
        silhouette = 0.0

    return ClusterMetrics(
        silhouette_score=silhouette,
        n_clusters=n_clusters,
        n_samples=n_samples,
        cluster_sizes=cluster_sizes,
        inertia=inertia,
    )
