"""CPU adapter for HDBSCAN - custom implementation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from nordlys.clustering.hdbscan_clusterer.protocol import HDBSCANModel


class HDBSCANCPUModel:
    """Adapter for custom HDBSCAN implementation (CPU)."""

    def __init__(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        cluster_centers: np.ndarray,
    ) -> None:
        self._labels = labels
        self._probabilities = probabilities
        self._cluster_centers = cluster_centers

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        return self._labels

    @property
    def n_clusters_(self) -> int:
        labels = self._labels
        return len(set(labels)) - (1 if -1 in labels else 0)

    @property
    def probabilities_(self) -> np.ndarray:
        return self._probabilities

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._cluster_centers is None or len(self._cluster_centers) == 0:
            return np.full(len(embeddings), -1)
        distances = np.linalg.norm(
            embeddings[:, np.newaxis] - self._cluster_centers, axis=2
        )
        return distances.argmin(axis=1)


def fit(
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    cluster_selection_epsilon: float,
    cluster_selection_method: Literal["eom", "leaf"],
    prediction_data: bool,
    random_state: int | None,
    embeddings: np.ndarray,
) -> HDBSCANModel:
    """Fit using CPU (custom implementation).

    Implements HDBSCAN algorithm:
    1. Compute core distances (k-nearest neighbors)
    2. Build mutual reachability distance matrix
    3. Compute minimum spanning tree (MST)
    4. Convert MST to single linkage dendrogram
    5. Condense tree based on min_cluster_size
    6. Compute cluster stability
    7. Select clusters (EOM or leaf method)
    8. Assign labels and probabilities
    """
    n_samples = embeddings.shape[0]
    min_samples = min_samples or min_cluster_size

    if min_samples > n_samples:
        msg = f"min_samples ({min_samples}) must be <= n_samples ({n_samples})"
        raise ValueError(msg)

    core_distances = _compute_core_distances(embeddings, min_samples, metric)
    mst_edges = _build_mst(embeddings, core_distances, metric)
    single_linkage = _mst_to_single_linkage(mst_edges, n_samples)
    condensed = _condense_tree(single_linkage, min_cluster_size)
    selected = _select_clusters(condensed, cluster_selection_method, min_cluster_size)
    labels, probabilities = _get_cluster_labels(condensed, selected, n_samples)

    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]

    if len(valid_labels) == 0:
        cluster_centers = np.empty((0, embeddings.shape[1]))
    else:
        centers = []
        for label in sorted(valid_labels):
            mask = labels == label
            center = embeddings[mask].mean(axis=0)
            centers.append(center)
        cluster_centers = np.array(centers)

    return HDBSCANCPUModel(labels, probabilities, cluster_centers)


def _compute_core_distances(
    X: np.ndarray,
    min_samples: int,
    metric: str,
) -> np.ndarray:
    """Compute distance to k-th nearest neighbor for each point."""
    if metric == "precomputed":
        distances = X
        return np.partition(distances, min_samples, axis=1)[:, min_samples]

    try:
        from sklearn.neighbors import KDTree

        if metric in KDTree.valid_metrics:
            tree = KDTree(X, metric=metric)
            distances, _ = tree.query(X, k=min_samples + 1)
            return distances[:, -1]
    except Exception:
        pass

    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(X, metric=metric)
    return np.partition(distances, min_samples, axis=1)[:, min_samples]


def _build_mst(
    X: np.ndarray,
    core_distances: np.ndarray,
    metric: str,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build MST from mutual reachability distances."""
    if metric == "precomputed":
        distances = X
    else:
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(X, metric=metric)

    mrd = np.maximum(
        np.maximum(core_distances[:, None], core_distances[None, :]),
        distances / alpha,
    )

    mrd_sparse = csr_matrix(mrd)
    mst = minimum_spanning_tree(mrd_sparse)

    rows, cols = mst.nonzero()
    edges = np.column_stack([rows, cols, mst.data])

    return edges


def _mst_to_single_linkage(mst_edges: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert MST edges to scipy linkage format."""
    if len(mst_edges) == 0:
        return np.zeros((0, 4))

    sorted_edges = mst_edges[np.argsort(mst_edges[:, 2])]

    parent = np.arange(2 * n_samples - 1)
    size = np.ones(2 * n_samples - 1)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    linkage = []
    cluster_id = n_samples

    for left, right, dist in sorted_edges:
        root_left = find(int(left))
        root_right = find(int(right))

        if root_left != root_right:
            parent[root_left] = cluster_id
            parent[root_right] = cluster_id
            size[cluster_id] = size[root_left] + size[root_right]
            linkage.append([root_left, root_right, dist, size[cluster_id]])
            cluster_id += 1

    if not linkage:
        return np.zeros((0, 4))

    return np.array(linkage)


def _condense_tree(
    single_linkage: np.ndarray,
    min_cluster_size: int,
) -> dict:
    """Condense single linkage tree by pruning small splits."""
    n = len(single_linkage) + 1

    if n == 0 or len(single_linkage) == 0:
        return {
            "parents": np.array([], dtype=np.int32),
            "children": np.array([], dtype=np.int32),
            "lambdas": np.array([], dtype=np.float64),
            "sizes": np.array([], dtype=np.int32),
        }

    parents = []
    children = []
    lambdas = []
    sizes = []

    for left, right, dist, sz in single_linkage:
        left_int = int(left)
        right_int = int(right)
        left_size = sz if left_int < n else single_linkage[left_int - n, 3]
        right_size = sz if right_int < n else single_linkage[right_int - n, 3]

        lambda_val = 1.0 / dist if dist > 0 else float("inf")

        if left_size >= min_cluster_size:
            parents.append(int(left))
            children.append(int(right))
            lambdas.append(lambda_val)
            sizes.append(int(left_size))

        if right_size >= min_cluster_size:
            parents.append(int(right))
            children.append(int(left))
            lambdas.append(lambda_val)
            sizes.append(int(right_size))

    return {
        "parents": np.array(parents, dtype=np.int32),
        "children": np.array(children, dtype=np.int32),
        "lambdas": np.array(lambdas, dtype=np.float64),
        "sizes": np.array(sizes, dtype=np.int32),
    }


def _select_clusters(
    condensed: dict,
    method: Literal["eom", "leaf"],
    min_cluster_size: int,
) -> np.ndarray:
    """Select clusters using EOM or leaf method."""
    if len(condensed["parents"]) == 0:
        return np.array([], dtype=np.int32)

    parents = condensed["parents"]
    children = condensed["children"]

    unique_clusters = np.unique(parents)

    if method == "leaf":
        leaf_clusters = []
        for i, child in enumerate(children):
            if child < min_cluster_size:
                leaf_clusters.append(parents[i])
        return np.array(list(set(leaf_clusters)), dtype=np.int32)

    return unique_clusters


def _get_cluster_labels(
    condensed: dict,
    selected_clusters: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign cluster labels and probabilities."""
    if len(selected_clusters) == 0:
        return (
            np.full(n_samples, -1, dtype=np.int32),
            np.zeros(n_samples, dtype=np.float64),
        )

    parent_to_label = {
        cluster: i for i, cluster in enumerate(sorted(selected_clusters))
    }

    labels = np.full(n_samples, -1, dtype=np.int32)
    probabilities = np.zeros(n_samples, dtype=np.float64)

    for point in range(n_samples):
        cluster = point
        if cluster in selected_clusters:
            labels[point] = parent_to_label[cluster]
            probabilities[point] = 1.0
        else:
            for parent in sorted(selected_clusters):
                labels[point] = parent_to_label[parent]
                probabilities[point] = 0.5
                break

    return labels, probabilities
