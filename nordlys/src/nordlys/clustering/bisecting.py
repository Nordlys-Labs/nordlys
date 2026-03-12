"""Bisecting K-Means clusterer."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from nordlys.clustering.base import Clusterer


class BisectingKMeansClusterer(Clusterer):
    """Bisecting K-Means clustering wrapper.

    Bisecting K-Means works by repeatedly splitting the largest cluster
    into two until the desired number of clusters is reached.

    This implementation uses MiniBatchKMeans for efficiency on large datasets.

    Example:
        >>> clusterer = BisectingKMeansClusterer(n_clusters=20)
        >>> clusterer.fit(embeddings)
        >>> labels = clusterer.predict(new_embeddings)
    """

    def __init__(
        self,
        n_clusters: int = 20,
        max_iter: int = 100,
        batch_size: int = 1024,
        random_state: int = 42,
        init_size: int | None = None,
        max_no_improvement: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize BisectingKMeans clusterer.

        Args:
            n_clusters: Number of clusters (default: 20)
            max_iter: Maximum iterations per split (default: 100)
            batch_size: Size of mini-batches (default: 1024)
            random_state: Random seed for reproducibility (default: 42)
            init_size: Number of samples for initialization (default: 3 * batch_size)
            max_no_improvement: Stop if no improvement for N iterations (default: None)
            **kwargs: Additional arguments passed to MiniBatchKMeans
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.init_size = init_size
        self.max_no_improvement = max_no_improvement
        self._kwargs = kwargs
        self._model: MiniBatchKMeans | None = None
        self._labels: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "BisectingKMeansClusterer":
        """Fit the clusterer on embeddings using bisecting K-Means.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Self
        """
        n_samples = embeddings.shape[0]

        if self.n_clusters <= 1:
            # Single cluster - all points belong to cluster 0
            self._labels = np.zeros(n_samples, dtype=np.int64)
            # Create a dummy centroid
            self._cluster_centers = embeddings.mean(axis=0, keepdims=True)
            return self

        # Start with all points in one cluster
        cluster_assignments = np.zeros(n_samples, dtype=np.int64)
        cluster_sizes = [n_samples]

        # List to store centroids
        centroids = [embeddings.mean(axis=0)]

        # Bisect until we have n_clusters clusters
        while len(centroids) < self.n_clusters:
            # Find the largest cluster to split
            largest_cluster_idx = np.argmax(cluster_sizes)
            largest_cluster_size = cluster_sizes[largest_cluster_idx]

            if largest_cluster_size < 2:
                # Can't split a cluster of size 1
                break

            # Get indices of points in the largest cluster
            cluster_indices = np.where(cluster_assignments == largest_cluster_idx)[0]
            cluster_data = embeddings[cluster_indices]

            # Split with MiniBatchKMeans (2 clusters)
            split_model = MiniBatchKMeans(
                n_clusters=2,
                max_iter=self.max_iter,
                batch_size=min(self.batch_size, len(cluster_data)),
                random_state=self.random_state,
                init_size=self.init_size,
                max_no_improvement=self.max_no_improvement,
                **self._kwargs,
            )
            split_model.fit(cluster_data)
            split_labels = split_model.predict(cluster_data)

            # Update assignments: cluster_indices[split_labels == 0] keep original idx
            # cluster_indices[split_labels == 1] get new cluster idx
            new_cluster_idx = len(centroids)

            # Update cluster_assignments
            for i, global_idx in enumerate(cluster_indices):
                if split_labels[i] == 1:
                    cluster_assignments[global_idx] = new_cluster_idx

            # Update cluster sizes
            split_size_0 = np.sum(split_labels == 0)
            split_size_1 = np.sum(split_labels == 1)
            cluster_sizes[largest_cluster_idx] = split_size_0
            cluster_sizes.append(split_size_1)

            # Update centroids
            # For the split cluster, recalculate centroid
            new_centroids = []
            for c_idx in range(len(centroids)):
                if c_idx == largest_cluster_idx:
                    # Recalculate centroid for the split cluster
                    mask = cluster_assignments == c_idx
                    if np.sum(mask) > 0:
                        new_centroids.append(embeddings[mask].mean(axis=0))
                    else:
                        new_centroids.append(centroids[c_idx])
                else:
                    new_centroids.append(centroids[c_idx])

            # Add centroid for the new cluster
            mask = cluster_assignments == new_cluster_idx
            if np.sum(mask) > 0:
                new_centroids.append(embeddings[mask].mean(axis=0))
            else:
                # Use the split model's centroid
                new_centroids.append(split_model.cluster_centers_[1])

            centroids = new_centroids

        # Store results
        self._labels = cluster_assignments
        self._cluster_centers = np.stack(centroids)

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for embeddings using nearest centroid.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if self._model is not None:
            # If model is set (shouldn't happen for bisecting), use it
            return self._model.predict(embeddings)

        # Nearest centroid assignment
        # Calculate distances to all centroids
        distances = np.sqrt(
            np.sum(
                (embeddings[:, np.newaxis, :] - self._cluster_centers[np.newaxis, :, :])
                ** 2,
                axis=2,
            )
        )
        return np.argmin(distances, axis=1)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the clusterer and predict cluster assignments.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            Cluster assignments of shape (n_samples,)
        """
        self.fit(embeddings)
        assert self._labels is not None
        return self._labels

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Cluster centers of shape (n_clusters, n_features)."""
        return self._cluster_centers

    @property
    def labels_(self) -> np.ndarray:
        """Labels assigned during fit() of shape (n_samples,)."""
        if self._labels is None:
            raise RuntimeError("Clusterer must be fitted first.")
        return self._labels

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found."""
        return self.n_clusters

    def __repr__(self) -> str:
        return f"BisectingKMeansClusterer(n_clusters={self.n_clusters})"
