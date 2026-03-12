"""Clustering components for Router."""

from nordlys.clustering.agglomerative import AgglomerativeClusterer
from nordlys.clustering.base import Clusterer
from nordlys.clustering.bisecting import BisectingKMeansClusterer
from nordlys.clustering.gmm import GMMClusterer
from nordlys.clustering.hdbscan_clusterer import HDBSCANClusterer
from nordlys.clustering.kmeans import KMeansClusterer
from nordlys.clustering.minibatch import MiniBatchKMeansClusterer
from nordlys.clustering.metrics import (
    ClusterInfo,
    ClusterMetrics,
    compute_cluster_metrics,
)
from nordlys.clustering.spectral import SpectralClusterer
from nordlys.clustering.sweep import ParameterSweep, SweepResult, SweepResults

__all__ = [
    # Protocol
    "Clusterer",
    # Clusterers
    "KMeansClusterer",
    "MiniBatchKMeansClusterer",
    "BisectingKMeansClusterer",
    "HDBSCANClusterer",
    "GMMClusterer",
    "AgglomerativeClusterer",
    "SpectralClusterer",
    # Metrics
    "ClusterInfo",
    "ClusterMetrics",
    "compute_cluster_metrics",
    # Sweep
    "ParameterSweep",
    "SweepResult",
    "SweepResults",
]
