"""Clustering components for Nordlys."""

from nordlys.clustering._agglomerative import AgglomerativeClusterer
from nordlys.clustering._base import Clusterer
from nordlys.clustering._gmm import GMMClusterer
from nordlys.clustering._hdbscan import HDBSCANClusterer
from nordlys.clustering._kmeans import KMeansClusterer
from nordlys.clustering._metrics import ClusterInfo, ClusterMetrics, compute_cluster_metrics
from nordlys.clustering._spectral import SpectralClusterer
from nordlys.clustering._sweep import ParameterSweep, SweepResult, SweepResults

__all__ = [
    # Protocol
    "Clusterer",
    # Clusterers
    "KMeansClusterer",
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

