"""Adaptive Router Benchmarking Suite.

This package provides benchmarking tools for evaluating adaptive router
performance on on-device AI workloads.
"""

__version__ = "0.1.0"

from benchmarks.core.routers import CactusProfileRouter, ClaudeOracleRouter
from benchmarks.core.runner import BenchmarkRunner
from benchmarks.core.metrics import BenchmarkMetrics

__all__ = [
    "CactusProfileRouter",
    "ClaudeOracleRouter",
    "BenchmarkRunner",
    "BenchmarkMetrics",
]
