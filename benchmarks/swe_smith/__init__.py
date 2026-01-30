"""SWE-smith Cluster Evaluation Pipeline for Nordlys model routing.

This module provides tools for evaluating models on SWE-smith instances
using the Doubleword batch API for one-shot patch generation, with
per-cluster error rates for Nordlys model routing.

## Workflow

1. Sample instances from clusters:
   ```
   python -m benchmarks.swe_smith.scripts.sample_instances \
       --output results/gpt4o/sampled_ids.json \
       --cluster-map results/gpt4o/cluster_map.json
   ```

2. Pre-fetch file contents:
   ```
   python -m benchmarks.swe_smith.scripts.prefetch_files \
       --input results/gpt4o/sampled_ids.json \
       --output results/gpt4o/file_contents.jsonl
   ```

3. Prepare batch:
   ```
   python -m benchmarks.swe_smith.scripts.prepare_batch \
       --file-contents results/gpt4o/file_contents.jsonl \
       --model "gpt-4o" \
       --output results/gpt4o/batch_input.jsonl
   ```

4. Submit to Doubleword:
   ```
   python -m benchmarks.swe_smith.scripts.submit_batch \
       --input results/gpt4o/batch_input.jsonl
   ```

5. Collect results:
   ```
   python -m benchmarks.swe_smith.scripts.collect_results \
       --batch-id batch_abc123 \
       --output results/gpt4o/predictions.jsonl
   ```

6. Run swesmith evaluation:
   ```
   python -m swesmith.harness.eval \
       -p results/gpt4o/predictions.jsonl \
       --run_id "gpt4o-eval" -w 10
   ```

7. Build profile:
   ```
   python -m benchmarks.swe_smith.scripts.build_profile \
       --report logs/run_evaluation/gpt4o-eval/report.json \
       --cluster-map results/gpt4o/cluster_map.json \
       --model "gpt-4o"
   ```

See README.md for detailed documentation.
"""

# Lazy imports - config values are lightweight
from swe_smith.config import (
    CLUSTER_ASSIGNMENTS,
    CLUSTERS_DIR,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    DOUBLEWORD_API_BASE,
    MIN_SAMPLES,
    N_CLUSTERS,
    ONE_SHOT_SYSTEM_PROMPT,
    ONE_SHOT_USER_PROMPT,
    SAMPLE_FRACTION,
)

__all__ = [
    # Config
    "CLUSTER_ASSIGNMENTS",
    "CLUSTERS_DIR",
    "DEFAULT_TIMEOUT",
    "DEFAULT_WORKERS",
    "DOUBLEWORD_API_BASE",
    "MIN_SAMPLES",
    "N_CLUSTERS",
    "ONE_SHOT_SYSTEM_PROMPT",
    "ONE_SHOT_USER_PROMPT",
    "SAMPLE_FRACTION",
    # Classes (lazy loaded)
    "ClusterProfiler",
    "ClusterSampler",
    "DoublewordBatchRunner",
    "FileFetcher",
]


def __getattr__(name: str):
    """Lazy import for classes that require external dependencies."""
    if name == "ClusterSampler":
        from swe_smith.sampler import ClusterSampler

        return ClusterSampler
    if name == "ClusterProfiler":
        from swe_smith.profiler import ClusterProfiler

        return ClusterProfiler
    if name == "DoublewordBatchRunner":
        from swe_smith.batch_runner import DoublewordBatchRunner

        return DoublewordBatchRunner
    if name == "FileFetcher":
        from swe_smith.file_fetcher import FileFetcher

        return FileFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
