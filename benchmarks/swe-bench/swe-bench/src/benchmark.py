"""SWE-bench Nordlys Routing Benchmark orchestrator."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset
from nordlys_py import Nordlys, SelectModelRequest
from tqdm import tqdm

from .s3_fetcher import S3Fetcher

# Default Nordlys API base URL (can be overridden via NORDLYS_API_BASE env var)
DEFAULT_NORDLYS_BASE_URL = "https://api.nordlyslabs.com/v1"

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    output_dir: Path
    aws_region: str | None = None  # Uses AWS_DEFAULT_REGION if not set


@dataclass
class InstanceResult:
    """Result for a single instance."""

    instance_id: str
    model_id: str
    model_folder: str
    patch: str
    cost: float
    prompt_tokens: int
    completion_tokens: int
    resolved: bool | None
    success: bool
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Result of benchmark run."""

    total_instances: int
    patches_found: int
    patches_missing: int
    total_cost: float
    routing_distribution: dict[str, int]
    cost_by_model: dict[str, float]
    predictions_file: Path
    summary_file: Path
    instances: list[InstanceResult] = field(default_factory=list)


def build_model_folder_map() -> dict[str, str]:
    """Build model_id -> model_folder mapping.

    Maps from Nordlys model IDs to S3 folder names.
    Can be overridden via NORDLYS_MODEL_FOLDER_MAP environment variable (JSON string)
    or NORDLYS_MODEL_FOLDER_MAP_FILE environment variable (path to JSON file).

    Returns:
        Dict mapping model_id (provider/model_name) to S3 folder name
    """
    # Try loading from environment variable (JSON string)
    mapping_json = os.environ.get("NORDLYS_MODEL_FOLDER_MAP")
    if mapping_json:
        try:
            mapping = json.loads(mapping_json)
            logger.info(f"Loaded model folder mapping from env var ({len(mapping)} entries)")
            return mapping
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse NORDLYS_MODEL_FOLDER_MAP: {e}")

    # Try loading from config file
    config_file = os.environ.get("NORDLYS_MODEL_FOLDER_MAP_FILE")
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    mapping = json.load(f)
                logger.info(f"Loaded model folder mapping from {config_path} ({len(mapping)} entries)")
                return mapping
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")

    # Default fallback mapping
    mapping = {
        "anthropic/claude-sonnet-4-5": "20250929_mini-v1.13.3_sonnet-4-5-20250929",
        "anthropic/claude-opus-4-5": "20251124_mini-v1.16.0_claude-opus-4-5-20251101",
        "zhipu/glm4-6": "20250930_zai_glm4-6",
        "google/gemini-3-pro-preview": "20251118_mini-v1.15.0_gemini-3-pro-preview-20251118",
        "openai/gpt-5.1-codex": "20251124_mini-v1.16.0_gpt-5.1-codex",
        "openai/gpt-5.2": "20251211_mini-v1.17.2_gpt-5.2-2025-12-11",
        "deepseek/deepseek-v3.2": "20251201_mini-v1.17.1_deepseek-v3.2-reasoner",
        "qwen/qwen-3-coder": "20250802_mini-v1.0.0_qwen3-coder-480b-a35b-instruct",
    }

    logger.info(f"Using default model folder mapping ({len(mapping)} entries)")
    return mapping


def load_verified_instances() -> list[dict[str, Any]]:
    """Load 500 verified SWE-bench instances from HuggingFace.

    Returns:
        List of instance dicts with instance_id and problem_statement
    """
    logger.info("Loading SWE-bench verified instances from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    instances = []
    for item in dataset:
        instances.append(
            {
                "instance_id": item["instance_id"],
                "problem_statement": item["problem_statement"],
                "hints_text": item.get("hints_text", ""),
                "repo": item["repo"],
            }
        )

    logger.info(f"Loaded {len(instances)} verified instances")
    return instances


def prepare_routing_prompt(instance: dict[str, Any]) -> str:
    """Prepare prompt for Nordlys routing.

    Args:
        instance: Instance dict with problem_statement and hints_text

    Returns:
        Formatted prompt for routing
    """
    problem = instance["problem_statement"].strip()
    hints = instance.get("hints_text", "") or ""

    if hints.strip():
        combined = f"{problem}\n\nHints:\n{hints}"
    else:
        combined = problem

    return combined


class SWEBenchBenchmark:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config

        # Initialize Nordlys with base URL from env var or default
        base_url = os.environ.get("NORDLYS_API_BASE", DEFAULT_NORDLYS_BASE_URL)
        self.nordlys = Nordlys(base_url=base_url)

        self.s3_fetcher = S3Fetcher(region=config.aws_region)
        self.model_folder_map = build_model_folder_map()

        config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Nordlys router (base_url={base_url})")
        logger.info(f"Model folder mappings: {len(self.model_folder_map)}")

    def _route_instance(self, instance: dict[str, Any]) -> tuple[str, str | None]:
        """Route an instance to a model.

        Args:
            instance: Instance dict

        Returns:
            Tuple of (model_id, model_folder or None if not found)
        """
        prompt = prepare_routing_prompt(instance)
        request = SelectModelRequest(prompt=prompt)
        result = self.nordlys.router.select_model(request)

        model_id = result.selected_model or ""
        model_folder = self.model_folder_map.get(model_id)

        if not model_folder:
            logger.warning(f"No folder mapping for model {model_id}")

        return model_id, model_folder

    def run(self) -> BenchmarkResult:
        """Run the benchmark.

        Returns:
            BenchmarkResult with all metrics and outputs
        """
        instances = load_verified_instances()

        predictions = []
        instance_results = []
        routing_distribution: dict[str, int] = defaultdict(int)
        cost_by_model: dict[str, float] = defaultdict(float)
        total_cost = 0.0
        patches_found = 0
        patches_missing = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        logger.info(f"Running benchmark on {len(instances)} instances")

        for instance in tqdm(instances, desc="Processing instances"):
            instance_id = instance["instance_id"]

            # Route to model
            model_id, model_folder = self._route_instance(instance)
            routing_distribution[model_id] += 1

            if not model_folder:
                patches_missing += 1
                instance_results.append(
                    InstanceResult(
                        instance_id=instance_id,
                        model_id=model_id,
                        model_folder="",
                        patch="",
                        cost=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        resolved=None,
                        success=False,
                        error=f"No folder mapping for {model_id}",
                    )
                )
                continue

            # Fetch patch and cost from S3
            fetch_result = self.s3_fetcher.fetch_instance_data(model_folder, instance_id)

            if not fetch_result.success:
                patches_missing += 1
                instance_results.append(
                    InstanceResult(
                        instance_id=instance_id,
                        model_id=model_id,
                        model_folder=model_folder,
                        patch="",
                        cost=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        resolved=None,
                        success=False,
                        error=fetch_result.error,
                    )
                )
                continue

            data = fetch_result.data
            patches_found += 1
            total_cost += data.cost
            cost_by_model[model_id] += data.cost
            total_prompt_tokens += data.prompt_tokens
            total_completion_tokens += data.completion_tokens

            predictions.append(
                {
                    "instance_id": instance_id,
                    "model_name_or_path": "nordlys-routing",
                    "model_patch": data.patch,
                }
            )

            instance_results.append(
                InstanceResult(
                    instance_id=instance_id,
                    model_id=model_id,
                    model_folder=model_folder,
                    patch=data.patch,
                    cost=data.cost,
                    prompt_tokens=data.prompt_tokens,
                    completion_tokens=data.completion_tokens,
                    resolved=data.resolved,
                    success=True,
                )
            )

        # Write outputs
        predictions_path = self.config.output_dir / "all_preds.jsonl"
        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        summary = {
            "total_instances": len(instances),
            "patches_found": patches_found,
            "patches_missing": patches_missing,
            "routing_distribution": dict(routing_distribution),
            "total_cost": round(total_cost, 4),
            "cost_by_model": {k: round(v, 4) for k, v in cost_by_model.items()},
            "total_tokens": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
            },
            "instances": [
                {
                    "instance_id": r.instance_id,
                    "model_id": r.model_id,
                    "cost": r.cost,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "resolved": r.resolved,
                    "success": r.success,
                    "error": r.error,
                }
                for r in instance_results
            ],
        }

        summary_path = self.config.output_dir / "routing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Predictions written to {predictions_path}")
        logger.info(f"Summary written to {summary_path}")
        logger.info(f"Routing distribution: {dict(routing_distribution)}")
        logger.info(f"Patches found: {patches_found}/{len(instances)}")
        logger.info(f"Total cost: ${total_cost:.2f}")

        return BenchmarkResult(
            total_instances=len(instances),
            patches_found=patches_found,
            patches_missing=patches_missing,
            total_cost=total_cost,
            routing_distribution=dict(routing_distribution),
            cost_by_model=dict(cost_by_model),
            predictions_file=predictions_path,
            summary_file=summary_path,
            instances=instance_results,
        )
