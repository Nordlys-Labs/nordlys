"""Fetch patches and cost data from SWE-bench S3 bucket."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds

# S3 configuration (can be overridden via environment variables)
S3_BUCKET = os.environ.get("SWE_BENCH_S3_BUCKET", "swe-bench-experiments")
S3_BASE_PATH = os.environ.get("SWE_BENCH_S3_PATH", "bash-only")


@dataclass
class InstanceData:
    """Data for a single SWE-bench instance."""

    instance_id: str
    patch: str
    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    resolved: bool | None = None


@dataclass
class FetchResult:
    """Result of fetching instance data."""

    success: bool
    data: InstanceData | None = None
    error: str | None = None


class S3Fetcher:
    """Fetch patches and costs from SWE-bench S3 bucket."""

    def __init__(self, region: str | None = None):
        """Initialize S3 fetcher.

        Args:
            region: AWS region (uses AWS_DEFAULT_REGION env var if not provided)
        """
        # Configure S3 client with longer timeouts and retries
        config = Config(
            connect_timeout=30,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        # Let boto3 handle credentials via standard chain
        # (env vars, ~/.aws/credentials, IAM roles, etc.)
        self.s3 = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_DEFAULT_REGION", "eu-west-2"),
            config=config,
        )
        self._cache: dict[tuple[str, str], InstanceData] = {}
        self._cache_lock = Lock()

    def _get_patch_path(self, model_folder: str, instance_id: str) -> str:
        """Get S3 path for patch file."""
        return f"{S3_BASE_PATH}/{model_folder}/logs/{instance_id}/patch.diff"

    def _get_traj_path(self, model_folder: str, instance_id: str) -> str:
        """Get S3 path for trajectory file."""
        return f"{S3_BASE_PATH}/{model_folder}/trajs/{instance_id}/{instance_id}.traj.json"

    def _get_report_path(self, model_folder: str, instance_id: str) -> str:
        """Get S3 path for report file."""
        return f"{S3_BASE_PATH}/{model_folder}/logs/{instance_id}/report.json"

    def _fetch_s3_object(self, key: str) -> bytes | None:
        """Fetch object from S3 with retry logic."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.s3.get_object(Bucket=S3_BUCKET, Key=key)
                return response["Body"].read()
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    return None
                last_error = e
                logger.warning(f"S3 ClientError on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            except ReadTimeoutError as e:
                last_error = e
                logger.warning(f"S3 ReadTimeout on attempt {attempt + 1}/{MAX_RETRIES}: {key}")
            except Exception as e:
                last_error = e
                logger.warning(f"S3 error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")

            # Exponential backoff before retry
            if attempt < MAX_RETRIES - 1:
                backoff = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                logger.info(f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

        # All retries exhausted
        logger.error(f"Failed to fetch {key} after {MAX_RETRIES} attempts")
        if last_error:
            raise last_error
        return None

    def fetch_instance_data(
        self, model_folder: str, instance_id: str
    ) -> FetchResult:
        """Fetch patch and cost data for an instance.

        Args:
            model_folder: S3 folder name for the model
            instance_id: SWE-bench instance ID

        Returns:
            FetchResult with success status and data or error
        """
        cache_key = (model_folder, instance_id)
        with self._cache_lock:
            if cache_key in self._cache:
                return FetchResult(success=True, data=self._cache[cache_key])

        try:
            # Fetch patch
            patch_key = self._get_patch_path(model_folder, instance_id)
            patch_content = self._fetch_s3_object(patch_key)
            if patch_content is None:
                return FetchResult(
                    success=False,
                    error=f"Patch not found: s3://{S3_BUCKET}/{patch_key}",
                )
            patch = patch_content.decode("utf-8")

            # Fetch trajectory for cost data
            traj_key = self._get_traj_path(model_folder, instance_id)
            traj_content = self._fetch_s3_object(traj_key)

            cost = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if traj_content:
                try:
                    traj = json.loads(traj_content.decode("utf-8"))
                    model_stats = traj.get("info", {}).get("model_stats", {})
                    cost = model_stats.get("instance_cost", 0.0)
                    prompt_tokens = model_stats.get("prompt_tokens", 0)
                    completion_tokens = model_stats.get("completion_tokens", 0)
                    total_tokens = model_stats.get("total_tokens", 0)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse traj for {instance_id}: {e}")

            # Fetch report for resolved status
            resolved = None
            report_key = self._get_report_path(model_folder, instance_id)
            report_content = self._fetch_s3_object(report_key)
            if report_content:
                try:
                    report = json.loads(report_content.decode("utf-8"))
                    resolved = report.get("resolved", None)
                except (json.JSONDecodeError, KeyError):
                    pass

            data = InstanceData(
                instance_id=instance_id,
                patch=patch,
                cost=cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                resolved=resolved,
            )
            with self._cache_lock:
                self._cache[cache_key] = data
            return FetchResult(success=True, data=data)

        except Exception as e:
            logger.error(f"Failed to fetch data for {instance_id}: {e}")
            return FetchResult(
                success=False,
                error=f"S3 fetch failed after retries: {str(e)}",
            )

    def batch_fetch(
        self, model_folder: str, instance_ids: list[str], max_workers: int = 10
    ) -> dict[str, FetchResult]:
        """Fetch data for multiple instances concurrently.

        Args:
            model_folder: S3 folder name for the model
            instance_ids: List of instance IDs
            max_workers: Maximum number of concurrent fetches (default: 10)

        Returns:
            Dict mapping instance_id to FetchResult
        """
        results: dict[str, FetchResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.fetch_instance_data, model_folder, iid): iid
                for iid in instance_ids
            }
            for future in as_completed(future_to_id):
                instance_id = future_to_id[future]
                try:
                    results[instance_id] = future.result()
                except Exception as e:
                    logger.error(f"Failed to fetch {instance_id}: {e}")
                    results[instance_id] = FetchResult(
                        success=False,
                        error=f"Concurrent fetch failed: {str(e)}",
                    )

        return results
