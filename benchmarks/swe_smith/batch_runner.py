"""Doubleword Batch API client for SWE-smith evaluation.

This module provides a client for submitting batch inference requests to
the Doubleword API (OpenAI-compatible) and processing the results.

The batch approach enables cost-effective evaluation of large numbers of
SWE-smith instances using one-shot patch generation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swe_smith.config import (
    AUTOBATCH_POLL_INTERVAL,
    AUTOBATCH_SIZE,
    AUTOBATCH_WINDOW_SECONDS,
    DOUBLEWORD_API_BASE,
    DOUBLEWORD_COMPLETION_WINDOW,
    FILE_CONTENT_TEMPLATE,
    GENERATION_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    ONE_SHOT_SYSTEM_PROMPT,
    ONE_SHOT_USER_PROMPT,
)

if TYPE_CHECKING:
    from openai.types import Batch, FileObject


@dataclass
class BatchStatus:
    """Status of a batch job."""

    id: str
    status: str
    total_requests: int
    completed_requests: int
    failed_requests: int
    output_file_id: str | None
    error_file_id: str | None
    created_at: int
    completed_at: int | None

    @classmethod
    def from_batch(cls, batch: "Batch") -> "BatchStatus":
        """Create from OpenAI Batch object."""
        counts = batch.request_counts or {}
        return cls(
            id=batch.id,
            status=batch.status,
            total_requests=counts.total if hasattr(counts, "total") else 0,
            completed_requests=counts.completed if hasattr(counts, "completed") else 0,
            failed_requests=counts.failed if hasattr(counts, "failed") else 0,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            created_at=batch.created_at,
            completed_at=getattr(batch, "completed_at", None),
        )

    @property
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if batch has failed."""
        return self.status in ("failed", "expired", "cancelled")

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.completed_requests / self.total_requests) * 100


class DoublewordBatchRunner:
    """Client for Doubleword batch API operations.

    Uses the OpenAI-compatible batch API to submit and retrieve
    inference results for SWE-smith instances.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = DOUBLEWORD_API_BASE,
        completion_window: str = DOUBLEWORD_COMPLETION_WINDOW,
    ):
        """Initialize the batch runner.

        Args:
            api_key: Doubleword API key.
            model: Model to use for inference.
            base_url: Base URL for the API.
            completion_window: Batch completion window ("24h" or "1h").
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.completion_window = completion_window

    def _format_prompt(
        self,
        instance: dict,
        file_data: dict | None = None,
    ) -> str:
        """Format the user prompt for an instance.

        Args:
            instance: SWE-smith instance dict with problem_statement.
            file_data: Optional pre-fetched file data with repo_structure and file_contents.

        Returns:
            Formatted user prompt string.
        """
        problem_statement = instance.get("problem_statement", "")

        # Format repository structure
        repo_structure = ""
        file_contents = ""

        if file_data:
            # Use pre-fetched data
            if "repo_structure" in file_data:
                repo_structure = "\n".join(file_data["repo_structure"][:50])

            if "file_contents" in file_data:
                file_sections = []
                for file_path, content in file_data["file_contents"].items():
                    # Detect language from extension
                    ext = Path(file_path).suffix.lstrip(".")
                    lang_map = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "rb": "ruby",
                        "go": "go",
                        "rs": "rust",
                        "java": "java",
                        "cpp": "cpp",
                        "c": "c",
                        "h": "c",
                    }
                    language = lang_map.get(ext, ext)

                    file_sections.append(
                        FILE_CONTENT_TEMPLATE.format(
                            file_path=file_path,
                            language=language,
                            content=content,
                        )
                    )
                file_contents = "\n\n".join(file_sections)

        return ONE_SHOT_USER_PROMPT.format(
            problem_statement=problem_statement,
            repo_structure=repo_structure or "(No structure available)",
            file_contents=file_contents or "(No file contents available)",
        )

    def create_batch_jsonl(
        self,
        instances: list[dict],
        file_data_map: dict[str, dict],
        output_path: Path,
    ) -> int:
        """Create JSONL file for batch submission.

        Args:
            instances: List of SWE-smith instance dicts.
            file_data_map: Mapping from instance_id to pre-fetched file data.
            output_path: Path to write the JSONL file.

        Returns:
            Number of requests written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for instance in instances:
                instance_id = instance["instance_id"]
                file_data = file_data_map.get(instance_id)

                request = {
                    "custom_id": instance_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": ONE_SHOT_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": self._format_prompt(instance, file_data),
                            },
                        ],
                        "max_tokens": MAX_OUTPUT_TOKENS,
                        "temperature": GENERATION_TEMPERATURE,
                    },
                }
                f.write(json.dumps(request) + "\n")
                count += 1

        return count

    def upload_batch_file(self, jsonl_path: Path) -> "FileObject":
        """Upload a batch input file.

        Args:
            jsonl_path: Path to the JSONL file.

        Returns:
            FileObject with the uploaded file ID.
        """
        with open(jsonl_path, "rb") as f:
            return self.client.files.create(file=f, purpose="batch")

    def submit_batch(self, jsonl_path: Path) -> str:
        """Upload file and create batch job.

        Args:
            jsonl_path: Path to the batch JSONL file.

        Returns:
            Batch ID for tracking.
        """
        # Step 1: Upload file
        batch_file = self.upload_batch_file(jsonl_path)
        print(f"Uploaded batch file: {batch_file.id}")

        # Step 2: Create batch
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window,
        )
        print(f"Created batch: {batch.id}")

        return batch.id

    def check_status(self, batch_id: str) -> BatchStatus:
        """Check batch status.

        Args:
            batch_id: Batch ID to check.

        Returns:
            BatchStatus with current status information.
        """
        batch = self.client.batches.retrieve(batch_id)
        return BatchStatus.from_batch(batch)

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout: int | None = None,
    ) -> BatchStatus:
        """Wait for batch to complete.

        Args:
            batch_id: Batch ID to wait for.
            poll_interval: Seconds between status checks.
            timeout: Optional timeout in seconds.

        Returns:
            Final BatchStatus.

        Raises:
            TimeoutError: If timeout is exceeded.
            RuntimeError: If batch fails.
        """
        start_time = time.time()

        while True:
            status = self.check_status(batch_id)

            print(
                f"[{time.strftime('%H:%M:%S')}] Status: {status.status} "
                f"({status.completed_requests}/{status.total_requests})"
            )

            if status.is_complete:
                return status

            if status.is_failed:
                raise RuntimeError(f"Batch failed with status: {status.status}")

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Batch did not complete within {timeout}s")

            time.sleep(poll_interval)

    def download_results(self, batch_id: str) -> list[dict]:
        """Download completed results and format for swesmith.

        Args:
            batch_id: Batch ID to download results for.

        Returns:
            List of prediction dicts with instance_id, model_patch, model_name_or_path.

        Raises:
            ValueError: If batch is not complete.
        """
        status = self.check_status(batch_id)
        if not status.is_complete:
            raise ValueError(f"Batch not complete: {status.status}")

        if not status.output_file_id:
            raise ValueError("No output file available")

        # Download output file
        response = self.client.files.content(status.output_file_id)

        # Parse JSONL results
        predictions = []
        for line in response.text.strip().split("\n"):
            if not line.strip():
                continue

            result = json.loads(line)
            instance_id = result["custom_id"]

            # Extract response content
            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                # Convert SEARCH/REPLACE to unified diff
                patch = self._search_replace_to_diff(content, instance_id)
            except (KeyError, IndexError) as e:
                print(f"Warning: Failed to extract response for {instance_id}: {e}")
                patch = ""

            predictions.append(
                {
                    "instance_id": instance_id,
                    "model_patch": patch,  # KEY_PREDICTION from swebench
                    "model_name_or_path": self.model,  # KEY_MODEL from swebench
                }
            )

        return predictions

    def download_errors(self, batch_id: str) -> list[dict]:
        """Download error results if any.

        Args:
            batch_id: Batch ID to download errors for.

        Returns:
            List of error dicts.
        """
        status = self.check_status(batch_id)

        if not status.error_file_id:
            return []

        response = self.client.files.content(status.error_file_id)

        errors = []
        for line in response.text.strip().split("\n"):
            if line.strip():
                errors.append(json.loads(line))

        return errors

    def _search_replace_to_diff(self, content: str, instance_id: str) -> str:
        """Convert SEARCH/REPLACE blocks to unified diff format.

        Args:
            content: Model response containing SEARCH/REPLACE blocks.
            instance_id: Instance ID for context.

        Returns:
            Unified diff string compatible with git apply.
        """
        # Pattern to match file path and SEARCH/REPLACE blocks
        # Format: ### path/to/file.py\n<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE
        block_pattern = re.compile(
            r"###\s*([^\n]+)\n"  # File path
            r"<<<<<<< SEARCH\n"
            r"(.*?)\n"  # Search content
            r"=======\n"
            r"(.*?)\n"  # Replace content
            r">>>>>>> REPLACE",
            re.DOTALL,
        )

        # Find all edit blocks
        matches = block_pattern.findall(content)

        if not matches:
            # Try alternative pattern without file header per block
            alt_pattern = re.compile(
                r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
                re.DOTALL,
            )
            alt_matches = alt_pattern.findall(content)

            if not alt_matches:
                # No valid blocks found, return empty
                return ""

            # For alternative format, try to extract file from content context
            # This is a fallback - ideally blocks should have file paths
            print(
                f"Warning: {instance_id} has SEARCH/REPLACE without file paths, "
                "attempting to generate patch anyway"
            )
            # Return raw content as a best-effort patch
            return content

        # Group edits by file
        file_edits: dict[str, list[tuple[str, str]]] = {}
        for file_path, search, replace in matches:
            file_path = file_path.strip()
            if file_path not in file_edits:
                file_edits[file_path] = []
            file_edits[file_path].append((search, replace))

        # Generate unified diff for each file
        diff_parts = []
        for file_path, edits in file_edits.items():
            # Create header
            diff_parts.append(f"diff --git a/{file_path} b/{file_path}")
            diff_parts.append(f"--- a/{file_path}")
            diff_parts.append(f"+++ b/{file_path}")

            # Generate hunks for each edit
            for search, replace in edits:
                # Split into lines
                search_lines = search.split("\n")
                replace_lines = replace.split("\n")

                # Use a placeholder line number since we don't know actual locations
                # The swesmith harness will apply these based on content matching
                search_count = len(search_lines)
                replace_count = len(replace_lines)

                diff_parts.append(f"@@ -1,{search_count} +1,{replace_count} @@")

                # Add search lines with - prefix
                for line in search_lines:
                    diff_parts.append(f"-{line}")

                # Add replace lines with + prefix
                for line in replace_lines:
                    diff_parts.append(f"+{line}")

        return "\n".join(diff_parts)


def save_predictions_jsonl(predictions: list[dict], output_path: Path) -> None:
    """Save predictions in JSONL format for swesmith evaluation.

    Args:
        predictions: List of prediction dicts.
        output_path: Path to write JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def load_predictions_jsonl(input_path: Path) -> list[dict]:
    """Load predictions from JSONL file.

    Args:
        input_path: Path to JSONL file.

    Returns:
        List of prediction dicts.
    """
    predictions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def append_predictions_jsonl(predictions: list[dict], output_path: Path) -> None:
    """Append predictions to a JSONL file.

    Args:
        predictions: List of prediction dicts to append.
        output_path: Path to the JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


class AutobatchRunner(DoublewordBatchRunner):
    """Async batch runner using the autobatcher SDK.

    Uses ``BatchOpenAI`` from the ``autobatcher`` package to automatically
    collect async ``chat.completions.create()`` calls into OpenAI-compatible
    batches, handling JSONL creation, file upload, polling, and response
    resolution transparently.

    This replaces the manual workflow of:
        1. create_batch_jsonl()
        2. submit_batch()
        3. wait_for_completion()
        4. download_results()

    With a single call to :meth:`generate_all_patches`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = DOUBLEWORD_API_BASE,
        batch_size: int = AUTOBATCH_SIZE,
        batch_window_seconds: float = AUTOBATCH_WINDOW_SECONDS,
        poll_interval_seconds: float = AUTOBATCH_POLL_INTERVAL,
        completion_window: str = DOUBLEWORD_COMPLETION_WINDOW,
    ):
        """Initialize the autobatch runner.

        Args:
            api_key: Doubleword API key.
            model: Model to use for inference.
            base_url: Base URL for the API.
            batch_size: Number of requests before auto-submitting a batch.
            batch_window_seconds: Seconds of inactivity before auto-submitting.
            poll_interval_seconds: Seconds between batch status polls.
            completion_window: Batch completion window ("24h" or "1h").
        """
        # We intentionally skip super().__init__() because AutobatchRunner
        # uses the async BatchOpenAI client instead of the sync OpenAI client.
        from autobatcher import BatchOpenAI

        self.batch_client = BatchOpenAI(
            api_key=api_key,
            base_url=base_url,
            batch_size=batch_size,
            batch_window_seconds=batch_window_seconds,
            poll_interval_seconds=poll_interval_seconds,
            completion_window=completion_window,
        )
        self.model = model
        self.completion_window = completion_window

    async def generate_patch(
        self,
        instance: dict,
        file_data: dict | None = None,
    ) -> dict:
        """Generate a patch for a single instance via autobatcher.

        The autobatcher transparently collects this call with others
        into a batch and resolves the future when results are ready.

        Args:
            instance: SWE-smith instance dict with problem_statement.
            file_data: Optional pre-fetched file data.

        Returns:
            Prediction dict with instance_id, model_patch, model_name_or_path.
        """
        instance_id = instance["instance_id"]
        user_prompt = self._format_prompt(instance, file_data)

        try:
            response = await self.batch_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ONE_SHOT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=GENERATION_TEMPERATURE,
            )
            content = response.choices[0].message.content or ""
            patch = self._search_replace_to_diff(content, instance_id)
        except Exception as e:
            print(f"Warning: Failed to generate patch for {instance_id}: {e}")
            patch = ""

        return {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": self.model,
        }

    async def generate_all_patches(
        self,
        instances: list[dict],
        file_data_map: dict[str, dict] | None = None,
        output_path: Path | None = None,
    ) -> list[dict]:
        """Generate patches for all instances using autobatcher.

        All requests are submitted as async calls. The autobatcher SDK
        automatically groups them into batches based on batch_size and
        batch_window_seconds, then polls for results.

        Supports resume: if output_path exists, already-completed
        instance IDs are skipped.

        Args:
            instances: List of SWE-smith instance dicts.
            file_data_map: Optional mapping from instance_id to file data.
            output_path: Optional path to save results (supports resume).

        Returns:
            List of all prediction dicts (existing + new).
        """
        import asyncio

        file_data_map = file_data_map or {}

        # Load existing results for resume
        completed_ids: set[str] = set()
        if output_path and output_path.exists():
            existing = load_predictions_jsonl(output_path)
            completed_ids = {p["instance_id"] for p in existing}
            print(f"Resuming: {len(completed_ids)} instances already completed")

        # Filter to remaining instances
        remaining = [i for i in instances if i["instance_id"] not in completed_ids]
        print(
            f"Generating patches for {len(remaining)} instances "
            f"({len(completed_ids)} already done)"
        )

        if not remaining:
            if output_path and output_path.exists():
                return load_predictions_jsonl(output_path)
            return []

        # Create async tasks - autobatcher collects and batches them
        tasks = []
        for instance in remaining:
            file_data = file_data_map.get(instance["instance_id"])
            tasks.append(self.generate_patch(instance, file_data))

        # Run all concurrently - autobatcher handles batching
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful predictions
        new_predictions = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Warning: Task failed: {result}")
                continue
            new_predictions.append(result)

        # Append new results to output file
        if output_path and new_predictions:
            append_predictions_jsonl(new_predictions, output_path)

        # Return all predictions (existing + new)
        if output_path and output_path.exists():
            return load_predictions_jsonl(output_path)
        return new_predictions

    async def close(self) -> None:
        """Close the autobatcher client and flush pending batches."""
        await self.batch_client.close()
