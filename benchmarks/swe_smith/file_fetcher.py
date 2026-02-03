"""Pre-fetch file contents from SWE-smith Docker containers.

This module provides utilities for fetching relevant source files from
SWE-smith instances using Docker containers with Rosetta emulation on macOS.

The pre-fetched files are used to provide context for one-shot patch generation,
following the Agentless approach which requires file context for effective fixes.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swerex.deployment.docker import DockerDeployment
    from swerex.runtime.abstract import Runtime


def get_swesmith_container(instance: dict) -> str:
    """Get the SWE-smith container image for an instance.

    Args:
        instance: SWE-smith instance dict containing instance_id.

    Returns:
        Docker image name for the instance.
    """
    # Parse instance_id: owner__repo.commit.mutation_type__mutation_id
    instance_id = instance["instance_id"]

    # Extract owner__repo.commit part (before the mutation type)
    # Format: owner__repo.commit.mutation_type__mutation_id
    parts = instance_id.split(".")
    owner_repo = parts[0]  # owner__repo
    commit = parts[1]  # commit hash

    # Convert __ to _1776_ for Docker naming
    owner_repo_docker = owner_repo.replace("__", "_1776_")

    # Build image name: swebench/swesmith.x86_64.{owner_repo}.{commit}
    # Docker requires lowercase image names
    image = f"swebench/swesmith.x86_64.{owner_repo_docker}.{commit}".lower()
    return image


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths mentioned in a git patch.

    Args:
        patch: Git diff/patch string.

    Returns:
        List of file paths mentioned in the patch.
    """
    # Pattern to match diff --git a/path b/path
    pattern = r"diff --git a/([^\s]+) b/([^\s]+)"
    matches = re.findall(pattern, patch)

    files = set()
    for a_path, b_path in matches:
        files.add(a_path)
        files.add(b_path)

    # Also try to extract from --- and +++ lines
    for prefix in ["--- a/", "+++ b/"]:
        for line in patch.split("\n"):
            if line.startswith(prefix):
                path = line[len(prefix) :].strip()
                if path and path != "/dev/null":
                    files.add(path)

    return sorted(files)


class FileFetcher:
    """Fetches relevant files from SWE-smith containers.

    Uses Docker with --platform linux/amd64 for Rosetta emulation on Apple Silicon.
    """

    def __init__(
        self,
        startup_timeout: int = 120,
        command_timeout: int = 30,
        use_rosetta: bool = True,
    ):
        """Initialize the file fetcher.

        Args:
            startup_timeout: Timeout for container startup in seconds.
            command_timeout: Timeout for individual commands in seconds.
            use_rosetta: Whether to use --platform linux/amd64 for Rosetta.
        """
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self.use_rosetta = use_rosetta

    def _get_docker_args(self) -> list[str]:
        """Get Docker arguments for container creation."""
        args = []
        if self.use_rosetta:
            args.extend(["--platform", "linux/amd64"])
        return args

    async def _create_deployment(self, image: str) -> "DockerDeployment":
        """Create a Docker deployment for an image.

        Args:
            image: Docker image name.

        Returns:
            DockerDeployment instance (not started).
        """
        from swerex.deployment.docker import DockerDeployment

        return DockerDeployment(
            image=image,
            startup_timeout=self.startup_timeout,
            docker_args=self._get_docker_args(),
        )

    async def _get_repo_structure(
        self,
        runtime: "Runtime",
        testbed_path: str = "/testbed",
        max_files: int = 200,
    ) -> list[str]:
        """Get repository file structure from container.

        Args:
            runtime: SWE-ReX runtime for command execution.
            testbed_path: Path to the repository in the container.
            max_files: Maximum number of files to return.

        Returns:
            List of file paths in the repository.
        """
        from swerex.runtime.abstract import BashAction

        # Find Python files (most common in SWE-smith)
        result = await runtime.run_in_session(
            BashAction(
                command=f"find {testbed_path} -type f -name '*.py' | head -{max_files}",
                timeout=self.command_timeout,
                session="default",
                check="silent",
            )
        )

        files = []
        if result.output:
            files = [
                f.strip() for f in result.output.strip().split("\n") if f.strip()
            ]

        return files

    async def _read_file(self, runtime: "Runtime", file_path: str) -> str | None:
        """Read a file from the container.

        Args:
            runtime: SWE-ReX runtime for command execution.
            file_path: Path to the file in the container.

        Returns:
            File contents or None if read failed.
        """
        from swerex.runtime.abstract import BashAction

        result = await runtime.run_in_session(
            BashAction(
                command=f"cat {file_path}",
                timeout=self.command_timeout,
                session="default",
                check="silent",
            )
        )

        if result.exit_code == 0 and result.output:
            return result.output

        return None

    def _filter_relevant_files(
        self,
        all_files: list[str],
        problem_statement: str,
        patch_files: list[str] | None = None,
        max_files: int = 10,
    ) -> list[str]:
        """Filter files to those most relevant to the issue.

        Uses simple heuristics based on problem statement keywords
        and known patch files.

        Args:
            all_files: List of all files in the repository.
            problem_statement: Problem description to extract keywords.
            patch_files: Files mentioned in the gold patch (if available).
            max_files: Maximum number of files to return.

        Returns:
            List of relevant file paths.
        """
        # If we have patch files, prioritize those
        if patch_files:
            # Return patch files that exist in all_files, plus some context files
            relevant = []
            for pf in patch_files:
                # Find matching file in all_files (may have different prefix)
                for af in all_files:
                    if af.endswith(pf) or pf in af:
                        relevant.append(af)
                        break

            # If we found patch files, return them
            if relevant:
                return relevant[:max_files]

        # Extract keywords from problem statement
        # Look for file names, class names, function names
        keywords = set()

        # Find potential file names (word.py, word_name.py, etc.)
        file_pattern = r"\b([a-z_][a-z0-9_]*\.py)\b"
        keywords.update(re.findall(file_pattern, problem_statement.lower()))

        # Find potential module/class/function names
        name_pattern = r"\b([A-Z][a-zA-Z0-9_]*|[a-z_][a-z0-9_]*)\b"
        for match in re.findall(name_pattern, problem_statement):
            if len(match) > 3:  # Skip short words
                keywords.add(match.lower())

        # Score files by keyword matches
        scored_files = []
        for file_path in all_files:
            score = 0
            file_lower = file_path.lower()

            for keyword in keywords:
                if keyword in file_lower:
                    score += 1

            # Boost for common important paths
            if "test" not in file_lower:
                score += 0.5  # Prefer non-test files
            if "__init__" in file_lower:
                score -= 0.5  # Lower priority for __init__ files

            scored_files.append((score, file_path))

        # Sort by score (descending) and return top files
        scored_files.sort(key=lambda x: (-x[0], x[1]))

        return [f for _, f in scored_files[:max_files]]

    async def fetch_instance_files(
        self,
        instance: dict,
        max_files: int = 10,
        max_file_size: int = 50000,
    ) -> dict:
        """Fetch files relevant to an issue from its container.

        Args:
            instance: SWE-smith instance dict with instance_id, problem_statement, patch.
            max_files: Maximum number of files to fetch.
            max_file_size: Maximum size of individual files (characters).

        Returns:
            Dict with instance_id, repo_structure, and file_contents.
        """
        from swerex.runtime.abstract import CreateBashSessionRequest

        instance_id = instance["instance_id"]
        image = get_swesmith_container(instance)

        deployment = await self._create_deployment(image)

        try:
            await deployment.start()
            runtime = deployment.runtime

            # Create bash session
            await runtime.create_session(CreateBashSessionRequest(session="default"))

            # Get repository structure
            repo_structure = await self._get_repo_structure(runtime)

            # Extract files from gold patch if available
            patch = instance.get("patch", "")
            patch_files = extract_files_from_patch(patch) if patch else []

            # Filter to relevant files
            relevant_files = self._filter_relevant_files(
                repo_structure,
                instance.get("problem_statement", ""),
                patch_files,
                max_files,
            )

            # Read file contents
            file_contents = {}
            for file_path in relevant_files:
                content = await self._read_file(runtime, file_path)
                if content:
                    # Truncate large files
                    if len(content) > max_file_size:
                        truncated = len(content) - max_file_size
                        content = (
                            content[: max_file_size // 2]
                            + f"\n\n... [truncated {truncated} characters] ...\n\n"
                            + content[-max_file_size // 2 :]
                        )
                    file_contents[file_path] = content

            return {
                "instance_id": instance_id,
                "repo_structure": repo_structure[:50],  # Limit structure list
                "file_contents": file_contents,
            }

        finally:
            await deployment.stop()

    async def fetch_batch(
        self,
        instances: list[dict],
        output_path: Path,
        workers: int = 5,
        resume: bool = True,
    ) -> list[dict]:
        """Fetch files for a batch of instances in parallel.

        Args:
            instances: List of SWE-smith instance dicts.
            output_path: Path to write JSONL results (incrementally).
            workers: Number of parallel workers.
            resume: Whether to skip already completed instances.

        Returns:
            List of file data dicts.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results for resume
        completed_ids = set()
        existing_results = []

        if resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        existing_results.append(result)
                        completed_ids.add(result["instance_id"])
            print(f"Resuming: {len(completed_ids)} already completed")

        # Filter remaining instances
        remaining = [
            inst for inst in instances if inst["instance_id"] not in completed_ids
        ]
        print(f"Fetching files for {len(remaining)} instances with {workers} workers")

        if not remaining:
            print("All instances already fetched!")
            return existing_results

        # Semaphore for limiting concurrency
        semaphore = asyncio.Semaphore(workers)
        results_lock = asyncio.Lock()

        async def fetch_with_semaphore(instance: dict) -> dict | None:
            async with semaphore:
                try:
                    return await self.fetch_instance_files(instance)
                except Exception as e:
                    print(f"Error fetching {instance['instance_id']}: {e}")
                    return {
                        "instance_id": instance["instance_id"],
                        "repo_structure": [],
                        "file_contents": {},
                        "error": str(e),
                    }

        # Process instances
        results = []
        tasks = [fetch_with_semaphore(inst) for inst in remaining]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result:
                results.append(result)

                # Append to file incrementally
                async with results_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

                print(f"Progress: {len(results)}/{len(remaining)}")

        return existing_results + results


def load_file_data(path: Path) -> dict[str, dict]:
    """Load pre-fetched file data from JSONL.

    Args:
        path: Path to the JSONL file.

    Returns:
        Dict mapping instance_id to file data.
    """
    file_map = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                file_map[data["instance_id"]] = data
    return file_map


def save_file_data(results: list[dict], path: Path) -> None:
    """Save file data to JSONL.

    Args:
        results: List of file data dicts.
        path: Path to write JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
