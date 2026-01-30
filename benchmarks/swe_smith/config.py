"""Configuration constants for SWE-smith cluster evaluation."""

from pathlib import Path

# Cluster data paths
CLUSTERS_DIR = (
    Path(__file__).parent.parent
    / "supernova"
    / "clustering_swe_smith"
    / "cluster_full_dataset"
)
CLUSTER_ASSIGNMENTS = CLUSTERS_DIR / "cluster_assignments.json"

# Sampling configuration
MIN_SAMPLES = 150  # Minimum samples per cluster
SAMPLE_FRACTION = 0.20  # Sample 20% of cluster size

# Execution defaults
DEFAULT_WORKERS = 10
DEFAULT_TIMEOUT = 240  # seconds per instance

# Output paths
RESULTS_DIR = Path(__file__).parent / "results"
PROFILES_DIR = RESULTS_DIR / "profiles"

# SWE-smith dataset
SWE_SMITH_DATASET = "SWE-bench/SWE-smith"
SWE_SMITH_SPLIT = "train"

# Number of clusters (excluding noise cluster -1)
N_CLUSTERS = 66

# Noise cluster label (to skip during sampling)
NOISE_CLUSTER = -1

# =============================================================================
# Doubleword Batch API Configuration
# =============================================================================

DOUBLEWORD_API_BASE = "https://api.doubleword.ai/v1"
DOUBLEWORD_COMPLETION_WINDOW = "24h"  # Options: "24h" or "1h" for faster results

# Maximum tokens for model responses
MAX_OUTPUT_TOKENS = 8192

# =============================================================================
# Autobatcher SDK Configuration
# =============================================================================

AUTOBATCH_SIZE = 100            # Auto-submit after N requests
AUTOBATCH_WINDOW_SECONDS = 1.0  # Auto-submit after N seconds of inactivity
AUTOBATCH_POLL_INTERVAL = 5.0   # Polling frequency in seconds

# Temperature for deterministic one-shot generation
GENERATION_TEMPERATURE = 0.0

# =============================================================================
# One-Shot Prompt Templates (Based on Agentless Research)
# =============================================================================
# Source: "Agentless: Demystifying LLM-based Software Engineering Agents"
# (FSE 2025, https://arxiv.org/abs/2407.01489)
#
# The Agentless approach achieves 32% on SWE-bench Lite using SEARCH/REPLACE
# diff format. We adapt their repair prompt for one-shot batch generation.
# =============================================================================

ONE_SHOT_SYSTEM_PROMPT = """\
You are an expert software engineer tasked with fixing bugs in code repositories.

Given an issue description and relevant code context, generate a patch to fix the issue.

## Output Format
Generate your fix using SEARCH/REPLACE blocks. Each block must:
1. Contain the EXACT lines to find (including whitespace and indentation)
2. Contain the replacement lines

Format each edit as:
### path/to/file.py
<<<<<<< SEARCH
[exact lines to find]
=======
[replacement lines]
>>>>>>> REPLACE

## Guidelines
- Only modify what's necessary to fix the issue
- Preserve existing code style and indentation exactly
- You may have multiple SEARCH/REPLACE blocks for different files or locations
- Each block must be prefixed with the file path as shown above
- The SEARCH section must match the original code EXACTLY (including whitespace)
- If a file needs multiple edits, use separate SEARCH/REPLACE blocks for each location"""

ONE_SHOT_USER_PROMPT = """## Issue Description
{problem_statement}

## Repository Structure
```
{repo_structure}
```

## Relevant Files
{file_contents}

Please analyze the issue and generate SEARCH/REPLACE blocks to fix it. Remember:
- Match the exact code including whitespace and indentation
- Only make changes necessary to fix the described issue
- Preserve the existing code style"""

# Template for formatting individual file contents in the prompt
FILE_CONTENT_TEMPLATE = """### {file_path}
```{language}
{content}
```"""
