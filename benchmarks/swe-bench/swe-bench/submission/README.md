# Nordlys-singularity (Adaptive Router)

## Overview

Nordlys-singularity is an intelligent model router that dynamically selects the optimal LLM for each SWE-bench instance based on semantic analysis of the problem statement.

## Approach

- **Routing Strategy**: Cluster-based model selection using semantic embeddings
- **Underlying Models**: Multiple LLMs accessed via unified Anthropic-compatible API
- **Agent Framework**: mini-swe-agent for SWE-bench task execution

## Setup

```bash
# Install dependencies
uv sync

# Install mini-swe-agent
uv tool install mini-swe-agent

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

## Reproduction

```bash
# Run benchmark
uv run python src/run.py \
    --model nordlys/Nordlys-singularity \
    --subset verified \
    --split test \
    --workers 4 \
    --output results/nordlys-run
```

## Results

| Dataset | Resolved | Total | Rate |
|---------|----------|-------|------|
| Verified | TBD | 500 | TBD% |

## Files

- `all_preds.jsonl` - Predictions for all instances
- `trajs/` - Reasoning trajectories
- `logs/` - Execution logs
- `metadata.yaml` - Submission metadata
