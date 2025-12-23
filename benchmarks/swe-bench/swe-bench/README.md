# SWE-bench Benchmarking with Nordlys Router

Benchmark the Nordlys Router on SWE-bench using mini-swe-agent with cloud-based evaluation via `sb-cli`.

## Quick Start

### 1. Install Dependencies

```bash
cd benchmarks

# Install Python dependencies (shared pyproject.toml)
uv sync

# Install mini-swe-agent as a tool
uv tool install mini-swe-agent
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# - NORDLYS_API_KEY: Your Nordlys/Adaptive API key
# - NORDLYS_API_BASE: API endpoint (default: https://api.llmadaptive.uk)
# - SWEBENCH_API_KEY: Your SWE-bench API key (for submission)
```

### 3. Get SWE-bench API Key (One-time Setup)

```bash
# Request API key
sb-cli gen-api-key your.email@example.com

# Verify with code from email
sb-cli verify-api-key YOUR_VERIFICATION_CODE

# Add to .env
echo "SWEBENCH_API_KEY=your-key" >> .env
```

### 4. Run Benchmark

```bash
# Source environment variables
source .env

# Quick test (5 instances)
uv run python swe-bench/swe-bench/src/run.py --slice :5 --output results/test-run

# Full benchmark (all 500 verified instances)
uv run python swe-bench/swe-bench/src/run.py
```

## Usage

### Running with Custom Options

```bash
# Custom number of workers
uv run python swe-bench/swe-bench/src/run.py --workers 8

# Specific instance range
uv run python swe-bench/swe-bench/src/run.py --slice 10:20

# Custom output directory
uv run python swe-bench/swe-bench/src/run.py --output results/my-run

# Use a different subset
uv run python swe-bench/swe-bench/src/run.py --subset lite
```

### Direct mini-swe-agent Usage

You can also use mini-swe-agent directly after registering the provider:

```python
from swe_bench.swe_bench.src.nordlys_provider import register_nordlys_provider
register_nordlys_provider()

# Then run mini-extra swebench commands
```

## Submit for Evaluation

### Check Your Quota

```bash
sb-cli get-quotas
```

### Submit Predictions

```bash
# Submit your run
sb-cli submit swe-bench_verified test \
    --predictions_path results/nordlys-run/preds.json \
    --run_id nordlys-singularity-v1

# Check status
sb-cli list-runs swe-bench_verified test

# Get results (typically within 20 minutes)
sb-cli get-report swe-bench_verified test nordlys-singularity-v1
```

## Leaderboard Submission

After evaluation, submit to the SWE-bench leaderboard:

### 1. Fork the Experiments Repo

```bash
# Fork https://github.com/SWE-bench/experiments
git clone https://github.com/YOUR_USERNAME/experiments.git
cd experiments
```

### 2. Create Submission Directory

```bash
# Create directory (use today's date)
mkdir -p evaluation/verified/YYYYMMDD_Nordlys-singularity/
```

### 3. Add Required Files

Copy from your results:
- `all_preds.jsonl` - Predictions
- `trajs/` - Reasoning trajectories
- `logs/` - Execution logs
- `metadata.yaml` - From `submission/metadata.yaml`
- `README.md` - From `submission/README.md`

### 4. Create Pull Request

Submit PR to the experiments repo for review.

## Output Files

```
swe-bench/
├── results/
│   └── nordlys-run/
│       ├── preds.json           # Predictions for sb-cli
│       ├── all_preds.jsonl      # All predictions (for leaderboard)
│       ├── trajs/               # Reasoning trajectories
│       │   └── *.traj.json
│       └── logs/                # Execution logs
└── submission/
    ├── metadata.yaml            # Leaderboard metadata template
    └── README.md                # Leaderboard README template
```

## How It Works

### 1. Custom LiteLLM Provider

The `src/nordlys_provider.py` registers a custom provider that routes `nordlys/*` model requests to the Nordlys Anthropic-compatible API.

```python
# Internally, mini-swe-agent uses LiteLLM
# Our custom provider handles: nordlys/Nordlys-singularity
litellm.custom_provider_map = [
    {"provider": "nordlys", "custom_handler": NordlysProvider()}
]
```

### 2. mini-swe-agent Execution

For each SWE-bench instance:
1. Load problem statement and repository
2. Agent explores codebase using bash commands
3. Agent generates patch to fix the issue
4. Patch and trajectory saved to output

### 3. Cloud Evaluation

sb-cli submits predictions to SWE-bench's cloud infrastructure for secure evaluation against the test harness.

## Configuration Reference

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NORDLYS_API_KEY` | Nordlys/Adaptive API key | Yes |
| `NORDLYS_API_BASE` | API endpoint | No (default: https://api.llmadaptive.uk) |
| `SWEBENCH_API_KEY` | SWE-bench submission key | For submission |

### mini-swe-agent Options

| Option | Description |
|--------|-------------|
| `--model` | Model to use (default: nordlys/Nordlys-singularity) |
| `--subset` | Dataset subset: verified, lite, full |
| `--split` | Dataset split: test, dev |
| `--workers` | Parallel workers (default: 4) |
| `--slice` | Instance range (e.g., :5, 10:20) |
| `--output` | Output directory |

## Troubleshooting

### "NORDLYS_API_KEY not found"

```bash
# Check environment
echo $NORDLYS_API_KEY

# Source .env file
source .env
```

### "mini-extra command not found"

```bash
# Install mini-swe-agent
uv tool install mini-swe-agent

# Verify installation
which mini-extra
```

### "sb-cli not found"

```bash
# Install via uv
uv sync

# Or install directly
pip install sb-cli
```

## Resources

- **mini-swe-agent**: https://mini-swe-agent.com/
- **SWE-bench**: https://www.swebench.com/
- **sb-cli Docs**: https://www.swebench.com/sb-cli/
- **Experiments Repo**: https://github.com/SWE-bench/experiments
