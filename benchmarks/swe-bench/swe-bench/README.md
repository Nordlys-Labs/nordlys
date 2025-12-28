# SWE-bench Routing Benchmark

Evaluate Nordlys model routing on SWE-bench Verified (500 instances).

## Quick Start

1. **Configure credentials:**
   ```bash
   # AWS (for S3 access to pre-computed patches)
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret

   # Nordlys (for model routing)
   export NORDLYS_API_KEY=your_key
   ```

2. **Run benchmark:**
   ```bash
   cd adaptive_router/benchmarks
   uv run python swe-bench/swe-bench/scripts/run_benchmark.py
   ```

3. **Submit results:**
   ```bash
   uv run sb-cli submit swe-bench_verified test \
     --predictions_path swe-bench/swe-bench/output/all_preds.jsonl \
     --run_id your-run-id
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NORDLYS_API_KEY` | Yes | - | Nordlys API key |
| `AWS_ACCESS_KEY_ID` | Yes | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | - | AWS secret key |
| `AWS_DEFAULT_REGION` | No | `eu-west-2` | AWS region |
| `NORDLYS_API_BASE` | No | `https://api.nordlyslabs.com/v1` | Nordlys API URL |

## Output

| File | Description |
|------|-------------|
| `output/all_preds.jsonl` | SWE-bench predictions (for submission) |
| `output/routing_summary.json` | Routing distribution and cost metrics |

## How It Works

1. Loads 500 SWE-bench Verified instances from HuggingFace
2. Routes each instance through Nordlys to select optimal model
3. Fetches pre-computed patches from S3 for each selected model
4. Aggregates results and generates submission file
