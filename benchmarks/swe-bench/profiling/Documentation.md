# Adaptive Router Profiling

Generate model profiles by mapping SWE-bench results to semantic clusters.

## Setup

```bash
cd profiling
uv sync

# Add GitHub token to clustering/.env
echo "GITHUB_TOKEN=ghp_your_token" > clustering/.env
```

## Commands

```bash
cd clustering

# List available models
uv run python profile_model.py --list-models
uv run python profile_model.py --list-models --eval-type bash-only

# Profile a model (fetches from GitHub, saves to profiles/)
uv run python profile_model.py --model-folder "20240620_sweagent_claude3.5sonnet"
uv run python profile_model.py --model-folder "20251124_mini-v1.16.0_claude-opus-4-5" --eval-type bash-only

# Add a single model to profile.json (without re-reading all)
uv run python profile_model.py --add "20250929_mini-v1.13.3_sonnet-4-5-20250929"

# Combine all profiles into profile.json (requires cluster centroids)
uv run python profile_model.py --combine
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--model-folder` | Fetch & profile model from swe-bench/experiments |
| `--eval-type` | `verified` (default) or `bash-only` |
| `--list-models` | List available models on GitHub |
| `--add MODEL` | Add single model from profiles/ to profile.json |
| `--combine` | Combine all profiles into profile.json |

## Output Structure

All output files are stored inside `clustering/`:

**Individual profile** (`clustering/profiles/{model}/profile.json`):
```json
{"model_name": "...", "error_rates": [0.33, 0.47, ...], "overall_error_rate": 0.29}
```

**Combined profile** (`clustering/profile.json`):
```json
{"cluster_centers": {...}, "models": [...], "metadata": {...}}
```

> **Note**: The `--combine` command requires cluster centroids to already exist in `clustering/profile.json`. Run the clustering notebook first to generate centroids before combining model profiles.

## Troubleshooting

- **403 Rate Limit**: Add GitHub token to `clustering/.env`
- **404 Not Found**: Check `--list-models` for valid names
- **401 Bad Credentials**: Regenerate token at github.com/settings/tokens
