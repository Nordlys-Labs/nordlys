# Adaptive Router Profiling

Generate model profiles by mapping SWE-bench results to semantic clusters.

## Setup

```bash
cd adaptive_router/profiling
uv sync

# Add GitHub token to clustering/.env
echo "GITHUB_TOKEN=ghp_your_token" > clustering/.env
```

## Commands

```bash
cd clustering

# List available models
python profile_model.py --list-models
python profile_model.py --list-models --eval-type bash-only

# Profile a model (fetches from GitHub, saves to profiles/)
python profile_model.py --model-folder "20240620_sweagent_claude3.5sonnet"
python profile_model.py --model-folder "20251124_mini-v1.16.0_claude-opus-4-5-20251101" --eval-type bash-only

# Add a single model to profile.json (without re-reading all)
python profile_model.py --add "20250929_mini-v1.13.3_sonnet-4-5-20250929"

# Combine all profiles into profile.json (but you need to make sure that you already have clusters centroids)
python profile_model.py --combine
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

**Individual profile** (`profiles/{model}/profile.json`):
```json
{"model_name": "...", "error_rates": [0.33, 0.47, ...], "overall_error_rate": 0.29}
```

**Combined profile** (`profile.json`):
```json
{"cluster_centers": {...}, "models": [...], "metadata": {...}}
```

## Troubleshooting

- **403 Rate Limit**: Add GitHub token to `clustering/.env`
- **404 Not Found**: Check `--list-models` for valid names
- **401 Bad Credentials**: Regenerate token at github.com/settings/tokens
