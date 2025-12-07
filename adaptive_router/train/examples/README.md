# Adaptive Router Training Examples

This directory contains example datasets and TOML configurations for training Adaptive Router profiles.

## Quick Start

```bash
# From the project root directory

# 1. Set required environment variables
export ADAPTIVE_API_KEY="your-adaptive-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # If using Anthropic models

# 2. Run training with example config
uv run python train/train.py --config train/examples/configs/train_with_pricing.toml

# 3. Output will be saved to profile_with_pricing.json
```

## Directory Structure

```
train/examples/
├── configs/               # TOML configuration files
│   ├── train_minimal.toml           # Basic config with API pricing
│   ├── train_with_pricing.toml      # Full pricing in TOML (no API calls)
│   ├── train_s3.toml                # S3/MinIO output
│   └── train_custom_params.toml     # Advanced clustering parameters
├── datasets/              # Sample datasets
│   ├── minimal_qa.csv               # CSV format (10 samples)
│   ├── minimal_qa.json              # JSON format (10 samples)
│   └── minimal_qa.parquet           # Parquet format (10 samples)
└── README.md              # This file
```

## Sample Datasets

All sample datasets contain 10 general knowledge questions with expected answers:

### CSV Format (`minimal_qa.csv`)
```csv
input,expected_output
"What is the capital of France?","Paris"
...
```

### JSON Format (`minimal_qa.json`)
```json
[
  {
    "input": "What is the capital of France?",
    "expected_output": "Paris"
  },
  ...
]
```

### Parquet Format (`minimal_qa.parquet`)
Same data as CSV/JSON, stored in Parquet format for efficient processing.

## Configuration Files

### 1. Minimal Configuration (`train_minimal.toml`)

Simplest configuration that fetches model pricing from the Adaptive API:

```toml
[api]
adaptive_api_key = "${ADAPTIVE_API_KEY}"

[dataset]
path = "train/examples/datasets/minimal_qa.csv"
type = "csv"

[[models]]
provider = "openai"
model_name = "gpt-4"

[[models]]
provider = "openai"
model_name = "gpt-3.5-turbo"

[providers.openai]
api_key = "${OPENAI_API_KEY}"

[output]
path = "profile_minimal.json"
storage_type = "local"
```

**Use when**: You want the simplest setup and have access to the Adaptive Models API.

### 2. Full Pricing Configuration (`train_with_pricing.toml`)

Defines all model pricing in the TOML file (no API calls for pricing):

```toml
[[models]]
provider = "openai"
model_name = "gpt-4"
cost_per_1m_input_tokens = 30.0
cost_per_1m_output_tokens = 60.0

[[models]]
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
cost_per_1m_input_tokens = 3.0
cost_per_1m_output_tokens = 15.0
```

**Use when**: You want full control over pricing or work offline/without API access.

### 3. S3/MinIO Output (`train_s3.toml`)

Saves trained profile to S3-compatible storage:

```toml
[output]
path = "profile_s3.json"
storage_type = "s3"

[output.s3]
endpoint_url = "https://s3.amazonaws.com"
access_key_id = "${AWS_ACCESS_KEY_ID}"
secret_access_key = "${AWS_SECRET_ACCESS_KEY}"
bucket_name = "adaptive-router-profiles"
region = "us-east-1"
profile_key = "profiles/my-profile.json"
```

**Use when**: You need to deploy profiles to cloud storage for production use.

### 4. Custom Parameters (`train_custom_params.toml`)

Advanced configuration with custom clustering and training parameters:

```toml
[training]
n_clusters = 30                    # More clusters for finer routing
max_parallel = 20                  # Higher parallelism
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
tfidf_max_features = 10000         # More TF-IDF features
tfidf_ngram_range = [1, 3]         # Include trigrams
random_seed = 12345
```

**Use when**: You want to optimize clustering quality or experiment with parameters.

## TOML Configuration Reference

### `[api]` - Adaptive API Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `adaptive_api_key` | string | Yes | API key for Adaptive Models API |
| `base_url` | string | No | Custom API endpoint (default: `https://api.llmadaptive.uk/v1`) |

### `[dataset]` - Dataset Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Path to dataset file (relative to project root) |
| `type` | string | Yes | Dataset format: `csv`, `json`, or `parquet` |
| `input_column` | string | No | Input column name (default: `input`) |
| `expected_column` | string | No | Expected output column name (default: `expected_output`) |

### `[[models]]` - Model Definitions (Array)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | Yes | Provider name (e.g., `openai`, `anthropic`) |
| `model_name` | string | Yes | Model name (e.g., `gpt-4`, `claude-3-5-sonnet-20241022`) |
| `cost_per_1m_input_tokens` | float | No* | Cost per 1M input tokens |
| `cost_per_1m_output_tokens` | float | No* | Cost per 1M output tokens |

\* If pricing is omitted, it will be fetched from the Adaptive API.
\* If pricing is provided, both fields must be set together.

### `[providers.{name}]` - Provider API Configs

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_key` | string | Yes | Provider API key (supports `${ENV_VAR}` syntax) |
| `base_url` | string | No | Custom API endpoint for provider |
| `organization` | string | No | Organization ID (if applicable) |
| `timeout` | float | No | Request timeout in seconds (default: 60.0) |
| `max_retries` | int | No | Maximum retry attempts (default: 3) |

### `[training]` - Training Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `n_clusters` | int | No | Number of K-means clusters (default: 20) |
| `max_parallel` | int | No | Max parallel inference requests (default: 10) |
| `embedding_model` | string | No | Sentence transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `tfidf_max_features` | int | No | Max TF-IDF features (default: 5000) |
| `tfidf_ngram_range` | array | No | N-gram range `[min, max]` (default: `[1, 2]`) |
| `random_seed` | int | No | Random seed for reproducibility (default: 42) |

### `[output]` - Output Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Output file path |
| `storage_type` | string | No | Storage backend: `local`, `s3`, or `minio` (default: `local`) |

### `[output.s3]` - S3/MinIO Configuration

Required only if `storage_type` is `s3` or `minio`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `endpoint_url` | string | No | S3 endpoint URL (for MinIO or custom S3) |
| `access_key_id` | string | Yes | AWS access key or MinIO root user |
| `secret_access_key` | string | Yes | AWS secret key or MinIO root password |
| `bucket_name` | string | Yes | S3 bucket name |
| `region` | string | No | AWS region (default: `us-east-1`) |
| `profile_key` | string | No | Object key in bucket (default: `profile.json`) |

## Environment Variable Syntax

TOML configs support environment variable substitution using `${VAR_NAME}` syntax:

```toml
[api]
adaptive_api_key = "${ADAPTIVE_API_KEY}"  # Reads from environment

[providers.openai]
api_key = "${OPENAI_API_KEY}"             # Reads from environment
```

Alternatively, you can provide values directly:

```toml
[api]
adaptive_api_key = "sk-..."              # Direct value (not recommended for security)
```

## Usage Examples

### Example 1: Train with Minimal Config

```bash
# Set environment variables
export ADAPTIVE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"

# Run training
uv run python train/train.py --config train/examples/configs/train_minimal.toml

# Output: profile_minimal.json
```

### Example 2: Train with Full TOML Pricing (No API Calls)

```bash
# Set environment variables
export ADAPTIVE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run training
uv run python train/train.py --config train/examples/configs/train_with_pricing.toml

# Output: profile_with_pricing.json
```

### Example 3: Train with Custom Parameters

```bash
# Set environment variables
export ADAPTIVE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# Run training with advanced clustering
uv run python train/train.py --config train/examples/configs/train_custom_params.toml

# Output: profile_custom.json (with 30 clusters, trigrams, etc.)
```

### Example 4: Train and Save to S3

```bash
# Set all required environment variables
export ADAPTIVE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"

# Run training with S3 output
uv run python train/train.py --config train/examples/configs/train_s3.toml

# Output: Saved to S3 bucket adaptive-router-profiles/profiles/my-profile.json
```

## Creating Your Own Dataset

Your dataset must contain at least two columns:

1. **Input column**: Contains prompts/questions
2. **Expected output column**: Contains correct answers/responses

### CSV Format

```csv
input,expected_output
"Your prompt here","Expected response"
"Another prompt","Another response"
```

### JSON Format

```json
[
  {
    "input": "Your prompt here",
    "expected_output": "Expected response"
  },
  {
    "input": "Another prompt",
    "expected_output": "Another response"
  }
]
```

### Parquet Format

Use Polars or Pandas to create Parquet files:

```python
import polars as pl

data = {
    'input': ['Your prompt here', 'Another prompt'],
    'expected_output': ['Expected response', 'Another response']
}

df = pl.DataFrame(data)
df.write_parquet('my_dataset.parquet')
```

## Hybrid Model Loading

The training script supports **hybrid model loading**:

- **API Pricing**: Omit pricing fields to fetch from Adaptive API
- **TOML Pricing**: Provide `cost_per_1m_input_tokens` and `cost_per_1m_output_tokens`
- **Mixed Mode**: Some models from API, some from TOML in the same config

Example:

```toml
[[models]]
provider = "openai"
model_name = "gpt-4"
# Will fetch pricing from API

[[models]]
provider = "openai"
model_name = "gpt-3.5-turbo"
cost_per_1m_input_tokens = 0.5
cost_per_1m_output_tokens = 1.5
# Uses TOML pricing
```

## Training Output

After training completes, you'll see:

```
================================================================================
TRAINING RESULTS
================================================================================
Total samples: 10
Number of clusters: 20
Silhouette score: 0.3421
Training time: 12.45 seconds
Inference time: 8.23 seconds

Model Error Rates:
  openai/gpt-4: 5.23%
  openai/gpt-3.5-turbo: 12.45%

Profile saved to: profile_with_pricing.json
================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================
```

## Troubleshooting

### Error: "Dataset file not found"

- Ensure paths are relative to the **project root** (not the train/ directory)
- Use absolute paths if working from a different directory

### Error: "Environment variable not set"

- Check that all required API keys are exported
- Verify syntax: `${VAR_NAME}` in TOML, not `$VAR_NAME`

### Error: "Missing provider configurations"

- Ensure every model's provider has a `[providers.{name}]` section
- Example: If using `openai` models, you need `[providers.openai]`

### Error: "Both pricing fields must be provided together"

- If specifying pricing in TOML, set both `cost_per_1m_input_tokens` and `cost_per_1m_output_tokens`
- Or omit both to fetch from API

## Advanced Topics

### Optimizing Cluster Count

- **Low clusters (10-15)**: Faster training, less granular routing
- **Medium clusters (20-30)**: Balanced performance and accuracy
- **High clusters (40-50+)**: More accurate routing, slower training

### Tuning TF-IDF Parameters

- `tfidf_max_features`: Higher values capture more vocabulary (slower)
- `tfidf_ngram_range`: Trigrams `[1, 3]` capture more context (larger features)

### Parallel Inference

- `max_parallel`: Control concurrent API requests during training
- Higher values = faster training, but may hit rate limits
- Lower values = slower training, more reliable

## Dataset Size Recommendations

The quality of your router profile depends significantly on dataset size:

| Clusters | Min Samples | Recommended | Notes |
|----------|-------------|-------------|-------|
| 5-10 | 50-100 | 200-500 | Good for simple categorization |
| 15-25 | 150-250 | 500-1000 | Balanced for general use |
| 30-40 | 300-400 | 1000-2000 | Better routing accuracy |
| 50+ | 500+ | 2000+ | Production-grade routing |

**Rule of thumb**: Aim for **50-100 samples per cluster** for reliable error rate estimation.

**Diverse prompts**: Include variety in prompt types, lengths, and complexity levels to ensure good cluster separation.

## Profile Optimization Tips

### Improving Cluster Quality

1. **Silhouette Score**: Target 0.3-0.5 for good cluster separation
   - Score < 0.2: Consider fewer clusters or more data
   - Score > 0.5: Excellent separation, may need more clusters for finer routing

2. **Feature Engineering**:
   ```toml
   [training]
   tfidf_max_features = 10000    # More vocabulary coverage
   tfidf_ngram_range = [1, 3]    # Include trigrams for context
   ```

3. **Embedding Model**: The default `all-MiniLM-L6-v2` works well for most cases
   - For multilingual: Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   - For long prompts: Use `sentence-transformers/all-mpnet-base-v2`

### Reducing Error Rates

1. **More training samples**: Error rates become more accurate with more data
2. **Model-specific prompts**: Include prompts that highlight each model's strengths
3. **Cost-quality balance**: Ensure models span different price/capability ranges

### Profile Size Optimization

For production with the C++ core:

```toml
[training]
n_clusters = 20                  # Good balance: smaller profile, fast routing
tfidf_max_features = 5000        # Standard vocabulary
```

Larger profiles (40+ clusters) provide finer routing but:
- Increase memory usage
- Require more training samples
- May slow down routing slightly

## Expected Training Output

### Sample Training Log

```
[2024-01-15 10:30:00] Starting training with config: train_with_pricing.toml
[2024-01-15 10:30:01] Loading dataset from train/examples/datasets/qa_dataset.csv
[2024-01-15 10:30:01] Dataset loaded: 500 samples
[2024-01-15 10:30:02] Computing embeddings with all-MiniLM-L6-v2...
[2024-01-15 10:30:15] Embedding shape: (500, 384)
[2024-01-15 10:30:15] Computing TF-IDF features...
[2024-01-15 10:30:16] TF-IDF shape: (500, 5000)
[2024-01-15 10:30:16] Running K-means clustering with 20 clusters...
[2024-01-15 10:30:18] Clustering completed. Silhouette score: 0.42
[2024-01-15 10:30:18] Running inference to compute error rates...
[2024-01-15 10:32:45] Inference completed for 3 models

================================================================================
TRAINING RESULTS
================================================================================
Total samples: 500
Number of clusters: 20
Silhouette score: 0.4234
Training time: 165.32 seconds
Inference time: 147.21 seconds

Model Error Rates (across all clusters):
  openai/gpt-4: 3.2% avg (range: 0.8% - 8.4%)
  openai/gpt-3.5-turbo: 11.4% avg (range: 5.2% - 22.1%)
  anthropic/claude-3-sonnet: 4.8% avg (range: 1.2% - 12.3%)

Cost Summary:
  Cheapest: openai/gpt-3.5-turbo ($0.50/1M input)
  Most expensive: openai/gpt-4 ($30.00/1M input)
  Best quality: openai/gpt-4 (3.2% error rate)

Profile saved to: profile_with_pricing.json (1.2 MB)
================================================================================
TRAINING COMPLETED SUCCESSFULLY
================================================================================
```

### Profile JSON Structure

```json
{
  "metadata": {
    "n_clusters": 20,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "silhouette_score": 0.4234,
    "created_at": "2024-01-15T10:32:45Z",
    "sample_count": 500,
    "clustering": {
      "max_iter": 300,
      "random_state": 42,
      "algorithm": "lloyd"
    },
    "routing": {
      "lambda_min": 0.0,
      "lambda_max": 2.0,
      "default_cost_preference": 0.5
    }
  },
  "cluster_centers": [...],
  "models": [
    {
      "model_id": "openai/gpt-4",
      "provider": "openai",
      "model_name": "gpt-4",
      "error_rates": [0.032, 0.045, ...],
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0
    }
  ],
  "scalers": {
    "embedding_mean": [...],
    "embedding_std": [...],
    "tfidf_mean": [...],
    "tfidf_std": [...]
  }
}
```

## Next Steps

1. **Create your own dataset** with domain-specific prompts
2. **Experiment with configurations** using the examples as templates
3. **Deploy profiles** to S3/MinIO for production use
4. **Integrate with Adaptive Router** to start intelligent model routing

For more information, see the main [Adaptive Router documentation](../README.md).
