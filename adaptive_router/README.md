# Adaptive Router - Core Library Package

Python library for intelligent LLM model selection using cluster-based routing with per-cluster error rates and cost optimization.

## Overview

The Adaptive Router library provides the core ML routing logic for selecting optimal LLM models based on prompt analysis. It uses cluster-based intelligent routing (UniRouter algorithm) to balance cost and quality, reducing LLM costs by 30-70% while maintaining high-quality responses.

This library can be used standalone in Python applications or as a dependency for the FastAPI application (`app/` package).

## Features

- **Cluster-Based Routing**: UniRouter algorithm with K-means clustering and per-cluster error rates
- **Cost Optimization**: Configurable cost-quality tradeoff via `cost_bias` parameter
- **Multiple Profile Loaders**: Local file and MinIO S3 profile loading
- **GPU Acceleration**: Optional GPU support for sentence transformer embeddings
- **Model Filtering**: Optional model filtering to restrict selection to specific providers/models
- **Type-Safe API**: Full type hints with Pydantic models

## Installation

### From Source

```bash
# Install from project root
cd /path/to/adaptive_router
uv install

# Or install the library package specifically
cd adaptive_router
uv install
```

### As Dependency

The library is used as a local path dependency by the `app/` package:

```toml
[tool.uv.sources]
adaptive-router = { path = "../adaptive_router", editable = true }
```

## Quick Start

### Basic Usage

```python
from adaptive_router import ModelRouter, ModelSelectionRequest

# Load router from local profile file
router = ModelRouter.from_local_file("router_profile.json")

# Select optimal model
request = ModelSelectionRequest(
    prompt="Write a Python function to sort a list",
    cost_bias=0.5  # 0.0 = cheapest, 1.0 = highest quality
)
response = router.select_model(request)

print(f"Selected: {response.model_id}")  # e.g., "openai/gpt-3.5-turbo"
print(f"Alternatives: {[alt.model_id for alt in response.alternatives]}")
```

### Quick Routing (Just Get Model ID)

```python
from adaptive_router import ModelRouter

router = ModelRouter.from_local_file("router_profile.json")

# Just get the model ID string
model_id = router.route("Write a sorting algorithm", cost_bias=0.3)
print(model_id)  # "openai/gpt-3.5-turbo"
```

### Advanced Usage with Model Filtering

```python
from adaptive_router import ModelRouter, ModelSelectionRequest, Model

router = ModelRouter.from_local_file("router_profile.json")

# Optionally filter to specific models
allowed_models = [
    Model(
        provider="openai",
        model_name="gpt-4",
        cost_per_1m_input_tokens=30.0,
        cost_per_1m_output_tokens=60.0,
    )
]

response = router.select_model(
    ModelSelectionRequest(
        prompt="Design a distributed system",
        cost_bias=0.9,  # Prefer quality
        models=allowed_models  # Optional: restrict to these models
    )
)
```

### MinIO Profile Loading

```python
from adaptive_router import ModelRouter, MinIOSettings

# Configure MinIO connection
settings = MinIOSettings(
    endpoint_url="https://minio.example.com",
    root_user="admin",
    root_password="password",
    bucket_name="profiles",
    object_key="router_profile.json"
)

# Load router from MinIO S3
router = ModelRouter.from_minio(settings)
```

## Core Components

### ModelRouter

**File**: `adaptive_router/core/router.py`

Main entry point for intelligent model selection.

**Key Methods:**
- `from_local_file(path)`: Initialize from local JSON profile
- `from_minio(settings)`: Initialize from MinIO S3 profile
- `from_profile(profile)`: Initialize from RouterProfile object
- `select_model(request)`: Select optimal model based on prompt
- `route(prompt, cost_bias)`: Quick routing (returns model ID string)

**Example:**
```python
from adaptive_router import ModelRouter, ModelSelectionRequest

router = ModelRouter.from_local_file("profile.json")
response = router.select_model(
    ModelSelectionRequest(prompt="Explain quantum computing", cost_bias=0.5)
)
```

### ClusterEngine

**File**: `adaptive_router/core/cluster_engine.py`

Handles K-means clustering operations for prompt assignment.

**Features:**
- Loads pre-trained cluster centers from profiles
- Assigns prompts to clusters using K-means prediction
- Manages cluster metadata and silhouette scores
- Fast cluster assignment (<5ms per request)

### Profile Loaders

**Files**: `adaptive_router/loaders/`

Profile loading system with multiple implementations:

**LocalFileProfileLoader** (`loaders/local.py`):
- Loads profiles from local filesystem
- Supports JSON format
- Used for testing and offline development

**MinIOProfileLoader** (`loaders/minio.py`):
- Loads profiles from MinIO S3-compatible storage
- Supports environment-based configuration
- Connection pooling and retry logic

**Example:**
```python
from adaptive_router.loaders import LocalFileProfileLoader, MinIOProfileLoader

# Local loading
loader = LocalFileProfileLoader("profile.json")
profile = loader.load_profile()

# MinIO loading
minio_loader = MinIOProfileLoader(settings)
profile = minio_loader.load_profile()
```

### Data Models

**Files**: `adaptive_router/models/`

Type-safe Pydantic models for all data structures:

- **`api.py`**: `ModelSelectionRequest`, `ModelSelectionResponse`, `Model`, `Alternative`
- **`storage.py`**: `RouterProfile`, `ProfileMetadata`, `ClusterCentersData`
- **`routing.py`**: `RoutingDecision`, `ModelInfo`, `ModelPricing`
- **`config.py`**: `ModelConfig`, training configuration models

## API Reference

### ModelRouter

```python
class ModelRouter:
    @classmethod
    def from_local_file(cls, path: str | Path) -> ModelRouter:
        """Initialize from local profile file."""
    
    @classmethod
    def from_minio(cls, settings: MinIOSettings) -> ModelRouter:
        """Initialize from MinIO S3 profile."""
    
    @classmethod
    def from_profile(cls, profile: RouterProfile) -> ModelRouter:
        """Initialize from RouterProfile object."""
    
    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis."""
    
    def route(self, prompt: str, cost_bias: float = 0.5) -> str:
        """Quick routing - returns model ID string."""
```

### ModelSelectionRequest

```python
class ModelSelectionRequest(BaseModel):
    prompt: str                   # Text to analyze
    cost_bias: float = 0.5       # 0.0=cheapest, 1.0=highest quality
    models: list[Model] = None   # Optional: constrain to specific models
    user_id: str = None          # Optional: user identifier
```

### ModelSelectionResponse

```python
class ModelSelectionResponse(BaseModel):
    model_id: str                     # Selected model (e.g., "openai/gpt-4")
    alternatives: list[Alternative]   # Alternative recommendations
    reasoning: str = None             # Optional: selection reasoning
```

### Model

```python
class Model(BaseModel):
    provider: str                     # Provider name (e.g., "openai")
    model_name: str                   # Model name (e.g., "gpt-4")
    cost_per_1m_input_tokens: float   # Cost per 1M input tokens
    cost_per_1m_output_tokens: float # Cost per 1M output tokens
    max_context_tokens: int = None    # Optional: context window size
    supports_function_calling: bool = False  # Optional: function calling support
    
    def unique_id(self) -> str:
        """Returns 'provider/model_name' format."""
```

## Routing Algorithm

### Cluster-Based Selection (UniRouter)

1. **Feature Extraction**: 
   - Sentence transformer embeddings (384D) using `all-MiniLM-L6-v2`
   - TF-IDF features (5000D) for lexical patterns
   - StandardScaler normalization
   - Concatenated 5384D feature vector

2. **Cluster Assignment**:
   - K-means prediction using pre-trained cluster centers
   - Fast assignment (<5ms) using scikit-learn

3. **Model Scoring**:
   - Per-cluster error rates for each model
   - Cost normalization
   - Cost-quality tradeoff: `score = predicted_accuracy - ? * normalized_cost`

4. **Selection**:
   - Selects model with highest routing score for assigned cluster
   - Returns alternatives ranked by score

### Cost-Performance Trade-off

The `cost_bias` parameter controls the balance:

- **`cost_bias = 0.0`**: Selects cheapest model (maximum cost savings)
- **`cost_bias = 0.5`**: Balanced selection (default)
- **`cost_bias = 1.0`**: Selects highest quality model (maximum performance)

## Performance

### Latency

- **Feature Extraction**: 10-20ms (CPU) or 5-10ms (GPU-accelerated)
- **Cluster Assignment**: <5ms (K-means prediction)
- **Model Selection**: <5ms (scoring and ranking)
- **Total Latency**: 15-30ms end-to-end per request

### Throughput

- **CPU Mode**: 50-100 requests/second
- **GPU Mode**: 200-500 requests/second (depends on GPU)

### Memory Usage

- **Sentence Transformers**: ~500MB (model cache)
- **Cluster Profiles**: ~10-50MB (depends on number of clusters)
- **Total Memory**: ~1-2GB typical usage

## Development

### Project Structure

```
adaptive_router/
??? adaptive_router/          # Package source
?   ??? __init__.py          # Package exports
?   ??? core/                # Core ML components
?   ?   ??? router.py        # ModelRouter class
?   ?   ??? cluster_engine.py  # ClusterEngine class
?   ?   ??? trainer.py       # Training logic
?   ??? loaders/             # Profile loaders
?   ?   ??? base.py          # ProfileLoader base class
?   ?   ??? local.py          # LocalFileProfileLoader
?   ?   ??? minio.py          # MinIOProfileLoader
?   ??? models/              # Pydantic data models
?   ?   ??? api.py           # Request/response models
?   ?   ??? storage.py       # Profile storage models
?   ?   ??? routing.py       # Routing decision models
?   ??? utils/               # Utility modules
??? tests/                   # Test suite
??? pyproject.toml           # Package configuration
??? README.md                # This file
```

### Running Tests

```bash
# Run all tests
uv run pytest adaptive_router/tests/

# Run unit tests only
uv run pytest adaptive_router/tests/ -m unit

# Run with coverage
uv run pytest adaptive_router/tests/ --cov=adaptive_router --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black adaptive_router/

# Lint code
uv run ruff check adaptive_router/

# Type checking
uv run mypy adaptive_router/
```

## Profile Format

Router profiles are JSON files containing:

```json
{
  "metadata": {
    "n_clusters": 10,
    "model_name": "all-MiniLM-L6-v2",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "models": [
    {
      "provider": "openai",
      "model_name": "gpt-4",
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0
    }
  ],
  "cluster_centers": [...],
  "error_rates": {...},
  "scalers": {...}
}
```

See `train/` directory for training scripts that generate profiles.

## Troubleshooting

### Import Errors

- **Verify installation**: Run `uv install` from project root
- **Check Python version**: Requires Python 3.12+
- **Verify package structure**: Ensure `adaptive_router/adaptive_router/` directory exists

### Model Selection Fails

- **Verify profile loaded**: Check that profile file exists and is valid JSON
- **Check model IDs**: Ensure models in profile have valid `provider/model_name` format
- **Validate cost_bias**: Must be between 0.0 and 1.0
- **Check prompt**: Ensure prompt is non-empty string

### Performance Issues

- **GPU availability**: GPU acceleration requires CUDA-capable GPU and PyTorch with CUDA
- **Model cache**: Sentence transformer models are cached in `~/.cache/huggingface/`
- **Profile size**: Large profiles (many clusters) may increase memory usage

### Common Errors

**`ModelNotFoundError`**: Requested models don't exist in profile or don't match filter criteria

**`InvalidModelFormatError`**: Model IDs in profile are malformed (must be `provider/model_name`)

**Profile loading errors**: Check file path, MinIO credentials, or network connectivity

## Related Documentation

- **Root README** (`../README.md`): General project overview and training instructions
- **App README** (`../app/README.md`): FastAPI application documentation
- **CLAUDE.md**: Detailed technical documentation and architecture

## License

MIT License - see [LICENSE](../LICENSE)

## Support

- [GitHub Issues](https://github.com/Egham-7/adaptive/issues)
- [GitHub Discussions](https://github.com/Egham-7/adaptive/discussions)
