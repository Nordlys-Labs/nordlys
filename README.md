# Adaptive Router

## Intelligent LLM Model Selection via ML-Powered Clustering

Reduce costs by 30-70% using cluster-based routing with per-cluster error rates and prompt analysis.

[![PyPI](https://img.shields.io/pypi/v/adaptive-router)](https://pypi.org/project/adaptive-router/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-router)](https://pypi.org/project/adaptive-router/)
[![License](https://img.shields.io/github/license/Egham-7/adaptive)](https://github.com/Egham-7/adaptive/blob/main/LICENSE)

## How It Works

1. **Feature Extraction**: Semantic embeddings using SentenceTransformers
2. **Clustering**: Groups similar prompts using K-means
3. **Performance Tracking**: Maintains per-cluster error rates for each model
4. **Smart Selection**: Balances quality vs. cost using configurable `cost_bias`

Prompts with similar semantics often require similar model capabilities. By clustering prompts and tracking performance per cluster, routing becomes data-driven rather than rule-based.

## Quick Start

### Installation

```bash
pip install adaptive-router
```

### Library Usage

**Basic Usage:**

```python
from adaptive_router import ModelRouter, ModelSelectionRequest

# Load router from profile (models included in profile)
router = ModelRouter.from_local_file("router_profile.json")

# Select model
response = router.select_model(
    ModelSelectionRequest(
        prompt="Write a Python function to sort a list",
        cost_bias=0.3  # 0.0=cheapest, 1.0=best quality
    )
)

print(f"Selected: {response.model_id}")  # e.g., "openai/gpt-3.5-turbo"
print(f"Alternatives: {[alt.model_id for alt in response.alternatives]}")
```

**Quick Routing (Just Get Model ID):**

```python
from adaptive_router import ModelRouter

router = ModelRouter.from_local_file("router_profile.json")

# Just get the model ID string
model_id = router.route("Write a sorting algorithm", cost_bias=0.3)
print(model_id)  # "openai/gpt-3.5-turbo"
```

**Advanced Usage with Model Filtering:**

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

### HTTP API

```bash
git clone https://github.com/Egham-7/adaptive
cd adaptive
uv install

# Development
fastapi dev adaptive_router_app/main.py

# Production
hypercorn adaptive_router_app.main:app --bind 0.0.0.0:8000
```

**Endpoints**:
- `POST /select-model` - Select optimal model
- `GET /health` - Health check

**Example Request**:
```json
{
  "prompt": "Explain neural networks",
  "cost_bias": 0.5
}
```

**Example Response**:
```json
{
  "selected_model": "openai/gpt-3.5-turbo",
  "alternatives": ["openai/gpt-4", "anthropic/claude-3-sonnet"]
}
```

## Training Custom Profiles

```bash
# Train from labeled dataset
uv run train/train.py --config train/examples/configs/train_minimal.toml
```

**Minimal TOML config**:
```toml
[dataset]
path = "train/examples/datasets/minimal_qa.csv"
type = "csv"

[[models]]
provider = "openai"
model_name = "gpt-4"

[training]
n_clusters = 5

[output]
path = "profile.json"
storage_type = "local"
```

Dataset requires `input` and `expected_output` columns (CSV, JSON, or Parquet).

**Note**: Training produces a profile containing:
- Cluster centers (from semantic embeddings)
- Model definitions with costs
- Per-cluster error rates for each model
- Scaler parameters for normalization

## API Reference

### ModelRouter

```python
class ModelRouter:
    @classmethod
    def from_profile(cls, profile: RouterProfile) -> ModelRouter:
        """Initialize from profile object"""

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model"""
```

### ModelSelectionRequest

```python
class ModelSelectionRequest(BaseModel):
    prompt: str                   # Text to analyze
    cost_bias: float = None       # 0.0=cheap, 1.0=quality
    models: list[Model] = None    # Constrain to specific models
    user_id: str = None           # User identifier
```

### ModelSelectionResponse

```python
class ModelSelectionResponse(BaseModel):
    model_id: str                     # Selected model (e.g., "openai/gpt-4")
    alternatives: list[Alternative]   # Alternative recommendations
```

## Performance

- **Cost Reduction**: 45% vs. always using GPT-4
- **Quality**: 92% maintain acceptable quality
- **Latency**: <50ms routing (semantic embeddings + clustering)
- **Throughput**: 1000+ requests/second

## Development

The project is structured as a **UV workspace** with two packages:
- **`adaptive_router/`**: Core library package (ML routing logic)
- **`adaptive_router_app/`**: FastAPI application (depends on library)

```bash
# Setup
git clone https://github.com/Egham-7/adaptive
cd adaptive
uv sync  # Syncs entire workspace from root

# Or sync specific package
uv sync --package adaptive-router-app

# The app automatically depends on the library via workspace dependency
# Single uv.lock at root ensures consistent dependencies across packages

# Test (from root)
uv run pytest
uv run pytest --cov

# Or run tests for specific package
uv run --package adaptive-router-app pytest

# Type check & lint
uv run mypy .
uv run ruff check .
```

## FAQ

**Q: How many training samples needed?**  
A: 50-100 per cluster. For 5 clusters: 250-500 samples.

**Q: Can I use without training?**  
A: No, trained profile required.

**Q: What do cost_bias values mean?**  
- `0.0` = cheapest model
- `0.5` = balanced (default)
- `1.0` = highest quality

**Q: Production deployment?**  
A: Yes. Use `hypercorn` or deploy via Railway/Docker.

## Troubleshooting

**Import errors**: 
- Run `uv install` to install dependencies
- The library (`adaptive_router`) is automatically installed as a local editable dependency
- If issues persist, ensure you're in the project root directory

**Model selection fails**: 
- Verify profile contains error rate data
- Check `cost_bias` is 0.0-1.0
- Ensure prompt is non-empty

**Training issues**: Verify dataset has `input` and `expected_output` columns

## License

MIT License - see [LICENSE](LICENSE)

## Support

- [GitHub Issues](https://github.com/Egham-7/adaptive/issues)
- [GitHub Discussions](https://github.com/Egham-7/adaptive/discussions)
