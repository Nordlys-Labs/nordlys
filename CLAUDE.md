# Nordlys Model Engine

## Memory and Documentation

**IMPORTANT**: When working on this service, remember to:

## Overview

The Nordlys Model Engine is a unified Python ML package that provides intelligent model selection for the Nordlys AI infrastructure. It uses advanced ML techniques including clustering algorithms and per-model performance optimization to select optimal models for requests. The engine supports two deployment modes: Library (import and use directly in Python code) and FastAPI (HTTP API server with GPU-accelerated inference).

## Key Features

- **Mixture of Models Selection**: ML-based selection algorithm with K-means clustering and performance optimization
- **Flexible Deployment**: Python library import or FastAPI HTTP server
- **Cost Optimization**: Balances performance vs. cost based on configurable preferences
- **High-Performance API**: FastAPI framework with Hypercorn ASGI server, OpenAPI documentation
- **Modal Serverless Deployment**: T4 GPU-accelerated inference with automatic scaling to zero
- **Local ML Processing**: Sentence transformers and scikit-learn for feature extraction and clustering

## Technology Stack

- **ML Framework**: PyTorch 2.2+ with CUDA 11.8 for sentence-transformers GPU acceleration
- **Clustering**: scikit-learn for K-means clustering and feature scaling
- **API Framework**: FastAPI 0.118+ for HTTP server mode
- **ASGI Server**: Hypercorn 0.17+ with HTTP/1.1, HTTP/2, and WebSocket support
- **Serverless Deployment**: Modal.com with T4 GPU (16GB VRAM) for cost-efficient inference
- **LLM Integration**: LangChain for orchestration and provider abstraction
- **Configuration**: Pydantic Settings for type-safe configuration management
- **Logging**: Standard Python logging with structured JSON output

## Project Structure

The project uses a **UV workspace** structure with unified dependency management:

1. **`nordlys/`**: Core ML library package (standalone, installable)
2. **`nordlys-backend/`**: FastAPI application (depends on library via workspace dependency)

```
nordlys/  # Repository root (workspace root)
├── pyproject.toml  # Workspace configuration
├── uv.lock  # Unified lockfile for all packages
├── nordlys/                          # Core ML library package
│   ├── __init__.py                           # Library exports for Python import
│   ├── pyproject.toml                        # Library package configuration
│   ├── core/                                 # Core ML selection components
│   │   ├── __init__.py
│   │   ├── selector.py                       # ModelSelector - main selection logic
│   │   ├── cluster_engine.py                 # ClusterEngine - K-means clustering
│   │   └── feature_extractor.py              # FeatureExtractor - sentence transformers + TF-IDF

│   ├── models/                               # Pydantic data models and schemas
│   │   ├── __init__.py
│   │   ├── api.py                            # Request/response models
│   │   ├── config.py                         # YAML configuration models
│   │   ├── evaluation.py                     # Evaluation metrics models
│   │   ├── health.py                         # Health check models
│   │   ├── registry.py                       # Model registry models
│   │   ├── selection.py                     # Selection decision models
│   │   └── storage.py                        # Storage/profile models
│   ├── utils/                                # Utility modules
│   │   ├── __init__.py
│   │   └── model_parser.py                   # Model name parsing utilities
│   └── tests/                                # Test suite (inside package)
│       ├── fixtures/                         # Test fixtures
│       │   ├── ml_fixtures.py
│       │   ├── model_fixtures.py
│       │   └── request_fixtures.py
│       ├── integration/                      # Integration tests
│       │   ├── test_api_endpoints.py
│       │   ├── test_cost_optimization.py
│       │   ├── test_model_selection_flows.py
│       │   └── test_task_selection.py
│       └── unit/                             # Unit tests
│           ├── models/
│           └── services/
├── nordlys-backend/                      # FastAPI HTTP server (separate package)
│   ├── __init__.py
│   ├── main.py                               # FastAPI entry point (inside package)
│   ├── pyproject.toml                        # App package configuration
│   ├── railway.json                          # Railway deployment configuration
│   ├── config.py                             # App configuration (env vars)
│   ├── health.py                             # Health check endpoints
│   ├── models.py                             # API-specific models
│   ├── registry/                             # External model registry integration
│   │   ├── __init__.py
│   │   ├── client.py                         # HTTP client for registry API
│   │   └── models.py                         # Registry model cache
│   └── utils/                                # App-specific utilities
│       ├── fuzzy_matching.py                 # Fuzzy model name matching
│       └── model_resolver.py                 # Model resolution logic
├── scripts/                                  # Utility scripts
│   ├── models/                               # Model management scripts
│   ├── training/                             # Training scripts
│   └── utils/                                # Utility scripts
├── train/                                    # Training scripts
├── uv.lock                                   # Dependency lock file (at root for workspace)
└── README.md                                 # Service documentation
```

**Package Dependencies:**

- The library (`nordlys/`) has its own `pyproject.toml` with ML dependencies (PyTorch, sentence-transformers, scikit-learn, etc.)
- The app (`pyproject.toml` root) depends on `nordlys` via local path dependency
- Both packages are installed in editable mode during development

## Environment Configuration

### Required Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0                     # Server host
PORT=8000                        # Server port

# FastAPI Configuration
FASTAPI_WORKERS=1                # Number of workers
FASTAPI_RELOAD=false             # Auto-reload on code changes (dev only)
FASTAPI_ACCESS_LOG=true          # Enable access logging
FASTAPI_LOG_LEVEL=info           # Log level

# Profile Storage Configuration (Modal Volume)
PROFILE_PATH=/data/profile.json  # Path to selector profile in Modal Volume
```

### Optional Configuration

```bash
# Debugging
DEBUG=false                      # Enable debug logging
LOG_LEVEL=INFO                  # Logging level

# Hypercorn-Specific Configuration (Optional)
# For HTTP/2 support (requires SSL/TLS):
# HYPERCORN_CERTFILE=cert.pem     # SSL certificate file
# HYPERCORN_KEYFILE=key.pem       # SSL private key file
```

## Deployment Modes

### 1. Library Mode (Python Import)

Use nordlys as a Python library in your code:

```python
from nordlys.core.router import ModelRouter
from nordlys.models.api import ModelSelectionRequest

# Initialize router from local file
router = ModelRouter.from_json_file("/path/to/profile.json")

# Select optimal model based on prompt
request = ModelSelectionRequest(
    prompt="Write a Python function to sort a list",
    cost_bias=0.5  # 0.0 = cheapest, 1.0 = most capable
)
response = router.select_model(request)
print(f"Selected: {response.model_id}")
print(f"Alternatives: {[alt.model_id for alt in response.alternatives]}")
```

### 2. FastAPI Server Mode (Modal Deployment)

Run as HTTP API server on Modal with T4 GPU acceleration:

```bash
# Install dependencies (sync workspace from root)
uv sync

# Deploy to Modal (requires Modal CLI and account)
# Deploy from repository root - Modal will use workspace structure
modal deploy nordlys_app/nordlys_app/main.py

# Or run locally in development
fastapi dev nordlys_app/nordlys_app/main.py

# Server starts on http://0.0.0.0:8000
# API docs available at http://localhost:8000/docs
# ReDoc available at http://localhost:8000/redoc
# Health check at http://localhost:8000/health
```

**API Endpoints:**

- `POST /select-model` - Select optimal model based on prompt
- `GET /health` - Health check endpoint

**Modal Deployment Configuration:**

- **GPU**: NVIDIA T4 (16GB VRAM) for GPU-accelerated embedding computation
- **Memory**: 8GB baseline memory allocation
- **Scaling**: Automatically scales to 0 when idle (60-second timeout)
- **Concurrency**: Handles up to 100 concurrent requests per container
- **Storage**: Modal Volume at `/data` for selector profile persistence
- **Model Cache**: Modal Volume at `/root/.cache` for sentence-transformer models

Access interactive API docs at `http://localhost:8000/docs`

## Development Commands

### Workspace Structure

The project uses a **UV workspace** for unified dependency management:

- **Single lockfile**: `uv.lock` at repository root ensures consistent dependencies
- **Workspace members**: `nordlys` (library) and `nordlys_app` (FastAPI app)
- **Inter-package dependencies**: `nordlys_app` depends on `nordlys` via `workspace = true`
- **Modal deployment**: Workspace is copied to `/root/nordlys` and synced from app package

### Local Development

```bash
# Install dependencies (library + app)
# The project uses a UV workspace - sync from repository root
# The app automatically depends on the library via workspace dependency
uv sync  # Syncs entire workspace from root

# Or sync specific package
uv sync --package nordlys-app

# Run commands for specific package
uv run --package nordlys-app pytest

# Start the FastAPI server (development mode with auto-reload)
fastapi dev nordlys_app/nordlys_app/main.py

# Or use Hypercorn directly (production-like)
hypercorn nordlys_app.main:app --bind 0.0.0.0:8000

# Start with custom configuration
HOST=0.0.0.0 PORT=8001 hypercorn nordlys_app.main:app --bind 0.0.0.0:8001

# Start with debug logging
DEBUG=true hypercorn nordlys_app.main:app --bind 0.0.0.0:8000

# For multi-process deployment
hypercorn nordlys_app.main:app --bind 0.0.0.0:8000 --workers 4
```

### Code Quality

```bash
# Format code with Black
uv run black .

# Lint with Ruff
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking with ty
uv run ty check

# Run all quality checks
uv run black . && uv run ruff check . && uv run ty check
```

### Testing

We provide multiple convenient ways to run tests:

#### Using Make Commands (Recommended)

```bash
# Show all available commands
make help

# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage report
make test-cov

# Run with HTML coverage report
make test-cov-html

# Run specific test categories
make test-config      # Configuration tests
make test-services    # Service tests
make test-models      # Model tests
make test-selection    # Selection integration tests

# Code quality
make lint            # Check with ruff
make lint-fix        # Fix issues
make format          # Format with black
make typecheck       # Type checking
make quality         # All quality checks
```

#### Using Shell Script

```bash
# Run all tests
./scripts/test.sh

# Run unit tests only
./scripts/test.sh unit

# Run integration tests only
./scripts/test.sh integration

# Run with coverage
./scripts/test.sh coverage

# Clean test artifacts
./scripts/test.sh clean
```

#### Using uv run directly

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest nordlys/tests/unit/core/test_config.py

# Run with verbose output
uv run pytest -v

# Run tests and generate HTML coverage report
uv run pytest --cov --cov-report=html
```

## API Interface

The service exposes a FastAPI REST API that accepts model selection requests and returns orchestration responses. The API includes automatic OpenAPI documentation available at `/docs`.

### Request Format

```python
{
    "chat_completion_request": {
        "messages": [
            {"role": "user", "content": "Write a Python function to sort a list"}
        ],
        "model": "gpt-4",
        "temperature": 0.7
    },
    "nordlys": {
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
                "max_context_tokens": 128000,
                "supports_function_calling": true
            },
            {
                "provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229",
                "cost_per_1m_input_tokens": 15.0,
                "cost_per_1m_output_tokens": 75.0,
                "max_context_tokens": 200000,
                "supports_function_calling": true
            }
        ],
        "cost_bias": 0.5,
        "complexity_threshold": 0.7,
        "token_threshold": 1000
    },
    "user_preferences": {
        "preferred_providers": ["openai", "anthropic"],
        "cost_optimization": true,
        "quality_preference": "high"
    }
}
```

### Response Format

```python
{
    "selection": {
        "provider": "openai",
        "model": "gpt-4",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "alternatives": [
            {
                "provider": "anthropic",
                "model": "claude-3-sonnet",
                "cost_ratio": 0.85,
                "reason": "fallback_option"
            }
        ],
        "reasoning": "Selected GPT-4 for balanced cost-performance trade-off based on prompt complexity analysis"
    }
}
```

## Core Services

### Model Selector

**File**: `nordlys/core/selector.py`

The `ModelSelector` class is the main entry point for intelligent model selection:

- Cluster-based selection using mixture of models algorithm
- Accepts `SelectionRequest` with prompt and cost preference
- Returns `SelectionResponse` with selected model and reasoning
- Uses pre-loaded cluster profiles from MinIO S3
- Combines feature extraction, cluster assignment, and cost optimization
- Supports multiple provider models with per-cluster error rates

### Cluster Engine

**File**: `nordlys/core/cluster_engine.py`

The `ClusterEngine` handles K-means clustering operations:

- Loads pre-trained cluster centers from storage profiles
- Assigns prompts to clusters using K-means prediction
- Manages cluster metadata and silhouette scores
- Provides fast cluster assignment (<5ms per request)
- Supports configurable number of clusters (typically 10-50)

### Feature Extractor

**File**: `nordlys/core/feature_extractor.py`

The `FeatureExtractor` converts prompts to feature vectors:

- **Sentence transformer embeddings** using `all-MiniLM-L6-v2` (384D)
- **TF-IDF features** for lexical patterns (5000D)
- **StandardScaler normalization** for both feature types
- **Concatenated 5384D feature vectors**
- **GPU-accelerated inference** on T4 GPUs (Modal deployment)
- **Cached models** for fast subsequent requests

### External Model Registry Integration

**Files**: `app/registry/`

Integration with external model registry service for model metadata:

**Registry Client** (`app/registry/client.py`):

- HTTP client for model registry API
- Fetches model metadata (pricing, capabilities, context limits)
- Supports provider/model validation
- Caching layer for performance

**Registry Models** (`app/registry/models.py`):

- Model cache with TTL expiration
- Provider model listings
- Fuzzy matching support for model resolution
- Replaces legacy YAML-based model database

### Model Resolution Utilities

**Files**: `app/utils/`

**Fuzzy Matching** (`app/utils/fuzzy_matching.py`):

- Fuzzy model name matching using string similarity
- Handles common typos and variations
- Provider-specific model name normalization

**Model Resolver** (`app/utils/model_resolver.py`):

- Resolves model names to canonical forms
- Integrates registry client with fuzzy matching
- Provides fallback strategies for unknown models

## Selection Algorithm

### Cluster-Based Selection

- **Algorithm**: K-means clustering of prompts based on semantic features
- **Features**: Sentence transformer embeddings (384D) + TF-IDF features (5000D)
- **Clusters**: Prompts grouped into K clusters (configurable, typically K=10-50)
- **Error Rates**: Each model has per-cluster error rates learned from historical data
- **Selection**: Combines error rates + cost + user preferences to rank models

### Feature Extraction

- **Embeddings**: Sentence transformers (all-MiniLM-L6-v2) for semantic similarity
- **TF-IDF**: Term frequency-inverse document frequency for lexical patterns
- **Scaling**: StandardScaler normalization for both feature types
- **Concatenation**: Combined 5384D feature vector per prompt

### Cost-Performance Trade-off

- **Cost Preference**: λ parameter (0.0 = cheapest, 1.0 = most accurate)
- **Selection Score**: Weighted combination of predicted accuracy and normalized cost
- **Formula**: `score = predicted_accuracy - λ * normalized_cost`
- **Optimization**: Selects model with highest score for assigned cluster

## Caching and Performance

### API Performance

- **Framework**: FastAPI with async/await for optimal performance
- **ASGI Server**: Hypercorn with HTTP/1.1 and HTTP/2 support
- **CORS**: Configured middleware for cross-origin requests
- **Error Handling**: Global exception handlers with structured logging

### Storage Integration

- **Modal Volumes**: Stores cluster profiles and model cache for persistence
- **Profile Caching**: Selector profile loaded once at service initialization
- **Model Cache**: Sentence transformers cached in `/root/.cache` for fast startup
- **Request Processing**: All computations done locally in-memory

### Request Processing

- **Feature Extraction**: GPU-accelerated sentence transformers + TF-IDF on T4 GPU
- **Cluster Assignment**: Fast K-means predict using pre-loaded centroids
- **Model Selection**: In-memory scoring of models based on cluster assignment
- **Response Time**: <100ms end-to-end for typical requests (T4 GPU acceleration)
- **Throughput**: 100+ requests/second on T4 GPU

## Configuration

### Model Metadata Configuration

Model metadata (pricing, capabilities, context limits) is now fetched from an **external model registry service** via HTTP API, replacing the legacy YAML-based configuration system.

**Registry Client Configuration** (`app/config.py`):

- `REGISTRY_API_URL` - Base URL for model registry service
- `REGISTRY_CACHE_TTL` - Cache time-to-live for model metadata (seconds)
- `REGISTRY_TIMEOUT` - HTTP request timeout for registry calls (seconds)

**Example Registry Response**:

```json
{
  "provider": "openai",
  "model_name": "gpt-4",
  "cost_per_1m_input_tokens": 30.0,
  "cost_per_1m_output_tokens": 60.0,
  "max_context_tokens": 128000,
  "supports_function_calling": true,
  "capabilities": ["general", "code", "analysis"]
}
```

### Cluster Profile Configuration

Cluster profiles (centers, error rates, scalers) are stored in **Modal Volumes** for serverless deployment:

**Profile Storage Configuration**:

- `PROFILE_PATH` environment variable points to the profile JSON file
- Default location in Modal: `/data/profile.json` (mounted to nordlys-data volume)

**Profile Structure**:

- JSON format containing model configurations and metadata
- Loaded once at service startup
- Persisted in Modal Volume for durability across deployments

## Monitoring and Observability

### Metrics

- **Classification Accuracy**: Track prediction confidence and user feedback
- **Model Selection Success**: Monitor downstream API success rates
- **Cost Optimization**: Track actual vs. projected cost savings
- **Performance**: Request latency, throughput, error rates

### Logging

- **Structured Logging**: JSON-formatted logs with request correlation
- **Classification Results**: Log task types, domains, and confidence scores
- **Model Selection**: Log selected providers, models, and reasoning
- **Performance Metrics**: Log inference times and batch processing stats

### Health Checks

- **Model Loading**: Verify all ML models are loaded and functional
- **Memory Usage**: Monitor memory consumption and detect leaks
- **Cache Performance**: Track cache hit rates and eviction patterns
- **Dependency Health**: Verify HuggingFace Hub connectivity

## Deployment

### Docker

The service is deployed on Modal.com with GPU acceleration rather than Docker. For local development:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

COPY . .
EXPOSE 8000

CMD ["fastapi", "dev", "nordlys_app/nordlys_app/main.py"]
```

### Modal Deployment

**Production Deployment** (GPU-accelerated on Modal.com):

```bash
# Deploy to Modal
modal deploy nordlys_app/nordlys_app/main.py

# View logs
modal logs nordlys

# Stop deployment
modal cancel nordlys
```

**Modal Configuration** (in `nordlys_app/nordlys_app/main.py`):

- GPU: T4 (16GB VRAM)
- Memory: 8GB
- CPU: Shared resources via Modal
- Scaling: min_containers=0, scales down after 60 seconds of inactivity
- Concurrency: 100 max concurrent inputs per container
- Volumes: Model cache and profile data persisted across deployments

### Resource Requirements

#### Library Mode

- **CPU**: 2-4 cores for feature extraction and clustering
- **GPU**: Optional (NVIDIA GPU with CUDA for acceleration)
- **Memory**: 2-4GB RAM for sentence transformers and scikit-learn
- **Storage**: 1-2GB for HuggingFace model cache (~500MB for all-MiniLM-L6-v2)

#### FastAPI Server Mode (Modal Deployment)

- **GPU**: NVIDIA T4 (16GB VRAM) - required for production
- **CPU**: Shared resources via Modal platform
- **Memory**: 8GB baseline allocation
- **Storage**: Modal Volumes for profile and model cache persistence
- **Network**: Outbound connectivity for HuggingFace Hub (first startup only)

### Hypercorn Benefits

- **Protocol Support**: HTTP/1.1, HTTP/2, WebSockets out of the box
- **Future-Ready**: HTTP/3 support available with `hypercorn[h3]` extra
- **Graceful Shutdown**: Built-in signal handling and graceful shutdown support
- **Sans-IO Architecture**: Modern implementation using sans-io hyper libraries

### Modal Benefits (Production Deployment)

- **Automatic Scaling**: Scales from 0 to N containers based on load
- **Cost Efficiency**: Pay only for compute when handling requests
- **GPU Support**: Easy GPU provisioning without infrastructure management
- **Built-in Caching**: Modal Volumes for persistent storage across deploys
- **Serverless**: No container management, automatic deployment and scaling

## Troubleshooting

### Common Issues

**Service won't start**

- Check Python version (3.11+ required)
- Verify all dependencies installed: `uv install`
- Check port availability (default: 8000)
- For Modal deployment: verify Modal CLI is installed and authenticated
- Ensure you're using the correct command: `fastapi dev nordlys_app/nordlys_app/main.py` (local) or `modal deploy nordlys_app/nordlys_app/main.py` (Modal)

**Modal deployment issues**

- Verify Modal account and authentication: `modal token new`
- Check volume creation: `modal volume ls`
- Review deployment logs: `modal logs <app_name>`
- Ensure CUDA wheels installed: `pip list | grep torch`

**Model loading failures**

- Verify sentence-transformers is installed: `uv sync`
- Check HuggingFace Hub connectivity for model downloads
- For GPU deployment: verify CUDA 11.8 wheels installed
- Check available disk space for model cache (~500MB for all-MiniLM-L6-v2)
- On macOS, CPU mode is used automatically (no CUDA required)

**Selection errors**

- Verify input format matches ModelSelectionRequest schema
- Check prompt length is reasonable (no hard limit, but very long prompts are slower)
- Ensure selector profile loaded correctly (check startup logs)
- Enable debug logging: `DEBUG=true fastapi dev nordlys_app/nordlys_app/main.py`

**Performance issues**

- Monitor GPU utilization on T4 (Modal deployment)
- Check memory usage (sentence transformers ~1-2GB)
- Verify profile loaded successfully at startup (check logs)
- Review Modal concurrent request limits if handling high load
- For local development: consider using CUDA if GPU available

### Debug Commands

**Library Mode:**

```bash
# Test model router
python -c "
from nordlys.core.router import ModelRouter
from nordlys.models.api import ModelSelectionRequest

router = ModelRouter.from_json_file('/path/to/profile.json')
request = ModelSelectionRequest(prompt='Explain quantum computing', cost_bias=0.5)
response = router.select_model(request)
print(f'Selected: {response.model_id}')
"

# Check CUDA availability on GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

**FastAPI Server Mode (Local):**

```bash
# Start with debug logging
DEBUG=true fastapi dev nordlys_app/nordlys_app/main.py

# Check service health
curl -X GET http://localhost:8000/health

# Test model selection endpoint
curl -X POST http://localhost:8000/select-model \
 -H "Content-Type: application/json" \
 -d '{"prompt": "Write a sorting algorithm", "cost_bias": 0.5}'
```

**Modal Deployment:**

```bash
# Deploy to Modal
modal deploy nordlys_app/nordlys_app/main.py

# View logs
modal logs nordlys

# Check GPU availability
modal run nordlys -c "python -c 'import torch; print(torch.cuda.is_available())'"
```

## Performance Benchmarks

### FastAPI Mode (T4 GPU - Modal Deployment)

- **Feature Extraction**: 10-20ms per request (GPU-accelerated embeddings)
- **Cluster Assignment**: <5ms (K-means predict on pre-loaded centroids)
- **Model Selection**: <5ms (scoring and ranking models)
- **Total Latency**: 15-30ms end-to-end per request (GPU acceleration)
- **Throughput**: 100-500 requests/second (T4 GPU dependent)
- **Memory Usage**: 2-4GB (sentence transformers + cluster profiles)
- **First Request**: Slower (~5-10s) due to loading models from HuggingFace

### Startup Performance

- **Profile Load**: Instant (from Modal Volume)
- **Model Download**: 5-10 seconds first time (HuggingFace cache), instant after that
- **GPU Initialization**: 2-5 seconds (CUDA/cuDNN initialization)
- **Total Startup**: 5-15 seconds depending on cache state and GPU availability

### Routing Quality

- **Cost Savings**: 30-70% compared to always using most capable models
- **Accuracy Retention**: >90% of optimal model selection vs. oracle selection
- **Cluster Silhouette**: Typically 0.3-0.5 (good cluster separation)
- **Per-Cluster Accuracy**: Varies by cluster, tracked in profile metadata

## Contributing

### Code Style

- **Formatting**: Black with 88-character line length
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: ty with strict configuration
- **Import Sorting**: Ruff isort with first-party module recognition

### Testing Requirements

- **Unit Tests**: All services and utilities must have unit tests
- **Integration Tests**: End-to-end testing with mock ML models
- **Performance Tests**: Benchmark classification speed and accuracy
- **Coverage**: Minimum 80% test coverage required

### Documentation Updates

**IMPORTANT**: When making changes to this service, always update documentation:

1. **Update this CLAUDE.md** when:
   - Adding new ML models or selection algorithms
   - Modifying API interfaces or request/response formats
   - Changing environment variables or configuration settings
   - Adding new providers, task types, or domain classifications
   - Updating Python dependencies or ML framework versions
   - Adding new services or modifying existing service logic

2. **Update root CLAUDE.md** when:
   - Changing service ports, commands, or basic service description
   - Modifying the service's role in the intelligent mixture of models architecture
   - Adding new ML capabilities or performance characteristics

3. **Update adaptive-docs/** when:
   - Adding new model selection features
   - Changing cost optimization algorithms
   - Modifying provider integration or selection logic

### Pull Request Process

1. Create feature branch from `dev`
2. Implement changes with comprehensive tests
3. Run full quality checks: `uv run black . && uv run ruff check . && uv run ty check && uv run pytest --cov`
4. **Update relevant documentation** (CLAUDE.md files, adaptive-docs/, README)
5. Submit PR with performance impact analysis and documentation updates
