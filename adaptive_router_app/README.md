# Adaptive Router - Modal Serverless Application

Modal serverless deployment for intelligent LLM model selection using cluster-based routing with per-cluster error rates and cost optimization. The application uses FastAPI as the web framework, deployed on Modal's serverless infrastructure with GPU acceleration.

## Overview

The Adaptive Router Modal application provides a production-ready serverless HTTP API for selecting optimal LLM models based on prompt analysis. It uses cluster-based intelligent routing to balance cost and quality, reducing LLM costs by 30-70% while maintaining high-quality responses.

**Architecture**: Modal serverless platform ? FastAPI web framework ? Adaptive Router library

## Features

- **Modal Serverless**: Primary deployment on Modal with automatic scaling to zero
- **GPU Acceleration**: L40S GPU (48GB VRAM) for fast embedding computation
- **FastAPI Framework**: Modern async Python web framework with automatic OpenAPI documentation
- **Railway Alternative**: Optional containerized deployment via Railway
- **Cost Optimization**: Configurable cost-quality tradeoff via `cost_bias` parameter
- **Model Filtering**: Optional model filtering to restrict selection to specific providers/models
- **Health Monitoring**: Comprehensive health check endpoint with service status
- **Persistent Storage**: Modal Volumes for profile and model cache persistence

## Quick Start

### Installation

```bash
# Install dependencies (from project root)
cd /path/to/adaptive_router
uv install

# The adaptive_router_app package automatically depends on the adaptive_router library
# via local path dependency configured in adaptive_router_app/pyproject.toml
```

### Local Development

```bash
# Start development server with auto-reload
fastapi dev adaptive_router_app/main.py

# Or use Hypercorn directly (production-like)
hypercorn adaptive_router_app.main:app --bind 0.0.0.0:8000

# Server starts on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

### Production Deployment

#### Modal (Primary - Serverless with GPU)

**Modal is the primary deployment platform.** The FastAPI application runs inside Modal's serverless infrastructure.

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Deploy the application
modal deploy adaptive_router_app/main.py

# View logs
modal logs adaptive-router

# Get deployment URL
modal app list

# Stop deployment
modal cancel adaptive-router
```

**Modal Architecture:**
- **Deployment**: `@app.cls()` decorator creates Modal service class
- **ASGI Serving**: `@modal.asgi_app()` decorator serves FastAPI app via Modal's ASGI handler
- **GPU**: L40S (48GB VRAM) for GPU-accelerated embedding computation
- **Scaling**: Automatically scales to 0 when idle (600-second scaledown window)
- **Concurrency**: Up to 100 concurrent requests per container (`@modal.concurrent(max_inputs=100)`)
- **Storage**: Modal Volumes mounted at `/data` (profile) and `/vol/model_cache` (model cache)
- **Secrets**: Modal secrets from `adaptive-router-secrets` for configuration
- **Image**: Custom Debian image with PyTorch CUDA 11.8, FastAPI, and dependencies

**Modal Image Build Workflow:**
- The deploy command **must** be run from the repository root so `../adaptive_router` resolves.
- The image copies both `adaptive_router` (library) and `adaptive_router_app` (FastAPI app) into `/root` with `copy=True` so they're available at build time.
- A single `uv_sync("/root/adaptive_router_app", frozen=True)` step installs all dependencies from `uv.lock` into a virtual environment, ensuring reproducible builds with exact dependency versions. The path dependency on `adaptive_router` is automatically resolved since both directories are in `/root`?no manual `pip install -e .` or extra PyPI indexes required.
- Environment variables (e.g., `SENTENCE_TRANSFORMERS_HOME`) remain available via `.env(...)`, while `PYTHONPATH` no longer needs the library path because both packages are copied to `/root` (which is on PYTHONPATH).

**Modal Volumes:**
- `adaptive-router-data`: Stores router profile JSON at `/data/profile.json`
- `sentence-transformer-cache`: Caches HuggingFace models at `/vol/model_cache`

#### Railway (Alternative - Containerized)

Railway deployment is available as an alternative using Hypercorn ASGI server:

```bash
# Railway automatically detects railway.json and deploys
# Ensure PROFILE_PATH environment variable is set
# Railway runs: hypercorn adaptive_router_app.main:app --bind "[::]:$PORT"
```

**Note**: Railway deployment uses the FastAPI app directly (via `app = create_app()` at bottom of `main.py`), not the Modal wrapper.

## API Endpoints

### POST /select-model

Select optimal model based on prompt analysis.

**Request:**
```json
{
  "prompt": "Write a Python function to sort a list",
  "cost_bias": 0.5,
  "user_id": "user123",
  "models": ["openai/gpt-4", "anthropic/claude-3-sonnet"]
}
```

**Response:**
```json
{
  "selected_model": "openai/gpt-3.5-turbo",
  "alternatives": ["openai/gpt-4", "anthropic/claude-3-sonnet"]
}
```

**Parameters:**
- `prompt` (required): Text prompt to analyze for model selection
- `cost_bias` (optional, default: 0.5): Cost-quality tradeoff (0.0 = cheapest, 1.0 = highest quality)
- `user_id` (optional): User identifier for tracking and analytics
- `models` (optional): List of model IDs to restrict selection (e.g., `["openai/gpt-4"]`)

**Status Codes:**
- `200`: Success
- `400`: Invalid request (invalid models, missing prompt, etc.)
- `500`: Internal server error

### GET /health

Comprehensive health check for router and profile models.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "status": "healthy",
    "message": "5 models loaded from profile"
  },
  "router": {
    "status": "healthy",
    "message": "Router initialized"
  }
}
```

**Status Codes:**
- `200`: All services healthy
- `503`: One or more services unhealthy

## Configuration

### Environment Variables

```bash
# Profile Configuration
PROFILE_PATH=/data/profile.json  # Path to router profile JSON file

# CORS Configuration
ALLOWED_ORIGINS=https://example.com,https://app.example.com  # Comma-separated origins
ENVIRONMENT=production  # development or production

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Development vs Production

**Development Mode** (`ENVIRONMENT=development`):
- Allows all CORS origins (`*`)
- Allows all HTTP methods and headers
- More permissive for local testing

**Production Mode** (`ENVIRONMENT=production`):
- Requires explicit `ALLOWED_ORIGINS` configuration
- Restricts to GET/POST methods
- Restricts to necessary headers only

## Architecture

### Deployment Architecture

```
Modal Platform
  ??? AdaptiveRouterService (@app.cls)
       ??? web_app() (@modal.asgi_app)
            ??? create_app() ? FastAPI
                 ??? Endpoints (/select-model, /health)
                      ??? ModelRouter (adaptive_router library)
```

### Application Structure

```
adaptive_router_app/
??? adaptive_router_app/  # Package directory
?   ??? main.py              # Modal deployment + FastAPI app factory
?   ?                        # - Modal App definition (@app.cls)
?   ?                        # - AdaptiveRouterService class with @modal.asgi_app()
?   ?                        # - create_app() FastAPI factory function
?   ?                        # - Railway fallback (app = create_app())
?   ??? config.py           # Application settings (Pydantic Settings)
?   ??? health.py           # Health check models and endpoints
?   ??? models.py           # API request/response models
?   ??? utils/              # Application utilities
?       ??? model_resolver.py  # Model name resolution and fuzzy matching
??? tests/              # Test suite
??? pyproject.toml      # Package configuration
??? railway.json        # Railway deployment configuration
??? README.md           # This file
```

### Modal Deployment Details

**Modal App Definition** (`adaptive_router_app/main.py` lines 357-415):
- `modal.App("adaptive-router")`: Creates Modal application
- `modal.Volume.from_name()`: Mounts persistent volumes
- `modal.Image.debian_slim()`: Custom image with dependencies
- `@app.cls()`: Defines Modal service class with GPU, volumes, secrets
- `@modal.asgi_app()`: Serves FastAPI app via Modal's ASGI handler
- `@modal.concurrent()`: Enables concurrent request handling

**FastAPI Application** (`create_app()` function):
- Creates FastAPI instance with lifespan management
- Loads router profile from Modal Volume at startup
- Configures CORS middleware
- Defines `/select-model` and `/health` endpoints
- Uses dependency injection for router and models

### Dependencies

- **adaptive-router**: Core library package (local path dependency, installed in Modal image)
- **fastapi**: Web framework (runs inside Modal)
- **pydantic-settings**: Configuration management
- **modal**: Serverless deployment platform (primary deployment method)

### Request Flow (Modal Deployment)

1. **HTTP Request**: Received by Modal's ASGI handler
2. **Modal Routing**: Routes to `AdaptiveRouterService.web_app()` method
3. **FastAPI Processing**: FastAPI receives request via `create_app()` instance
4. **Dependency Injection**: FastAPI resolves `get_router()` and `get_available_models()` dependencies
5. **Model Resolution**: Optional model filtering via `resolve_models()` utility
6. **Router Selection**: `ModelRouter.select_model()` performs cluster-based routing
7. **Response Formatting**: Converts internal response to API response format
8. **Error Handling**: Structured error responses with appropriate status codes
9. **Modal Response**: Modal ASGI handler returns HTTP response

## Performance

### Latency

- **Feature Extraction**: 10-20ms (GPU-accelerated embeddings on L40S)
- **Cluster Assignment**: <5ms (K-means prediction)
- **Model Selection**: <5ms (scoring and ranking)
- **Total Latency**: 15-30ms end-to-end per request

### Throughput

- **Modal Deployment**: 100-500 requests/second (L40S GPU dependent)
- **Concurrency**: Up to 100 concurrent requests per container
- **Scaling**: Automatic scaling from 0 to N containers based on load

### Startup Performance

- **Profile Load**: Instant (from Modal Volume or local file)
- **Model Download**: 5-10 seconds first time (HuggingFace cache), instant after
- **GPU Initialization**: 2-5 seconds (CUDA/cuDNN initialization)
- **Total Startup**: 5-15 seconds depending on cache state

## Monitoring and Observability

### Logging

Structured logging with request correlation:

```python
logger.info(
    "Model selection completed",
    extra={
        "elapsed_ms": 25.3,
        "selected_model_id": "openai/gpt-3.5-turbo",
        "alternatives_count": 2,
    },
)
```

### Health Checks

The `/health` endpoint provides:
- Overall service status
- Router initialization status
- Profile model loading status
- Detailed error messages for troubleshooting

### Metrics to Monitor

- Request latency (p50, p95, p99)
- Error rates (4xx, 5xx)
- Model selection distribution
- Cost savings vs. always using premium models
- GPU utilization (Modal deployment)

## Troubleshooting

### Service Won't Start

**Modal Deployment:**
- **Verify Modal CLI**: Install with `pip install modal`
- **Authenticate**: Run `modal token new` to authenticate
- **Check Modal account**: Ensure you have an active Modal account
- **Verify volumes**: Check that `adaptive-router-data` and `sentence-transformer-cache` volumes exist (`modal volume ls`)
- **Check secrets**: Verify `adaptive-router-secrets` secret exists (`modal secret ls`)
- **View deployment logs**: `modal logs adaptive-router` for detailed error messages

**Local Development:**
- **Check Python version**: Requires Python 3.12+
- **Verify dependencies**: Run `uv install` from project root
- **Check port availability**: Default port 8000

### Model Selection Fails

- **Verify profile loaded**: Check `/health` endpoint for profile status
- **Check model IDs**: Ensure requested models exist in profile
- **Validate request format**: Check API docs at `/docs` for correct schema
- **Enable debug logging**: Set `LOG_LEVEL=DEBUG` for detailed logs

### Performance Issues

**Modal Deployment:**
- **Monitor GPU utilization**: Check Modal dashboard for L40S GPU usage
- **Check container scaling**: Verify containers are scaling appropriately (not stuck at 0)
- **Review concurrent requests**: Modal limits to 100 concurrent inputs per container (`@modal.concurrent(max_inputs=100)`)
- **Profile loading**: Ensure profile is cached in Modal Volume (`adaptive-router-data`) for fast startup
- **Model cache**: Verify sentence-transformer models are cached in `sentence-transformer-cache` volume
- **Cold start latency**: First request after scale-to-zero may take 5-15 seconds (GPU initialization + model loading)
- **Warm container performance**: Subsequent requests should be <30ms end-to-end

**Local Development:**
- **Check memory usage**: Sentence transformers require 2-4GB RAM
- **GPU availability**: Local development uses CPU mode (no GPU required)

### Common Errors

**400 Bad Request - "Unable to resolve requested models"**
- Model IDs in request don't match profile models
- Check model ID format: `provider/model-name` (e.g., `openai/gpt-4`)

**503 Service Unhealthy - "Router profile not loaded"**
- Profile file missing or invalid
- Check `PROFILE_PATH` environment variable
- Verify profile JSON is valid and accessible

**500 Internal Server Error**
- **Modal logs**: Check `modal logs adaptive-router` for detailed error messages
- **Profile loading**: Verify profile JSON exists in Modal Volume at `/data/profile.json`
- **GPU availability**: Ensure L40S GPU is available in your Modal account (check quotas)
- **Dependencies**: Verify all dependencies installed correctly in Modal image (check image build logs)
- **Memory limits**: Check if container is hitting memory limits (L40S has 48GB VRAM, but container memory may be limited)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest adaptive_router_app/tests/

# Run with coverage
uv run pytest adaptive_router_app/tests/ --cov=adaptive_router_app --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black adaptive_router_app/adaptive_router_app/

# Lint code
uv run ruff check adaptive_router_app/adaptive_router_app/

# Type checking
uv run mypy adaptive_router_app/adaptive_router_app/
```

### Adding New Endpoints

1. Add endpoint function to `adaptive_router_app/main.py`
2. Define request/response models in `adaptive_router_app/models.py`
3. Add tests in `tests/`
4. Update this README with endpoint documentation

## Related Documentation

- **Root README**: General project overview and training instructions
- **Library README** (`../adaptive_router/README.md`): Core library package documentation
- **CLAUDE.md**: Detailed technical documentation and architecture

## License

MIT License - see [LICENSE](../LICENSE)

## Support

- [GitHub Issues](https://github.com/Egham-7/adaptive/issues)
- [GitHub Discussions](https://github.com/Egham-7/adaptive/discussions)
