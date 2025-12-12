# CUDA Package Setup

The CUDA package (`adaptive_router_core_cu12`) is **not** included in the main workspace to avoid dependency conflicts on non-Linux platforms.

## Development Workflow

### CPU Development (macOS, Windows, Linux)
```bash
# Install CPU-only version
uv sync --extra cpu

# This works on all platforms since CUDA dependencies are not resolved
```

### CUDA Development (Linux with CUDA only)
```bash
# Build CUDA package separately
./scripts/build_cuda.sh

# Install CUDA package manually if needed
pip install adaptive_router_core_cu12/dist/adaptive-router-core-cu12-*.whl
```

## Why This Setup?

- **CPU package** is in workspace → works everywhere
- **CUDA package** is separate → only built on Linux with CUDA
- **No platform conflicts** during development
- **Clean separation** of concerns

## CI/CD

In CI, build the CUDA package on Linux runners with CUDA installed:

```yaml
- name: Build CUDA package
  run: ./scripts/build_cuda.sh
  if: matrix.os == 'ubuntu-latest'
```