# Nordlys Model Engine

Nordlys is an AI lab building a Mixture of Models. This repository contains the internal engine and tooling that power Nordlys models.

## What is Nordlys?

Nordlys is an intelligent model selection system that:
- **Selects optimal LLMs** based on prompt analysis using ML clustering
- **Optimizes costs** while maintaining quality through sophisticated selection algorithms
- **Supports multiple providers** (OpenAI, Anthropic, local models, etc.)
- **Provides high-performance inference** with GPU acceleration

## For Users

Use the public API or SDKs to access Nordlys models. Default model ID:
- `nordlys/nordlys-code`

## For Contributors

### Python Package Development

```bash
git clone https://github.com/Nordlys-Labs/nordlys
cd nordlys
uv sync --package nordlys
```

Requirements: Python 3.11+.

Run tests:
```bash
uv run pytest nordlys/tests/
```

### C++ Core Development

For C++ development (including C bindings):

```bash
cd nordlys-core
conan install . --build=missing -s compiler.cppstd=20
cmake --preset conan-release -DNORDLYS_BUILD_C=ON
cmake --build . --target nordlys_c
```

This builds:
- `libnordlys_core.a` - Core C++ library
- `libnordlys_c.so` - C FFI bindings

## Package Variants

- **nordlys**: Python package (CPU-only)
- **nordlys[cu12]**: Python package with CUDA 12.x support (Linux)
- **nordlys-core**: C++ core library
- **nordlys-core-cu12**: C++ core with CUDA 12.x (Linux)

## Development Commands

```bash
# Install all dependencies
uv sync

# Code quality checks
uv run ruff check .
uv run ruff format .
uv run ty check

# Run tests with coverage
uv run pytest --cov

# Build documentation
uv run mkdocs serve
```

## Links

- **Documentation**: https://docs.llmadaptive.uk
- **Issues**: https://github.com/Nordlys-Labs/nordlys/issues
- **License**: MIT (see LICENSE)
