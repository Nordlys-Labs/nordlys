# Contributing to Nordlys

Thank you for your interest in contributing to Nordlys! This guide will help you get started with contributing to the intelligent mixture of models engine.

## Overview

Nordlys is a sophisticated ML selection system with two main components:
- **Python Package**: High-level ML API with selection algorithms
- **C++ Core**: High-performance inference engine with language bindings

## Prerequisites

### Core Requirements
- Python 3.11+
- UV package manager
- CMake 3.24+ (for C++ development)
- C++20 compatible compiler

### Optional Requirements
- CUDA Toolkit 12.x (for GPU acceleration, Linux only)
- Conan 2.0+ (for C++ dependency management)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Nordlys-Labs/nordlys.git
cd nordlys

# Install Python dependencies
uv sync

# Run tests
uv run pytest
```

## Development Workflow

### 1. Setting Up Your Development Environment

#### Python Development
```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

#### C++ Development (Optional)
```bash
cd nordlys-core

# Install C++ dependencies
conan install . --build=missing -s compiler.cppstd=20

# Configure CMake
cmake --preset conan-release -DNORDLYS_BUILD_TESTS=ON

# Build
cmake --build .
```

### 2. Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Test your changes thoroughly

4. Commit with a clear message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. Push and create a pull request

### 3. Code Quality

Run these commands before committing:
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run ty check .

# Run tests
uv run pytest
```

## Coding Standards

### Python Code
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings
- Use `uv` for package management
- Follow existing naming conventions

#### Example Python Code
```python
from typing import List, Optional
import numpy as np

def route_prompt(
    prompt: str,
    models: List[str],
    cost_bias: float = 0.5,
) -> Optional[str]:
    """Route a prompt to the optimal model.

    Args:
        prompt: The input prompt to route
        models: List of available model IDs
        cost_bias: Cost preference (0.0=quality, 1.0=cost)

    Returns:
        Selected model ID or None if routing fails
    """
    # Implementation here
    pass
```

### C++ Code
- Use C++20 features appropriately
- Follow Google C++ style guidelines
- Use RAII and smart pointers
- Write comprehensive comments

#### Example C++ Code
```cpp
#include <memory>
#include <vector>
#include <string>

class Router {
public:
    // Constructor with clear documentation
    explicit Router(const std::string& profile_path);

    // Use smart pointers for memory management
    auto Route(const Embedding& embedding, float cost_bias)
        -> std::unique_ptr<RouteResult>;

private:
    // Private members with trailing underscore
    std::unique_ptr<RouterProfile> profile_;
    bool is_gpu_enabled_{false};
};
```

## Testing

### Python Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_router.py

# Run with specific marker
uv run pytest -m unit
```

### C++ Tests
```bash
# Build and run tests
cd nordlys-core/build
cmake --build . --target test

# Run with CTest
ctest --output-on-failure
```

## Project Structure

```
nordlys/
├── nordlys/                  # Python package
│   ├── src/nordlys/         # Source code
│   │   ├── core/            # Core ML algorithms
│   │   ├── models/          # Data models
│   │   └── utils/           # Utilities
│   ├── tests/               # Test suite
│   └── pyproject.toml       # Package config
├── nordlys-core/            # C++ core library
│   ├── core/                # C++ source
│   ├── bindings/            # Language bindings
│   └── tests/               # C++ tests
├── scripts/                 # Utility scripts
└── .github/workflows/       # CI/CD
```

## Areas to Contribute

### High Priority
1. **Algorithm Improvements**
   - Enhance selection accuracy
   - Optimize clustering algorithms
   - Add new features for model selection

2. **Performance Optimizations**
   - GPU acceleration improvements
   - Memory usage optimizations
   - Batch processing enhancements

3. **Provider Integrations**
   - Add support for new LLM providers
   - Improve model metadata handling
   - Add provider-specific optimizations

### Medium Priority
1. **Documentation**
   - Improve API documentation
   - Add tutorials and examples
   - Create architectural diagrams

2. **Testing**
   - Add comprehensive integration tests
   - Performance benchmarking
   - Stress testing

3. **Developer Experience**
   - Improved error messages
   - Better debugging tools
   - CLI utilities

## Submitting Changes

### Pull Request Requirements
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains changes

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Getting Help

1. **Documentation**: Check the `/docs` directory and README files
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Start a GitHub discussion for questions
4. **Code Review**: Request help in your PR

## Release Process

Releases are automated through GitHub Actions:
1. Bump version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag
4. CI/CD builds and publishes to PyPI

## Performance Guidelines

When contributing performance-critical code:

1. **Profile first**: Use profilers to identify bottlenecks
2. **Benchmark**: Add performance tests
3. **Document**: Explain optimization decisions
4. **Test**: Verify no regressions

Example benchmark code:
```python
def benchmark_routing():
    """Benchmark routing performance."""
    router = ModelRouter.from_file("profile.json")

    start_time = time.perf_counter()
    for _ in range(1000):
        router.route("test prompt")

    elapsed = time.perf_counter() - start_time
    print(f"1000 routes in {elapsed:.3f}s")
    print(f"Average: {elapsed/1000*1000:.3f}ms per route")
```

## Security Considerations

- Never commit API keys or secrets
- Validate all inputs
- Use secure defaults
- Follow OWASP guidelines for web components
- Report security issues privately

## License

By contributing to Nordlys, you agree that your contributions will be licensed under the same MIT license as the project.