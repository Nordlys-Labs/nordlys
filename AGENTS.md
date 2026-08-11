# Agent Development Guide

## Repository Layout

Independent uv/CMake projects (no root workspace):

- `nordlys/` — Python ML router library (`nordlys` package)
- `nordlys-core/` — C++20 inference core (CMake + Conan 2); nanobind Python bindings in `nordlys-core/bindings/python/`
- `nordlys-core-cu12/`, `nordlys-core-cu13/` — CUDA wheel variants of the core
- `runners/` — Modal self-hosted CI runner pools (see `runners/README.md`)

## Build/Test Commands

### Python library (all from `nordlys/`)

- **Install**: `uv sync --group dev --extra cpu`
- **Test All**: `uv run pytest tests -v --tb=short`
- **Single Test**: `uv run pytest tests/unit/test_checkpoint.py -vv`
- **Coverage**: `uv run pytest tests --cov --cov-report=html`
- **Build**: `uv build`

### C++ core (all from `nordlys-core/`)

- **Install deps**: `uv pip install conan --system && conan profile detect --force && conan install . --build=missing -s compiler.cppstd=20`
- **Configure**: `cmake --preset conan-release -DNORDLYS_BUILD_TESTS=ON -DNORDLYS_BUILD_PYTHON=ON -DNORDLYS_BUILD_C=ON -DNORDLYS_ENABLE_CUDA=OFF`
- **Build/Test**: `cmake --build --preset conan-release --parallel && ctest --preset conan-release --output-on-failure -E python_bindings`
- **Bindings tests**: `cd bindings/python && uv pip install -e ".[test]" --system && pytest tests/ -v`

### CI runners (from `runners/`)

- **Install**: `uv sync`
- **Deploy pools**: `uv run modal deploy cpu_app.py` (also `core_app.py`, `gpu_app.py`)
- Secrets/webhooks/teardown runbook: `runners/README.md`

## Code Style

Lint commands run from `nordlys/` (CI lints that package only):

- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .` (fix: `--fix`)
- **Types**: `uv run ty check` (strict)
- **Imports**: first-party (`nordlys`), then third-party, then standard library
- **Naming**: `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants
- **Type hints**: always used; return types required
- **Docstrings**: Google style for public APIs; explain "why", not "what"
- **Tests**: AAA pattern (Arrange/Act/Assert)

## CI Notes

- Linux x86_64 legs run on the Modal pools (`runs-on: [self-hosted, modal, ci|ci-core|ci-gpu, job-<unique pin>]`) deployed from `runners/`. Each matrix leg's pin **must** end in `${{ strategy.job-index }}` or legs collide.
- macOS/Windows legs, cibuildwheel wheel builds, and PyPI publish jobs stay on hosted runners (Modal is Linux-only; cibuildwheel needs nested Docker; publish holds OIDC/PyPI credentials).
- The `conan_home` matrix field points Modal core legs at the shared `/cache/conan2` Volume; `actions/cache` for conan is skipped on self-hosted runners.
- The `linux-gpu` leg (T4, `--extra cu12`) is `continue-on-error` until the CUDA-guarded tests are proven green.
