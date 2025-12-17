# Agent Development Guide

## Build/Test Commands

- **Install**: `uv install` (dev: `uv install --all-extras`)
- **Test All**: `uv run pytest`
- **Test Single**: `uv run pytest tests/unit/services/test_model_router.py::TestModelRouter::test_initialization -vv`
- **Coverage**: `uv run pytest --cov --cov-report=html`
- **Run Server**: `fastapi dev adaptive_router_app/main.py` (dev) or `hypercorn adaptive_router_app.main:app --bind 0.0.0.0:8000` (prod)

## Code Style

- **Format**: `uv run black .` (88 chars, Python 3.11+)
- **Lint**: `uv run ruff check .` (fix: `--fix`)
- **Types**: `uv run mypy .` (strict, Pydantic plugin enabled)
- **Imports**: First-party (`adaptive_router`), then third-party, then standard library
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- **Type Hints**: Always use type hints; return types required
- **Error Handling**: Use custom exceptions from `models/`, log with structured logging
- **Docstrings**: Google style for public APIs, explain "why" not "what"
- **Tests**: AAA pattern (Arrange/Act/Assert)

