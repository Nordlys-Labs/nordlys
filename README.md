# Nordlys Model Engine

Nordlys is an AI lab building a Mixture of Models. This repository contains the internal engine and tooling that power Nordlys models.

## For Users

Use the public API or SDKs to access Nordlys models. Default model ID:

- `nordlys/nordlys-code`

## For Contributors

Development setup:

```bash
git clone https://github.com/Egham-7/nordlys
cd nordlys
uv sync --package nordlys
```

Requirements: Python 3.11+, CMake, optional CUDA 12.x.

Run tests:

```bash
uv run pytest nordlys/tests/
```

## Links

- Docs: https://docs.llmadaptive.uk
- Issues: https://github.com/Egham-7/nordlys/issues
- License: MIT (see LICENSE)
