# Nordlys Monorepo

This repository contains the Nordlys runtime router stack and related services.

## Main Package (`nordlys/`)

The `nordlys` Python package now follows a clean split:
- `Dataset`: training data abstraction
- `Trainer`: compiles a checkpoint from labeled data
- `Router`: runtime-only router that loads a checkpoint and routes prompts

Canonical flow:

```python
from nordlys import Dataset, Trainer, Router, ModelConfig

models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-4o-mini", cost_input=0.15, cost_output=0.6),
]

dataset = Dataset.from_list([
    {
        "id": "1",
        "input": "Write a parser for this grammar",
        "targets": {"openai/gpt-4": 1, "openai/gpt-4o-mini": 0},
    }
])

checkpoint = Trainer(models=models).fit(dataset)
router = Router(checkpoint=checkpoint)
result = router.route("Fix this failing test")
```

## Development

```bash
git clone https://github.com/Nordlys-Labs/nordlys
cd nordlys
uv sync
uv run pytest
uv run ruff check .
uv run ty check
```

Requirements: Python 3.11+.

## Links

- Docs: https://docs.nordlyslabs.com
- Issues: https://github.com/Nordlys-Labs/nordlys/issues
- License: MIT
