# Nordlys

Smart LLM model routing with a checkpoint-based runtime.

## Install

```bash
uv pip install -e .
```

## Quick Start

```python
from nordlys import Dataset, Trainer, Router, ModelConfig

# 1. Define models
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-4o-mini", cost_input=0.15, cost_output=0.6),
]

# 2. Build training dataset with binary targets per model
dataset = Dataset.from_list([
    {
        "id": "1",
        "input": "Design a database schema for this app",
        "targets": {"openai/gpt-4": 1, "openai/gpt-4o-mini": 0},
    },
    {
        "id": "2",
        "input": "Summarize this short changelog",
        "targets": {"openai/gpt-4": 0, "openai/gpt-4o-mini": 1},
    },
])

# 3. Train checkpoint, then create runtime router
checkpoint = Trainer(models=models).fit(dataset)
router = Router(checkpoint=checkpoint)

result = router.route("Implement this parser")
print(result.model_id)
```

## How It Works

1. **Clusters** similar prompts together
2. **Learns** which model performs best per cluster
3. **Routes** new prompts to the optimal model

## Runtime API

- `router.route(prompt, models=None)` routes one prompt.
- `router.route_batch(prompts, models=None)` routes a list of prompts.
- Optional `models` filter restricts candidates to specific model IDs.

## Checkpoint I/O

```python
checkpoint.to_json_file("router.json")
loaded = Router(checkpoint="router.json")
```

## Links

- [Docs](https://docs.nordlyslabs.com)
- [Issues](https://github.com/Nordlys-Labs/nordlys/issues)

## Citation

This project is inspired by the Universal Router approach:

```bibtex
@article{universalrouter2025,
  title={Universal Router: Foundation Model Routing for Arbitrary Tasks},
  author={},
  journal={arXiv preprint arXiv:2502.08773},
  year={2025},
  url={https://arxiv.org/pdf/2502.08773}
}
```

**Paper**: [Universal Router: Foundation Model Routing for Arbitrary Tasks](https://arxiv.org/pdf/2502.08773)
