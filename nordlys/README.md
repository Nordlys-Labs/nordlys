# Nordlys

Smart LLM model router. Picks the best model for each prompt based on cost and quality.

## Install

```bash
uv pip install -e .
```

## Quick Start

```python
from nordlys import Nordlys, ModelConfig
import pandas as pd

# 1. Define your models
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
]

# 2. Training data: questions + accuracy scores per model
df = pd.DataFrame({
    "questions": ["Write code", "What is 2+2?", "Explain quantum physics"],
    "openai/gpt-4": [0.95, 0.99, 0.92],
    "openai/gpt-3.5-turbo": [0.70, 0.99, 0.60],
})

# 3. Fit and route
router = Nordlys(models=models)
router.fit(df)

result = router.route("Write a sorting algorithm", cost_bias=0.5)
print(result.model_id)  # Best model for this prompt
```

## How It Works

1. **Clusters** similar prompts together
2. **Learns** which model performs best per cluster
3. **Routes** new prompts to the optimal model

## Cost Bias

```python
# cost_bias=0.0 → Always cheapest
router.route("prompt", cost_bias=0.0)

# cost_bias=1.0 → Always best quality
router.route("prompt", cost_bias=1.0)

# cost_bias=0.5 → Balanced
router.route("prompt", cost_bias=0.5)
```

## Save & Load

```python
router.save("router.json")
loaded = Nordlys.load("router.json")
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
