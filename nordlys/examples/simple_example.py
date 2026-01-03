"""Simple example demonstrating the new Nordlys API.

This example shows how to:
1. Define model configurations
2. Create training data
3. Fit a Nordlys router
4. Route prompts to optimal models
5. Inspect clusters and metrics
"""

import pandas as pd
import numpy as np

# Import the new API
from nordlys import Nordlys, ModelConfig

# =============================================================================
# 1. Define models with costs
# =============================================================================
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
]

print("Models defined:")
for m in models:
    print(f"  - {m.id}: ${m.cost_average:.2f}/1M tokens")

# =============================================================================
# 2. Create synthetic training data
# =============================================================================
# In real usage, this would come from your evaluation dataset
np.random.seed(42)

# Generate sample questions
questions = [
    # Coding questions (cluster 1) - GPT-4 excels
    "Write a Python function to sort a list",
    "Implement a binary search algorithm",
    "Create a REST API endpoint",
    "Debug this JavaScript code",
    "Write unit tests for this function",
    # General knowledge (cluster 2) - Claude excels
    "Explain quantum computing",
    "What is machine learning?",
    "Describe the water cycle",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    # Simple tasks (cluster 3) - GPT-3.5 is good enough
    "Translate hello to Spanish",
    "What is 2+2?",
    "Capitalize this sentence",
    "Count the words in this text",
    "Format this date",
]

# Simulated accuracy scores (in real usage, from actual evaluations)
# Higher = better performance on that question type
df = pd.DataFrame(
    {
        "questions": questions,
        # GPT-4: great at coding, good at everything
        "openai/gpt-4": [
            0.95,
            0.92,
            0.90,
            0.88,
            0.91,  # coding
            0.85,
            0.87,
            0.84,
            0.86,
            0.83,  # general
            0.95,
            0.98,
            0.97,
            0.96,
            0.94,
        ],  # simple
        # Claude: great at explanations, good at coding
        "anthropic/claude-3-sonnet": [
            0.88,
            0.85,
            0.82,
            0.80,
            0.84,  # coding
            0.95,
            0.93,
            0.94,
            0.92,
            0.91,  # general
            0.90,
            0.95,
            0.93,
            0.92,
            0.91,
        ],  # simple
        # GPT-3.5: decent at simple tasks, struggles with complex
        "openai/gpt-3.5-turbo": [
            0.70,
            0.65,
            0.60,
            0.55,
            0.62,  # coding
            0.72,
            0.75,
            0.78,
            0.74,
            0.71,  # general
            0.92,
            0.98,
            0.95,
            0.94,
            0.93,
        ],  # simple
    }
)

print(f"\nTraining data: {len(df)} samples")
print(df.head())

# =============================================================================
# 3. Create and fit Nordlys router
# =============================================================================
print("\n" + "=" * 60)
print("Fitting Nordlys router...")
print("=" * 60)

# Basic usage with defaults
router = Nordlys(
    models=models,
    nr_clusters=3,  # We expect ~3 clusters based on our data
)

# Fit on the training data
router.fit(df)

print(f"\nFitted! {router}")

# =============================================================================
# 4. Inspect clusters and metrics
# =============================================================================
print("\n" + "=" * 60)
print("Cluster Information")
print("=" * 60)

metrics = router.get_metrics()
print("\nOverall Metrics:")
print(f"  - Silhouette Score: {metrics.silhouette_score:.3f}")
print(f"  - Number of Clusters: {metrics.n_clusters}")
print(f"  - Cluster Sizes: {metrics.cluster_sizes}")

print("\nPer-Cluster Details:")
for cluster in router.get_clusters():
    print(f"\n  Cluster {cluster.cluster_id} ({cluster.size} samples):")
    for model_id, accuracy in sorted(
        cluster.model_accuracies.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    - {model_id}: {accuracy:.2%} accuracy")

# =============================================================================
# 5. Route new prompts
# =============================================================================
print("\n" + "=" * 60)
print("Routing New Prompts")
print("=" * 60)

test_prompts = [
    ("Write a recursive fibonacci function", "coding"),
    ("Explain how neural networks learn", "explanation"),
    ("What is 15 * 23?", "simple math"),
]

for prompt, category in test_prompts:
    print(f'\nPrompt ({category}): "{prompt}"')

    # Route with different cost preferences
    for cost_bias in [0.0, 0.5, 1.0]:
        result = router.route(prompt, cost_bias=cost_bias)
        bias_label = {0.0: "cheapest", 0.5: "balanced", 1.0: "best quality"}[cost_bias]
        print(f"  cost_bias={cost_bias} ({bias_label}): {result.model_id}")

# =============================================================================
# 6. Batch routing
# =============================================================================
print("\n" + "=" * 60)
print("Batch Routing")
print("=" * 60)

batch_prompts = [
    "Implement quicksort",
    "What causes rain?",
    "Convert 5 km to miles",
]

results = router.route_batch(batch_prompts, cost_bias=0.5)
for prompt, result in zip(batch_prompts, results):
    print(f'  "{prompt[:30]}..." -> {result.model_id}')

# =============================================================================
# 7. Save and load
# =============================================================================
print("\n" + "=" * 60)
print("Save and Load")
print("=" * 60)

# Save to JSON
router.save("/tmp/nordlys_example.json")
print("Saved to /tmp/nordlys_example.json")

# Load it back
loaded_router = Nordlys.load("/tmp/nordlys_example.json")
print(f"Loaded: {loaded_router}")

# Verify it works
result = loaded_router.route("Test prompt", cost_bias=0.5)
print(f"Loaded router routes to: {result.model_id}")

print("\n" + "=" * 60)
print("Example Complete!")
print("=" * 60)
