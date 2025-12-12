# AuroraAI Router Benchmarks

Comprehensive benchmarking suite comparing **Cactus Profile Router** (ML-based) vs **Claude Sonnet 4 Oracle** (API-based) for on-device AI model routing...

## Overview

This benchmark evaluates whether a local ML router (trained on Cactus embeddings) can match the routing quality of Claude Sonnet 4 acting as an oracle. It combines real-world mobile prompts with the Mobile-MMLU benchmark to test routing accuracy across diverse workloads.

### What We're Comparing

| Router | Method | Cost | Latency | Accuracy Goal |
|--------|--------|------|---------|---------------|
| **Cactus Profile Router** | KMeans clustering on Cactus LFM2-350M embeddings | Free (local) | <5ms | Match Claude oracle |
| **Claude Sonnet 4 Oracle** | API calls to Claude for routing decisions | $3/MTok | ~500ms | Baseline (100%) |

### Key Metrics

- **Routing Agreement**: % of prompts where both routers select the same model
- **Model Size Distribution**: Average model size selected (proxy for quality vs speed)
- **Routing Latency**: Time to compute routing decision
- **Cost**: Profile router is free, Claude oracle costs API tokens

---

## Architecture

```
benchmarks/
├── src/benchmarks/
│   ├── bindings/              # Cactus C library Python bindings
│   │   ├── cactus_bindings.py # ctypes wrapper for libcactus
│   │   └── __init__.py
│   ├── core/
│   │   ├── routers.py         # Router wrappers (uses adaptive_router + Claude)
│   │   ├── runner.py          # Benchmark orchestration
│   │   ├── metrics.py         # Agreement & performance metrics
│   │   └── simulator.py       # On-device latency simulation
│   ├── datasets/
│   │   ├── loader.py          # Dataset loading (custom + Mobile-MMLU)
│   │   └── custom_ondevice.json  # 48 mobile prompts (10 categories)
│   ├── visualizations/
│   │   └── charts.py          # Plotly/Matplotlib charts
│   └── cli.py                 # Command-line interface
├── pyproject.toml             # uv package definition
└── README.md                  # This file
```

---

## Requirements

### **Mac (ARM) - REQUIRED for Accurate Results**

The Cactus Profile Router uses **Cactus LFM2-350M embeddings** which require ARM architecture (Apple Silicon).

**Prerequisites:**

1. **Mac with Apple Silicon** (M1/M2/M3/M4)
2. **Cactus built** from source
3. **Profile trained** on Cactus embeddings (you already did this)

### Why Mac is Required

The profile was trained using Cactus LFM2-350M embeddings (1024d). To get accurate routing, the benchmark MUST use the **same embedding model**. The Cactus C library only runs on ARM (Mac/Android), not x86 Windows/Linux.

**Fallback mode:** On x86, the router falls back to BAAI/bge-large-en-v1.5, but this produces **0% agreement** because the embedding spaces are different.

---

## Installation (Mac)

### 1. Build Cactus Library

First, ensure Cactus is built on your Mac:

```bash
cd /path/to/auroraai/cactus
./apple/build.sh
```

This creates `libcactus.dylib` at:
```
cactus/apple/cactus-macos.xcframework/macos-arm64/cactus.framework/Versions/A/cactus
```

### 2. Download Cactus LFM2-350M Weights

Download the embedding model used for training:

```bash
cd /path/to/auroraai/cactus
mkdir -p weights
cd weights

# Download LFM2-350M from Hugging Face or Cactus releases
# Expected structure:
# weights/lfm2-350m/
#   ├── config.txt
#   └── weights.gguf (or similar)
```

### 3. Install Benchmark Package

```bash
cd /path/to/auroraai/adaptive_router/benchmarks

# Install with uv
uv sync

# Or with pip
pip install -e .
```

### 4. Set Anthropic API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Usage

### Quick Start (Mac)

```bash
cd /path/to/auroraai/adaptive_router/benchmarks

# Run full benchmark
benchmark-cactus \
  --profile ../auroraai-router/cactus-final/profiles/production_profile.json \
  --mmlu-size 250 \
  --cost-bias 0.5 \
  --output results/benchmark_run_001 \
  --visualize \
  --api-key $ANTHROPIC_API_KEY
```

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--profile` | Path to production_profile.json (trained on Cactus embeddings) | **Required** |
| `--mmlu-size` | Number of Mobile-MMLU samples to include | 250 |
| `--cost-bias` | 0.0 = prefer speed, 1.0 = prefer quality | 0.5 |
| `--output` | Output directory for results | **Required** |
| `--visualize` | Generate charts after benchmark | Flag |
| `--api-key` | Anthropic API key (or set ANTHROPIC_API_KEY env) | `$ANTHROPIC_API_KEY` |

### Example Output

```
================================================================================
🏁 Starting Adaptive Router Benchmark
================================================================================

Router Configuration:
  Profile Router: production_profile.json
  Oracle Router:  Claude Sonnet 4 (claude-sonnet-4-20250514)
  Cost Bias:      0.5 (balanced)

🌵 Initializing Cactus LFM2-350M for embeddings
   Model path: /path/to/cactus/weights/lfm2-350m
   Embedding dimension: 1024
✅ Using REAL Cactus embeddings (production mode)

Dataset Configuration:
  Custom Prompts: 48 (10 categories)
  Mobile-MMLU:    250 samples
  Total:          298 prompts

================================================================================
📊 Running Benchmark on 298 Prompts
================================================================================

Progress: 100% |████████████████████████| 298/298 [02:30<00:00, 1.98 prompts/s]

================================================================================
✅ Benchmark Complete!
================================================================================

📊 Results Summary:

Routing Agreement:
  Overall Agreement:     73.5%
  Same Model Selected:   219 / 298 prompts

Profile Router Stats:
  Avg Routing Latency:   3.2ms
  Most Selected Model:   lfm2-1.2b (28%)
  Avg Model Size:        645 MB

Claude Oracle Stats:
  Avg Routing Latency:   523ms
  Most Selected Model:   lfm2-1.2b (31%)
  Avg Model Size:        698 MB

Cost Analysis:
  Profile Router:        $0.00 (local)
  Claude Oracle:         $4.23 (API calls)

📁 Results saved to: results/benchmark_run_001/
📈 Charts saved to: results/benchmark_run_001/charts/
```

---

## How It Works

### 1. Cactus Profile Router

**Training (already done):**
1. Loaded 2,250 MMLU samples across 15 topics
2. Generated embeddings using **Cactus LFM2-350M** (1024d)
3. Clustered embeddings with KMeans (4 clusters)
4. Computed error rates per cluster per model
5. Saved profile (~100KB JSON)

**Inference (this benchmark):**
1. Encode prompt using **Cactus LFM2-350M** (same model as training!)
2. Find nearest cluster (cosine distance)
3. Score models: `score = error_rate + λ × cost`
4. Select lowest-scoring model

### 2. Claude Oracle Router

1. Send prompt to Claude Sonnet 4
2. Ask Claude to select best model from 12 Cactus models
3. Parse response for model selection
4. Return selected model

### 3. Comparison

- Run same 298 prompts through both routers
- Compute agreement rate
- Analyze model size distributions
- Compare latency and cost

---

## Datasets

### Custom On-Device Prompts (48 prompts)

10 categories typical of mobile use cases:

| Category | Examples | Count |
|----------|----------|-------|
| `quick_factual` | "What's the capital of France?" | 5 |
| `calculations` | "What's 15% of $89.99?" | 5 |
| `messaging` | "Write a birthday text to my mom" | 5 |
| `travel` | "How do I get from SFO to downtown?" | 5 |
| `first_aid` | "How to treat a minor burn?" | 5 |
| `shopping` | "Which laptop under $1000?" | 5 |
| `technical_help` | "My WiFi won't connect, help" | 5 |
| `education` | "Explain photosynthesis simply" | 5 |
| `vision` | "What's in this image?" | 4 |
| `tool_using` | "Book me a table at 7pm" | 4 |

### Mobile-MMLU (250 samples)

Subset of Mobile-MMLU covering 80 domains (knowledge-heavy prompts).

**Combined Total:** 298 prompts

---

## Visualizations

When run with `--visualize`, generates:

### 1. Agreement Chart
- Bar chart showing % agreement
- Breakdown by prompt category

### 2. Model Selection Distribution
- Heatmap: Profile Router vs Claude Oracle
- Shows which models each router prefers

### 3. Routing Latency
- Histogram: Profile (3-5ms) vs Claude (500ms)
- P50/P95/P99 percentiles

### 4. Cost Analysis
- Total cost: Profile ($0) vs Claude ($3-5)
- Cost per 1000 prompts

All charts saved as `.png` and `.html` (interactive Plotly).

---

## Interpreting Results

### High Agreement (>80%)

✅ **Profile router works!** Your ML-trained router matches Claude's routing decisions, proving local routing is viable.

### Medium Agreement (60-80%)

⚠️ **Partial success**. Profile router is reasonable but makes different trade-offs. Review disagreements to see if Claude prefers larger models (quality bias).

### Low Agreement (<60%)

❌ **Issue detected**. Possible causes:
- Wrong embedding model (MUST use Cactus LFM2-350M)
- Profile trained on different data distribution
- Cost bias mismatch

### Zero Agreement (0%)

🚨 **Not using Cactus embeddings!** You're on x86 or Cactus isn't built. The router fell back to BAAI/bge which has a completely different embedding space.

**Fix:** Run on Mac with Cactus built.

---

## Design Decisions

### Why Cactus Embeddings?

The profile was **trained** using Cactus LFM2-350M embeddings. For the router to work correctly, it MUST use the **exact same embedding model** at inference time. Using a different embedding model (even with the same dimensionality) results in a different embedding space where the cluster centers are meaningless.

### Why Not Use C++ Core?

The original `adaptive-router` has a C++ core, but:
1. Windows path spaces break Conan builds (C:\Users\House Computer\...)
2. Profile router doesn't need C++ (just embeddings + KMeans inference)
3. Python implementation is 10x simpler and portable

### Why Claude Sonnet 4 (Not Opus)?

- Claude Opus 4.5 model ID was incorrect (`claude-opus-4-5-20250514` doesn't exist)
- Claude Sonnet 4 (`claude-sonnet-4-20250514`) is available, fast, and cheaper
- For routing decisions, Sonnet 4 is plenty smart enough

### Why Mobile-MMLU?

Mobile-MMLU is a standard benchmark for on-device AI (16K questions, 80 domains). It tests diverse knowledge domains typical of mobile assistants.

---

## Troubleshooting

### "Platform: x86_64 (x86) - Cactus requires ARM architecture"

**Cause:** You're on Windows/x86 Linux.

**Fix:** Transfer the repo to your Mac and run there. The benchmark will automatically detect ARM and use Cactus.

### "Model file not found: /path/to/cactus/weights/lfm2-350m"

**Cause:** Cactus model weights aren't downloaded.

**Fix:**
```bash
cd /path/to/auroraai/cactus/weights
# Download lfm2-350m model
# Ensure structure: weights/lfm2-350m/config.txt + weights
```

### "Could not find Cactus library 'libcactus.dylib'"

**Cause:** Cactus not built.

**Fix:**
```bash
cd /path/to/auroraai/cactus
./apple/build.sh
```

### "Agreement: 0.0%"

**Cause:** Using BAAI/bge fallback instead of Cactus embeddings.

**Fix:** Run on Mac with Cactus built. Check the startup messages - it should say "Using REAL Cactus embeddings (production mode)".

### "Warning: Failed to generate charts: 'model_size_mb'"

**Cause:** Visualization bug (will be fixed).

**Workaround:** Results JSON still saved correctly. Charts are optional.

---

## Files Generated

After running the benchmark, the output directory contains:

```
results/benchmark_run_001/
├── results.json              # Full results (routing decisions, metrics)
├── summary.json              # Summary stats (agreement, latency, cost)
├── agreement_matrix.csv      # Profile vs Claude model selections
└── charts/                   # Visualizations (if --visualize)
    ├── agreement.png
    ├── agreement.html        # Interactive
    ├── model_distribution.png
    ├── model_distribution.html
    ├── latency_comparison.png
    └── cost_analysis.png
```

---

## Technical Details

### Embedding Model

- **Model:** Cactus LFM2-350M (LiquidAI foundation model)
- **Dimensions:** 1024
- **Normalization:** L2 normalized
- **Context:** 2048 tokens
- **Speed:** ~145 tok/s on M4 Pro

### Clustering

- **Algorithm:** KMeans (from profile training)
- **Clusters:** 4 (determined during training via silhouette score)
- **Metric:** Cosine distance
- **Normalization:** L2

### Routing Formula

```python
score = error_rate + λ × normalized_cost

where:
  error_rate = trained error rate for cluster C and model M
  λ = cost_bias × 2.0  (maps [0, 1] to [0, 2])
  normalized_cost = model_size_mb / 2000.0
```

### 12 Cactus Models

| Model ID | Size | Speed | Capabilities |
|----------|------|-------|--------------|
| gemma-270m | 172 MB | 173 tok/s | text |
| lfm2-350m | 233 MB | 145 tok/s | text, tools, embed |
| smollm-360m | 227 MB | 150 tok/s | text |
| qwen-600m | 394 MB | 129 tok/s | text, tools, embed |
| lfm2-vl-450m | 420 MB | 113 tok/s | text, vision, embed |
| lfm2-700m | 467 MB | 115 tok/s | text, tools, embed |
| gemma-1b | 642 MB | 100 tok/s | text |
| lfm2-1.2b | 722 MB | 95 tok/s | text, tools, embed |
| lfm2-1.2b-tools | 722 MB | 95 tok/s | text, tools, embed |
| qwen-1.7b | 1161 MB | 75 tok/s | text, tools, embed |
| smollm-1.7b | 1161 MB | 72 tok/s | text, embed |
| lfm2-vl-1.6b | 1440 MB | 60 tok/s | text, vision, embed |

---

## Future Improvements

- [ ] Fix visualization 'model_size_mb' error
- [ ] Add per-category agreement breakdown
- [ ] Include actual model inference (not just routing)
- [ ] Support Android ARM (Linux ARM builds)
- [ ] Add confidence scores for routing decisions
- [ ] Export results to TensorBoard

---

## License

MIT License - See [LICENSE](../LICENSE)

---

## Credits

- **Cactus SDK**: https://github.com/cactus-compute/cactus
- **Mobile-MMLU**: https://huggingface.co/datasets/mobile-mmlu
- **Anthropic Claude**: https://anthropic.com
- **AuroraAI Team**: Botir Khaltaev & contributors

---

## Questions?

**Platform Issues:** "Why Mac only?"
- Because Cactus C library requires ARM. The profile was trained on Cactus embeddings, so inference must use the same model.

**Cost Concerns:** "Claude API is expensive for benchmarking?"
- Yes, ~$4 per 300 prompts. But this is a one-time benchmark to validate the profile router works.

**Agreement Target:** "What's good agreement?"
- 70%+ is excellent (shows ML router matches Claude)
- 50-70% is okay (different trade-offs)
- <50% suggests an issue

**Next Steps:** "Profile router works, now what?"
- Integrate into your app
- Route prompts locally (free, <5ms)
- Fall back to cloud APIs only when needed
