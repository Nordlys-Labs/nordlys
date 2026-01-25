# Benchmarks

Performance benchmarks for nordlys-core routing algorithms.

## Overview

Measures routing performance across different profile sizes and usage patterns using [Google Benchmark](https://github.com/google/benchmark). Benchmarks run automatically in CI.

## Building

To build benchmarks, enable the `NORDLYS_BUILD_BENCHMARKS` option:

```bash
cd nordlys-core

# Install dependencies
conan install . --build=missing -of=build -s compiler.cppstd=20

# Configure with benchmarks enabled
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON

# Build
cmake --build --preset conan-release
```

This creates the benchmark executables:
- `build/Release/benchmarks/bench_nordlys_core` - CPU benchmarks
- `build/Release/benchmarks/bench_nordlys_cuda` - GPU benchmarks (if CUDA enabled)

## Running Benchmarks

### Run All Benchmarks

```bash
cd nordlys-core
./build/Release/benchmarks/bench_nordlys_core
```

### Filter Specific Benchmarks

```bash
# Run only single routing benchmarks
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle

# Run only medium profile benchmarks
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=Medium

# Run benchmarks matching a pattern
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter="Routing.*Small"
```

### Save Results to JSON

```bash
./build/Release/benchmarks/bench_nordlys_core \
  --benchmark_format=json \
  --benchmark_out=results.json
```

### Other Useful Options

```bash
# Run benchmarks with more iterations for stability
./build/Release/benchmarks/bench_nordlys_core --benchmark_repetitions=10

# Set minimum benchmark time
./build/Release/benchmarks/bench_nordlys_core --benchmark_min_time=1.0

# Display counters as rates
./build/Release/benchmarks/bench_nordlys_core --benchmark_counters_tabular=true
```

See `--help` for all available options.

## Benchmark Suite

### Routing Performance (`bench_routing_e2e.cpp`)
- `BM_RoutingSingle_*` - Single embedding routing (Small/Medium/Large/XL)
- `BM_RoutingBatch` - Batch routing (10/100/1000 embeddings)
- `BM_RoutingCostBias` - Performance at different cost bias values
- `BM_RoutingColdStart_*` - Router initialization + first route
- `BM_RoutingConcurrent` - Multi-threaded routing (2/4/8 threads)

### Checkpoint Operations (`bench_checkpoint_e2e.cpp`)
- `BM_CheckpointLoadJSON_*` - JSON file loading and parsing
- `BM_RouterInitialization_*` - Router creation from checkpoint
- `BM_CheckpointValidation_*` - Validation overhead

### GPU Benchmarks (`bench_routing_cuda.cpp`)
- `BM_RoutingGPU_Single_*` - GPU single embedding routing
- `BM_RoutingGPU_Batch` - GPU batch routing
- `BM_GPUTransferOverhead_*` - Host ↔ Device transfer overhead

**Note:** CUDA benchmarks require `NORDLYS_ENABLE_CUDA=ON` and are not run in CI.

## Fixtures

Synthetic routing profiles used for reproducible benchmarks:

| Profile | Clusters | Models | Embedding Dim | Size | Use Case |
|---------|----------|--------|---------------|------|----------|
| `profile_small.json` | 10 | 3 | 128 | ~9KB | Quick iteration, unit tests |
| `profile_medium.json` | 100 | 10 | 512 | ~790KB | Representative workload |
| `profile_large.json` | 1000 | 10 | 1536 | ~23MB | Stress testing |
| `profile_xl.json` | 2000 | 10 | 1536 | ~46MB | Extreme scale |

All fixtures use the same schema as production routing profiles with realistic:
- Model providers (OpenAI, Anthropic, Meta, Google)
- Cost structures
- Error rates per cluster

## Interpreting Results

**Example Output:**
```
BM_RoutingSingle_Small    42.3 us    42.2 us    16574
BM_RoutingSingle_Medium    156 us     156 us     4489
BM_RoutingBatch/10        1.58 ms   1.58 ms      443
```

**Columns:** Time (wall clock), CPU time, Iterations

**Performance Tips:**
- Routing latency scales with cluster count O(n_clusters)
- Batch processing amortizes initialization overhead
- Cost bias has minimal performance impact
- Router is thread-safe for concurrent operations
- Cold start dominated by checkpoint loading

## CI Integration

Benchmarks run automatically on every commit:
- All CPU benchmarks run on Ubuntu and macOS
- Results compared against previous runs
- PR comments when performance regresses >10%
- Results stored for 90 days (non-blocking)

## Performance Baselines

Expected performance:
- **Single routing**: ~50-500μs (depends on profile size)
- **Batch routing**: ~0.1-1ms per 100 embeddings
- **Cold start**: ~1-50ms (depends on profile size)
- **Checkpoint loading**: ~1-200ms (depends on profile size)

Performance varies with hardware, system load, and profile characteristics.

## Profiling

Profile benchmarks using headless tools (no GUI required).

**Quick Start (macOS):**
```bash
bash benchmarks/scripts/profile.sh
```

**Quick Start (Linux):**
```bash
perf record ./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle_Medium
perf report
```

**Tools:**
- `sample` (macOS) - Built-in, use provided script
- `perf` (Linux) - Recommended
- `gprof` (All) - Requires `-pg` flag

**Expected Results:**
- 60-90% in SIMD distance calculation (expected bottleneck)
- <1% in model scoring
- <0.5% in memory allocations
- 0% string copy operations

See [PROFILING.md](./PROFILING.md) for detailed guide.

## Contributing

When adding benchmarks:
1. Follow naming: `BM_<Component>_<Operation>_<Variant>`
2. Use appropriate units (`kMicrosecond`/`kMillisecond`)
3. Document purpose with comments
4. Use fixtures from `fixtures/`
5. Test locally before PR

## See Also

- [Profiling Guide](./PROFILING.md) - Detailed profiling docs
- [Google Benchmark Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Main README](../README.md)
