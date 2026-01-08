# Nordlys Core Benchmarks

End-to-end performance benchmarks for the nordlys-core C++ library.

## Overview

This benchmark suite measures the real-world performance of nordlys-core's routing algorithms across different profile sizes and usage patterns. The benchmarks use [Google Benchmark](https://github.com/google/benchmark) framework and run automatically in CI.

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

### `bench_routing_e2e.cpp` - End-to-End Routing Performance

Measures the complete routing pipeline from embedding input to model selection.

**Benchmarks:**
- `BM_RoutingSingle_Small/Medium/Large/XL` - Single embedding routing across profile sizes
- `BM_RoutingBatch` - Batch routing with 10/100/1000 embeddings
- `BM_RoutingCostBias` - Routing performance at different cost bias values (λ = 0.0 to 1.0)
- `BM_RoutingColdStart_Small/Medium` - Router initialization + first route
- `BM_RoutingConcurrent` - Multi-threaded concurrent routing (2/4/8 threads)

### `bench_checkpoint_e2e.cpp` - Checkpoint Loading and Initialization

Measures checkpoint I/O and router initialization time.

**Benchmarks:**
- `BM_CheckpointLoadJSON_Small/Medium/Large/XL` - JSON file loading and parsing
- `BM_RouterInitialization_Small/Medium/Large/XL` - Complete router creation from checkpoint
- `BM_CheckpointValidation_Medium` - Checkpoint validation overhead

### `bench_routing_cuda.cpp` - GPU Benchmarks

GPU-accelerated routing performance (only built when `NORDLYS_ENABLE_CUDA=ON`).

**Benchmarks:**
- `BM_RoutingGPU_Single_Small/Medium/Large` - GPU single embedding routing
- `BM_RoutingGPU_Batch` - GPU batch routing (10/100/1000 embeddings)
- `BM_GPUTransferOverhead_Medium` - Host ↔ Device transfer overhead

**Note:** CUDA benchmarks are not run in CI.

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

### Example Output

```
-----------------------------------------------------------------------
Benchmark                              Time             CPU   Iterations
-----------------------------------------------------------------------
BM_RoutingSingle_Small              42.3 us         42.2 us        16574
BM_RoutingSingle_Medium              156 us          156 us         4489
BM_RoutingBatch/10                  1.58 ms         1.58 ms          443
```

**Columns:**
- **Time**: Wall clock time (includes I/O, system calls)
- **CPU**: CPU time spent in the benchmark
- **Iterations**: Number of times the benchmark ran

### Performance Tips

1. **Profile Size**: Routing latency scales roughly O(n_clusters)
2. **Batch Processing**: Amortizes initialization overhead
3. **Cost Bias**: Different λ values have minimal performance impact
4. **Concurrency**: Router is thread-safe; use thread pools for high throughput
5. **Cold Start**: Checkpoint loading dominates first-request latency

## CI Integration

Benchmarks run automatically on every commit to `main` and `dev` branches:

- **Execution**: All CPU benchmarks run on Ubuntu and macOS
- **Comparison**: Results compared against previous runs (cached)
- **Alerts**: PR comments when performance regresses >10%
- **Artifacts**: Benchmark results stored for 90 days
- **No Blocking**: Regressions comment on PR but don't fail CI

### Viewing Results

1. Check PR comments for performance comparison
2. Download artifacts from workflow runs
3. View trends in cached benchmark data

## Performance Baselines

No fixed performance baselines are enforced. Benchmarks serve as:
- **Regression detection** - Catch unexpected slowdowns
- **Optimization validation** - Verify improvements
- **Profiling guidance** - Identify bottlenecks

Expected performance characteristics:
- **Single routing**: ~50-500μs depending on profile size
- **Batch routing**: ~0.1-1ms per 100 embeddings
- **Cold start**: ~1-50ms depending on profile size
- **Checkpoint loading**: ~1-200ms depending on profile size

Actual performance varies based on:
- Hardware (CPU model, cache size, RAM speed)
- System load
- Profile characteristics (dimensionality, cluster count)
- Compiler optimizations

## Contributing

When adding new benchmarks:

1. **Follow naming convention**: `BM_<Component>_<Operation>_<Variant>`
2. **Use appropriate units**: `kMicrosecond` for fast ops, `kMillisecond` for slow ops
3. **Document purpose**: Add comment explaining what the benchmark measures
4. **Use fixtures**: Leverage existing profiles or create new ones in `fixtures/`
5. **Test locally**: Run benchmarks before submitting PR

## See Also

- [Google Benchmark User Guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)
- [Nordlys Core README](../README.md)
- [Contributing Guide](../../CONTRIBUTING.md)
