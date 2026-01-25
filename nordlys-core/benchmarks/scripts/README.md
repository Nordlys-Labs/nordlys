# Benchmark Scripts

Simple utility script for profiling benchmarks.

## Script

### `profile.sh`

Profile a benchmark using macOS `sample` tool and display key metrics.

**Usage:**
```bash
cd nordlys-core
bash benchmarks/scripts/profile.sh [benchmark_filter] [duration]
```

**Examples:**
```bash
# Default: RoutingSingle_Medium, 10 seconds
bash benchmarks/scripts/profile.sh

# Custom benchmark
bash benchmarks/scripts/profile.sh RoutingSingle_Large

# Custom benchmark and duration
bash benchmarks/scripts/profile.sh RoutingSingle_Small 5
```

**Output:**
- Runs the benchmark and profiles it
- Displays top 5 functions
- Shows key metrics:
  - SIMD distance calculation references
  - String copy operations (should be 0)
  - Memory allocations
  - Tracy profiler status
- Saves full profile to `benchmarks/profile.txt`

**Viewing full profile:**
```bash
cat benchmarks/profile.txt
```

## Requirements

- **macOS**: Uses `sample` tool (built-in)
- **Linux**: Use `perf` instead (see [PROFILING.md](../PROFILING.md))
- **bash**: Requires bash 4.0+

## See Also

- [PROFILING.md](../PROFILING.md) - Detailed profiling guide
- [README.md](../README.md) - Benchmark suite overview
