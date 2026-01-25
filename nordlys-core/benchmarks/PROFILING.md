# Profiling Benchmarks

Profile nordlys-core benchmarks using headless profiling tools that work without GUI connections. This guide covers using `sample` (macOS), `perf` (Linux), and `gprof` (all platforms) for performance analysis.

## Quick Start

### Using the Profiling Script (macOS - Easiest)

The easiest way to profile on macOS is using the provided script:

```bash
cd nordlys-core
bash benchmarks/scripts/profile.sh
```

This script will:
1. Find or build the benchmark executable
2. Run the benchmark in the background
3. Profile it for 10 seconds using `sample`
4. Save results to `benchmarks/profile.txt`
5. Display key metrics and top functions

**Usage:**
```bash
# Default: profile RoutingSingle_Medium for 10 seconds
bash benchmarks/scripts/profile.sh

# Custom benchmark
bash benchmarks/scripts/profile.sh RoutingSingle_Large

# Custom benchmark and duration
bash benchmarks/scripts/profile.sh RoutingSingle_Small 5
```

**Note:** The script must be run from the `nordlys-core` directory (not from inside `benchmarks/`).

### Using sample (macOS - Manual)

`sample` is the built-in profiling tool for macOS:

```bash
# Build benchmarks (if not already built)
conan install . --output-folder=build/Release --build=missing -s build_type=Release
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
cmake --build build/Release/build/Release --target bench_nordlys_core

# Run benchmark in background and profile
./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium \
  --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!

# Profile for 10 seconds
sample $BENCH_PID 10 -f profile.txt

# Wait for benchmark to finish
wait $BENCH_PID

# View results
cat profile.txt
```

### Using perf (Linux only)

`perf` is the recommended profiling tool for Linux:

```bash
# Build benchmarks (if not already built)
conan install . --output-folder=build/Release --build=missing -s build_type=Release
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
cmake --build build/Release/build/Release --target bench_nordlys_core

# Profile a benchmark
perf record ./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium

# View results
perf report

# Or save to file
perf report > profile.txt
```

### Using gprof (Alternative)

`gprof` works on all platforms but requires recompiling with profiling flags:

```bash
# Build with profiling enabled
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DCMAKE_CXX_FLAGS="-pg"

cmake --build --preset conan-release

# Run benchmark (generates gmon.out)
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle_Medium

# Analyze results
gprof ./build/Release/benchmarks/bench_nordlys_core gmon.out
```

## Profiling Tools

### sample (macOS - Recommended)

**Installation:**
- Built into macOS (no installation needed)
- Located at `/usr/bin/sample`

**Advantages:**
- No code changes required
- Low overhead
- Works with any binary
- Built into macOS

**Basic Usage:**

```bash
# Run benchmark in background
./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium \
  --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!

# Profile for 10 seconds
sample $BENCH_PID 10 -f profile.txt

# Wait for benchmark to finish
wait $BENCH_PID

# View results
cat profile.txt
```

**Advanced Usage:**

```bash
# Profile with more detail and demangling
sample $BENCH_PID 30 -mayDemangle -f detailed_profile.txt

# Profile and filter for specific function
sample $BENCH_PID 10 -f profile.txt | grep -A 10 "route"

# Profile and search for specific patterns
sample $BENCH_PID 10 -f profile.txt | grep -E "simsimd|ModelScorer|route_impl"

# Get top functions summary
grep -A 5 "Sort by top of stack" profile.txt
```

**Interpreting sample Output:**

The `sample` output shows:
- **Call graph**: Hierarchical view of function calls with sample counts
- **Total number in stack**: Flat list of functions sorted by sample count
- **Sort by top of stack**: Functions that appear at the top of call stacks

Key sections to look for:
- `simsimd_l2sq_f32_neon` - SIMD distance calculation (expected bottleneck)
- `Nordlys::route_impl` - Main routing function
- `ModelScorer::score_models` - Model scoring logic
- `operator new` / `malloc` - Memory allocations
- `__init_copy_ctor_external` - String copy operations (should be minimal after optimizations)

### perf (Linux only)

**Installation:**
- **Linux**: Usually pre-installed, or `sudo apt-get install linux-perf` / `sudo yum install perf`
- **macOS**: Not available - use `sample` instead (see below)

**Advantages:**
- No code changes required
- Low overhead
- Works with any binary
- Rich analysis features

**Basic Usage:**

```bash
# Record profile data
perf record ./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium

# View flat profile
perf report

# View call graph
perf report --call-graph

# View specific function
perf report --symbol-filter=route

# Save report to file
perf report > profile.txt

# View with more context
perf report --stdio -n
```

**Advanced Usage:**

```bash
# Profile with call stacks
perf record --call-graph dwarf ./build/Release/benchmarks/bench_nordlys_core

# Profile specific events (cache misses, branch mispredictions)
perf record -e cache-misses,branch-misses ./build/Release/benchmarks/bench_nordlys_core

# Profile with sampling frequency
perf record -F 1000 ./build/Release/benchmarks/bench_nordlys_core

# Compare two profiles
perf diff baseline.data current.data
```

### gprof (All Platforms)

**Installation:**
- Usually comes with GCC/Clang toolchain

**Advantages:**
- Works everywhere
- Simple text output
- Call graph generation

**Usage:**

```bash
# Build with -pg flag
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DCMAKE_CXX_FLAGS="-pg -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-pg"

cmake --build --preset conan-release

# Run benchmark
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle_Medium

# Analyze flat profile
gprof ./build/Release/benchmarks/bench_nordlys_core gmon.out

# Generate call graph
gprof -b ./build/Release/benchmarks/bench_nordlys_core gmon.out | gprof2dot | dot -Tpng -o callgraph.png
```

## Interpreting Results

### sample Output (macOS)

The `sample` output shows a call graph with sample counts. Key things to look for:

**Example Call Graph:**
```
2038 Thread_12092304   DispatchQueue_1: com.apple.main-thread
  2038 main
    2038 benchmark::RunSpecifiedBenchmarks()
      1350 BM_RoutingSingle_Medium
        1313 Nordlys<float>::route_impl
          1297 CpuClusterBackend<float>::assign
            1286 simsimd_l2sq_f32_neon  ← Main bottleneck
```

**Flat Summary (at end of file):**
```
Sort by top of stack, same collapsed (when >= 5):
    simsimd_l2sq_f32_neon(...)        1286
    benchmark::CPUInfo::CPUInfo()      684
    CpuClusterBackend<float>::assign   17
    ModelScorer::score_models(...)      11
```

**What to Check:**
1. **SIMD distance calculation** (`simsimd_l2sq_f32_neon`) - Should be the dominant cost (60-90%)
2. **String operations** - Look for `__init_copy_ctor_external` (should be minimal/zero after optimizations)
3. **Memory allocations** - Look for `operator new` or `malloc` (should be <1% of total)
4. **Tracy profiler** - Search for `tracy` (should be zero - profiler removed)
5. **Timing overhead** - Look for `clock_gettime` (should only be in benchmark framework)

### perf Output (Linux)

The `perf report` shows:
- **Overhead**: Percentage of time spent in each function
- **Samples**: Number of profiling samples
- **Symbol**: Function name

**Example:**
```
Overhead  Samples  Symbol
  63.1%    1286   simsimd_l2sq_f32_neon
  33.6%     684   benchmark::CPUInfo::CPUInfo
   0.5%      11   ModelScorer::score_models
```

### gprof Output

The `gprof` output shows:
- **% time**: Percentage of total execution time
- **cumulative seconds**: Cumulative time
- **self seconds**: Time in this function
- **calls**: Number of function calls

**Example:**
```
% time  cumulative seconds  self seconds  calls  name
 63.1           0.123        0.123       1000   simsimd_l2sq_f32_neon
  0.5           0.234        0.111       1000   ModelScorer::score_models
```

## Expected Performance Profile

After optimizations, a typical profile should show:

| Component | Expected % | What It Means |
|-----------|------------|---------------|
| SIMD Distance Calc | 60-90% | Expected bottleneck - algorithmic |
| Benchmark Overhead | 10-30% | Google Benchmark framework |
| Model Scoring | 0.5-1% | Fast, acceptable |
| Memory Alloc | <0.5% | Should be minimal |
| String Operations | <0.1% | Should be near zero (using move semantics) |
| Tracy Profiler | 0% | Should be completely removed |

**Red Flags:**
- String copy operations (`__init_copy_ctor_external`) > 1% - indicates unnecessary copies
- Memory allocations > 2% - may indicate optimization opportunities
- Tracy symbols present - profiler not fully removed
- Timing calls in production code - should only be in benchmark framework

## Common Profiling Workflows

### Finding Hotspots

```bash
# Use perf to find functions taking most time
perf record ./build/Release/benchmarks/bench_nordlys_core
perf report --sort=overhead
```

### Analyzing Cache Performance

```bash
# Profile cache misses
perf record -e cache-misses,cache-references ./build/Release/benchmarks/bench_nordlys_core
perf report
```

### Comparing Performance

```bash
# Profile baseline
perf record -o baseline.data ./build/Release/benchmarks/bench_nordlys_core

# Profile optimized version
perf record -o optimized.data ./build/Release/benchmarks/bench_nordlys_core

# Compare
perf diff baseline.data optimized.data
```

### Profiling Specific Benchmarks

```bash
# Profile only routing benchmarks
perf record ./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=Routing

# Profile batch operations
perf record ./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=Batch

# Profile specific benchmark (macOS with sample)
./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium \
  --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!
sample $BENCH_PID 10 -f profile_routing.txt
wait $BENCH_PID
```

### Analyzing Profile Results

The profiling script automatically displays key metrics. For detailed analysis:

```bash
# View full profile
cat benchmarks/profile.txt

# Count samples for specific functions
grep -c "simsimd_l2sq" benchmarks/profile.txt
grep -c "ModelScorer" benchmarks/profile.txt
grep -c "route_impl" benchmarks/profile.txt

# Check for string copies (should be zero)
grep -c "__init_copy_ctor_external" benchmarks/profile.txt

# Check for Tracy (should be zero)
grep -ci tracy benchmarks/profile.txt

# Get top 10 functions
grep -A 10 "Sort by top of stack" benchmarks/profile.txt | head -15
```

## CI Integration

For CI environments, use profiling tools in headless mode:

### Linux (perf)

```bash
# Record profile data
perf record -o profile.data ./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium

# Generate report as artifact
perf report --stdio > profile.txt

# Upload profile.txt as CI artifact
```

### macOS (sample)

```bash
# Run benchmark and profile
./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium \
  --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!
sample $BENCH_PID 10 -f profile.txt
wait $BENCH_PID

# Upload profile.txt as CI artifact
```

## Troubleshooting

### perf not found on macOS

**Problem:** `perf` command not available

**Solution:** Install via Homebrew:
```bash
brew install perf
```

Note: macOS perf support may be limited compared to Linux.

### gprof shows no data

**Problem:** `gmon.out` is empty or missing

**Solutions:**
1. Ensure `-pg` flag is used for both compilation and linking
2. Check that benchmark ran to completion
3. Verify `gmon.out` exists in current directory

### perf shows [unknown] symbols

**Problem:** Function names appear as memory addresses

**Solution:** Build with debug symbols:
```bash
cmake --preset conan-release -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### sample shows no useful data

**Problem:** Profile shows mostly benchmark framework overhead

**Solutions:**
1. Increase profiling duration: `sample $BENCH_PID 30 -f profile.txt`
2. Use `--benchmark_min_time` to ensure benchmark runs long enough
3. Filter specific benchmarks: `--benchmark_filter=RoutingSingle_Medium`
4. Check that benchmark is actually running (not stuck in initialization)

### Finding the benchmark executable

**Problem:** Can't find `bench_nordlys_core`

**Solution:** The build path depends on your CMake preset:
```bash
# Find the executable
find build -name "bench_nordlys_core" -type f

# Or use the script (run from nordlys-core directory)
bash benchmarks/scripts/profile.sh  # Will find or build automatically
```

## Profiling Workflow Example

Here's a complete example of profiling and analyzing results:

```bash
cd nordlys-core

# 1. Build benchmarks
conan install . --output-folder=build/Release --build=missing -s build_type=Release
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
cmake --build build/Release/build/Release --target bench_nordlys_core

# 2. Profile (macOS)
./build/Release/build/Release/benchmarks/bench_nordlys_core \
  --benchmark_filter=RoutingSingle_Medium \
  --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!
sample $BENCH_PID 10 -f profile.txt
wait $BENCH_PID

# 3. Analyze results
echo "=== Top Functions ==="
grep -A 10 "Sort by top of stack" profile.txt | head -15

echo ""
echo "=== String Operations ==="
grep -c "__init_copy_ctor_external" profile.txt || echo "0 (good - no string copies)"

echo ""
echo "=== Memory Allocations ==="
grep -c "operator new" profile.txt

echo ""
echo "=== Tracy Profiler ==="
grep -ci tracy profile.txt || echo "0 (good - Tracy removed)"

echo ""
echo "=== SIMD Distance Calculation ==="
grep "simsimd_l2sq" profile.txt | head -1
```

## Further Reading

- [perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
- [gprof Manual](https://sourceware.org/binutils/docs/gprof/)
- [Linux Performance Tools](https://www.brendangregg.com/linuxperf.html)
- [macOS sample man page](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/InstrumentsUserGuide/Instrument-Sample.html)
- [benchmarks/README.md](./README.md) - Benchmark suite overview
- [Nordlys Core README](../README.md) - Project overview
