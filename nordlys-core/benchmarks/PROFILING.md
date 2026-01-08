# Profiling Guide

This guide covers profiling nordlys-core benchmarks to identify performance bottlenecks and optimization opportunities.

## Table of Contents

- [Quick Start](#quick-start)
- [Available Tools](#available-tools)
- [Building with Profiling Support](#building-with-profiling-support)
- [Tool Usage](#tool-usage)
  - [CPU Performance Counters (perf)](#cpu-performance-counters-perf)
  - [Flamegraphs](#flamegraphs)
  - [Cache/Branch Analysis (callgrind)](#cachebranch-analysis-callgrind)
  - [Heap Memory Analysis (heaptrack)](#heap-memory-analysis-heaptrack)
  - [System Call Tracing (strace)](#system-call-tracing-strace)
- [Interpreting Results](#interpreting-results)
- [Common Optimization Patterns](#common-optimization-patterns)

## Quick Start

```bash
# 1. Check tool availability
./benchmarks/scripts/check_profiling_tools.sh

# 2. Build with profiling symbols
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DNORDLYS_BUILD_PROFILE=ON
cmake --build --preset conan-release

# 3. Run quick CPU profile
./benchmarks/scripts/profile_perf.sh RoutingSingle_Medium

# 4. Run comprehensive profiling suite
./benchmarks/scripts/profile_all.sh RoutingSingle_Medium
```

## Available Tools

| Tool | Purpose | Overhead | Output |
|------|---------|----------|--------|
| **perf** | CPU counters (IPC, cache, branches) | <5% | Text statistics |
| **flamegraph** | Visual CPU hotspot analysis | ~5% | Interactive SVG |
| **callgrind** | Detailed cache/branch simulation | 20-50x | kcachegrind GUI |
| **heaptrack** | Heap allocation tracking | ~10% | GUI + text report |
| **strace** | System call tracing | 10-30x | Text log |

## Building with Profiling Support

Profiling requires debug symbols (`-g`) and frame pointers (`-fno-omit-frame-pointer`):

```bash
# Standard profiling build
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DNORDLYS_BUILD_PROFILE=ON
cmake --build --preset conan-release

# Alternative: Manual flags
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer"
cmake --build --preset conan-release
```

**Note:** Profiling builds use `Release` optimization level (`-O3`) with debug symbols. This preserves realistic performance while enabling accurate profiling.

## Tool Usage

### CPU Performance Counters (perf)

Measures hardware performance counters to identify CPU bottlenecks.

```bash
# Basic usage
./benchmarks/scripts/profile_perf.sh RoutingSingle_Medium

# Custom iterations
./benchmarks/scripts/profile_perf.sh RoutingSingle_Large 20

# Output: benchmarks/profiling/perf_stat.txt
```

**Key Metrics:**
- **Instructions per Cycle (IPC):** Higher is better (ideal: 2-4 for modern CPUs)
- **Cache miss rate:** Lower is better (<5% is good)
- **Branch miss rate:** Lower is better (<2% is good)

**Advanced:** Edit `profile_perf.sh` to uncomment advanced hardware events (L1/LLC caches, TLB misses).

### Flamegraphs

Visual representation of CPU time spent in each function (stack sampling).

```bash
# Generate flamegraph
./benchmarks/scripts/profile_flamegraph.sh RoutingSingle_Medium

# Open in browser
firefox benchmarks/profiling/flamegraph.svg

# Custom sampling frequency (default: 99Hz)
./benchmarks/scripts/profile_flamegraph.sh RoutingSingle_Large 499
```

**Reading Flamegraphs:**
- **Width:** Proportional to CPU time (wider = more time)
- **Height:** Call stack depth (bottom = entry point, top = leaf functions)
- **Color:** Random (for visual separation only)
- **Search:** Click to zoom, search box to filter functions

**Requirements:** FlameGraph scripts must be in PATH. Install:
```bash
git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph
export PATH="$PATH:$HOME/tools/FlameGraph"
```

### Cache/Branch Analysis (callgrind)

Detailed simulation of CPU caches and branch prediction.

```bash
# Run callgrind (slow: ~5-10 minutes)
./benchmarks/scripts/profile_callgrind.sh RoutingSingle_Medium

# View in GUI
kcachegrind benchmarks/profiling/callgrind.out

# View in terminal
callgrind_annotate benchmarks/profiling/callgrind.out | head -100
```

**What to Look For:**
- **High cache miss rates** in hot functions
- **Functions with poor instruction cache usage** (Ir, I1mr, ILmr columns)
- **Data cache misses** (Dr, D1mr, DLmr columns) in tight loops
- **Branch prediction failures** (Bc, Bcm columns)

**Optimization Hints:**
- Data locality: Group frequently-accessed data together
- Loop blocking: Process data in cache-sized chunks
- Branch prediction: Avoid unpredictable branches in hot loops

### Heap Memory Analysis (heaptrack)

Tracks heap allocations to find memory leaks and excessive allocations.

```bash
# Track allocations
./benchmarks/scripts/profile_memory.sh RoutingSingle_Medium

# View in GUI
heaptrack_gui benchmarks/profiling/heaptrack.gz

# Text summary
heaptrack_print benchmarks/profiling/heaptrack.gz
```

**What to Look For:**
- **Temporary allocations** in hot paths (allocation/deallocation pairs)
- **Large allocations** that could be reused
- **Allocation hotspots** (functions that allocate frequently)

**Optimization Hints:**
- Pre-allocate buffers and reuse them
- Use stack allocation for small, short-lived objects
- Consider object pooling for frequently allocated types

### System Call Tracing (strace)

Traces system calls to identify I/O bottlenecks.

```bash
# Trace system calls
./benchmarks/scripts/profile_strace.sh RoutingSingle_Medium

# Output: benchmarks/profiling/strace_summary.txt
```

**What to Look For:**
- **Excessive open/close calls** (consider caching file handles)
- **Small read/write operations** (consider buffering)
- **Repeated stat/access calls** (consider caching file metadata)

## Interpreting Results

### Example: Identifying CPU Bottlenecks

```bash
# 1. Run perf to get high-level metrics
./benchmarks/scripts/profile_perf.sh RoutingSingle_Medium

# Look for:
#   - Low IPC (<1.0) → CPU stalls, check cache misses
#   - High cache miss rate (>10%) → Poor data locality
#   - High branch miss rate (>5%) → Unpredictable branches

# 2. Generate flamegraph for detailed analysis
./benchmarks/scripts/profile_flamegraph.sh RoutingSingle_Medium

# Identify wide functions (hotspots) and drill down

# 3. Run callgrind on specific hotspot
./benchmarks/scripts/profile_callgrind.sh RoutingSingle_Medium
kcachegrind benchmarks/profiling/callgrind.out

# Analyze cache misses in the identified hotspot
```

### Example: Memory Optimization

```bash
# 1. Check for allocation hotspots
./benchmarks/scripts/profile_memory.sh RoutingSingle_Medium
heaptrack_print benchmarks/profiling/heaptrack.gz

# 2. Identify temporary allocations in hot paths
# 3. Refactor to use stack allocation or reuse buffers
```

## Common Optimization Patterns

### 1. Cache Optimization
**Problem:** High cache miss rate in perf  
**Solution:**
- Use `std::vector` instead of `std::list` (better locality)
- Process data in blocks that fit in L2 cache (~256KB)
- Align frequently-accessed data structures to cache line boundaries (64 bytes)

### 2. Branch Prediction
**Problem:** High branch miss rate  
**Solution:**
- Replace conditional branches with branchless code (e.g., `result = mask & value`)
- Use `__builtin_expect()` for likely/unlikely branches
- Sort data to make branches more predictable

### 3. Memory Allocation
**Problem:** Allocation hotspots in heaptrack  
**Solution:**
- Pre-allocate vectors with `.reserve()` to avoid reallocations
- Use custom allocators for frequently allocated types
- Consider `std::pmr::monotonic_buffer_resource` for temporary allocations

### 4. SIMD Vectorization
**Problem:** Low IPC with sequential operations  
**Solution:**
- Use auto-vectorization hints (`#pragma omp simd`)
- Align data to 16/32/64 byte boundaries
- Use intrinsics for critical loops (AVX2/AVX-512)

## Troubleshooting

### perf: Permission Denied

**Problem:** `perf` requires elevated privileges  
**Solutions:**
```bash
# Option 1: Run with sudo
sudo ./benchmarks/scripts/profile_perf.sh RoutingSingle_Medium

# Option 2: Lower kernel security setting (persistent)
sudo sysctl kernel.perf_event_paranoid=1

# Option 3: Temporary (resets on reboot)
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### FlameGraph Scripts Not Found

**Problem:** `flamegraph.pl` not in PATH  
**Solution:**
```bash
git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph
export PATH="$PATH:$HOME/tools/FlameGraph"

# Add to ~/.bashrc for persistence:
echo 'export PATH="$PATH:$HOME/tools/FlameGraph"' >> ~/.bashrc
```

### Callgrind Too Slow

**Problem:** Callgrind takes too long on large benchmarks  
**Solution:**
```bash
# Profile a smaller benchmark
./benchmarks/scripts/profile_callgrind.sh RoutingSingle_Small

# Or reduce benchmark runtime (edit script to use --benchmark_min_time=0.01s)
```

## See Also

- [Benchmarking README](./README.md) - Benchmark suite overview
- [Google Benchmark Docs](https://github.com/google/benchmark) - Benchmark framework
- [perf Examples](https://www.brendangregg.com/perf.html) - Brendan Gregg's perf guide
- [Flamegraph Guide](https://www.brendangregg.com/flamegraphs.html) - Understanding flamegraphs
