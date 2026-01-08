# Profiling with Tracy

Profile nordlys-core benchmarks using Tracy Profiler - a unified real-time profiling tool with GUI that provides CPU, memory, GPU, and lock profiling with <1% overhead.

## Quick Start

```bash
# 1. Build with Tracy enabled
cmake --preset conan-release \
  -DNORDLYS_BUILD_BENCHMARKS=ON \
  -DNORDLYS_ENABLE_TRACY=ON
cmake --build --preset conan-release

# 2. Run Tracy profiler (auto-downloads GUI on first run)
./benchmarks/scripts/run_tracy.sh RoutingSingle_Medium

# Or use CMake target
cmake --build --preset conan-release --target bench_tracy
```

The Tracy GUI will open automatically and connect to the running benchmark. Profile data appears in real-time.

## What is Tracy?

Tracy is a modern profiler with:
- **Real-time visualization** - See profiling data as it happens
- **Interactive GUI** - Click to zoom, filter, analyze
- **CPU profiling** - Flame graphs, statistics, call trees
- **Memory tracking** - Allocation timelines and hotspots
- **GPU support** - CUDA profiling built-in
- **Lock analysis** - Contention detection
- **<1% overhead** - Minimal performance impact

## Tracy Features

### Timeline View
Shows function execution over time:
- **Zones**: Each instrumented function appears as a colored bar
- **Zoom**: Scroll to zoom, click-drag to pan
- **Statistics**: Click any zone to see timing stats
- **Frame marks**: Benchmark iterations appear as frames

### Statistics View
Aggregate timing data:
- **Mean/Min/Max**: See function timing distribution
- **Call count**: How many times each function runs
- **Time percentage**: Which functions dominate CPU
- **Flame graph**: Visual hotspot identification

### Memory View
Allocation tracking:
- **Allocation timeline**: See when memory is allocated
- **Per-zone breakdown**: Which functions allocate
- **Peak memory**: Maximum memory usage
- **Leak detection**: Identify memory leaks

## Interpreting Results

### Example: Analyzing Routing Performance

After running Tracy, you'll see:

```
Timeline:
├─ Frame 1 (1.2ms)                    ← One benchmark iteration
│  ├─ BM_RoutingSingle_Medium (1.2ms)
│  │  ├─ Nordlys::route (1.15ms)
│  │  │  ├─ cluster_assign (800μs)   ← Hotspot!
│  │  │  └─ score_models (300μs)
│  │  └─ route_iteration (1.15ms)
├─ Frame 2 (1.3ms)
...
```

**What to look for:**
1. **Wide zones** = Functions taking the most time (hotspots)
2. **Many zones** = Functions called frequently
3. **Deep nesting** = Complex call chains
4. **Gaps** = Time unaccounted for (overhead, I/O)

### Common Patterns

**Pattern: cluster_assign is slow**
```
Solution: Check distance calculation algorithm
```

**Pattern: Many small allocations**
```
Statistics → Memory → Group by Zone
Solution: Pre-allocate buffers, use object pooling
```

**Pattern: score_models called repeatedly**
```
Timeline shows same calculation multiple times
Solution: Cache scoring results
```

## Adding Instrumentation

Tracy uses macros to mark code regions for profiling:

### Available Macros

```cpp
#include <nordlys_core/tracy.hpp>

void my_function() {
  NORDLYS_ZONE;  // Profile entire function (uses function name)
  
  {
    NORDLYS_ZONE_N("custom_name");  // Profile scope with custom name
    // ... code to profile
  }
  
  NORDLYS_FRAME_MARK;  // Mark frame boundary (for benchmarks)
}
```

### When to Add Zones

✅ **Add zones to:**
- Functions you suspect are slow
- Loops that process large amounts of data
- I/O operations (file read/write)
- Complex algorithms

❌ **Don't add zones to:**
- Trivial functions (<1μs)
- Very frequently called functions (>1M calls/sec) unless investigating hotspot
- Inline functions (compiler warning)

### Example: Instrumenting a New Function

```cpp
// Before (no profiling)
std::vector<float> compute_embeddings(const std::string& text) {
  auto tokens = tokenize(text);
  auto vectors = lookup_vectors(tokens);
  return aggregate(vectors);
}

// After (with Tracy)
std::vector<float> compute_embeddings(const std::string& text) {
  NORDLYS_ZONE;  // Profile entire function
  
  {
    NORDLYS_ZONE_N("tokenize");
    auto tokens = tokenize(text);
  }
  
  {
    NORDLYS_ZONE_N("lookup_vectors");
    auto vectors = lookup_vectors(tokens);
  }
  
  {
    NORDLYS_ZONE_N("aggregate");
    return aggregate(vectors);
  }
}
```

Now Tracy shows which part of `compute_embeddings` is slow.

## CI Integration

Tracy captures are automatically generated in CI for every commit on Linux:

### Viewing CI Captures

1. Go to GitHub Actions run for your commit
2. Download `tracy-capture-<sha>` artifact
3. Open with Tracy GUI locally:
   ```bash
   # Download Tracy GUI if needed
   ./benchmarks/scripts/run_tracy.sh --download-only
   
   # Open capture
   .tracy/tracy-profiler path/to/capture.tracy
   ```

### Comparing Performance

```bash
# Download baseline from main branch
gh run download <main-run-id> -n tracy-capture-main

# Download PR capture
gh run download <pr-run-id> -n tracy-capture-pr

# Open both for side-by-side comparison
./benchmarks/scripts/.tracy/tracy-profiler tracy_capture_main.tracy &
./benchmarks/scripts/.tracy/tracy-profiler tracy_capture_pr.tracy &
```

Compare zone timing in Statistics view to identify regressions.

### Headless Capture (CI)

```bash
# Capture without GUI (for automation)
./benchmarks/scripts/tracy_ci_capture.sh RoutingSingle_Medium output.tracy
```

## Advanced Usage

### Filtering Benchmarks

```bash
# Profile specific benchmark
./benchmarks/scripts/run_tracy.sh RoutingBatch

# Profile cold start
./benchmarks/scripts/run_tracy.sh RoutingColdStart_Medium

# Profile concurrent benchmark
./benchmarks/scripts/run_tracy.sh RoutingConcurrent
```

### Tracy GUI Tips

**Keyboard shortcuts:**
- `F1` - Help / Keyboard shortcuts
- `Ctrl +/-` - Zoom timeline
- `Ctrl 0` - Reset zoom
- `Ctrl F` - Find zone by name

**Mouse controls:**
- `Scroll` - Zoom timeline
- `Middle-click drag` - Pan timeline
- `Left-click` - Select zone, show statistics
- `Right-click` - Context menu

**Useful views:**
- `Statistics` - Aggregate timing data
- `Memory` - Allocation tracking
- `Compare` - Side-by-side comparison
- `Frame` - Per-frame breakdown (benchmarks)

### Performance Tips

Based on Tracy profiling, common optimizations:

1. **Cache optimization**
   - See high L1/L2 cache miss rates in zone details
   - Solution: Improve data locality, reduce pointer chasing

2. **Memory allocation**
   - See many allocations in Memory view
   - Solution: Pre-allocate, use stack allocation, object pooling

3. **Algorithm complexity**
   - See zone time grows non-linearly with input
   - Solution: Use better algorithm (O(n) → O(log n))

4. **Unnecessary work**
   - See same calculation repeated
   - Solution: Cache results, memoization

## Troubleshooting

### Tracy GUI won't connect

**Problem:** Benchmark runs but Tracy shows "Waiting for connection"

**Solutions:**
1. Check firewall allows port 8086 (Tracy default)
2. Ensure benchmark was built with `-DNORDLYS_ENABLE_TRACY=ON`
3. Check benchmark is still running (`ps aux | grep bench_nordlys`)

### No zones appear in Tracy

**Problem:** Timeline is empty

**Solutions:**
1. Verify benchmark has Tracy zones (check source for `NORDLYS_ZONE`)
2. Rebuild with Tracy enabled
3. Check Tracy macros are not `#ifdef`'d out

### Performance overhead is high

**Problem:** Benchmark is slower with Tracy

**Solution:** This is normal! Tracy has <1% overhead, but debug symbols (`-g`) add overhead. Use Release build with Tracy for realistic profiling.

### Tracy GUI download fails

**Problem:** Script can't download Tracy

**Manual download:**
```bash
# Linux
wget https://github.com/wolfpld/tracy/releases/download/v0.11.1/tracy-linux-x64.tar.gz
tar -xzf tracy-linux-x64.tar.gz -C benchmarks/scripts/.tracy/

# macOS
wget https://github.com/wolfpld/tracy/releases/download/v0.11.1/tracy-macos.tar.gz
tar -xzf tracy-macos.tar.gz -C benchmarks/scripts/.tracy/
```

## Platform Support

Tracy works on all platforms:
- **Linux**: Fully supported (CI + local)
- **macOS**: Fully supported (local only)
- **Windows**: Fully supported (local only, not tested)

CI only runs Tracy on Linux for speed, but developers can profile locally on any platform.

## Further Reading

- [Tracy Manual](https://github.com/wolfpld/tracy/releases/download/v0.11.1/tracy.pdf) - Comprehensive Tracy documentation
- [Tracy GitHub](https://github.com/wolfpld/tracy) - Source code and examples
- [benchmarks/README.md](./README.md) - Benchmark suite overview
- [Nordlys Core README](../README.md) - Project overview

## Getting Help

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Review Tracy Manual (linked above)
3. Ask in project Discord/Slack
4. File issue: https://github.com/Nordlys-Labs/nordlys/issues
