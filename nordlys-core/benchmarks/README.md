# Benchmarks

Performance benchmarks for nordlys-core clustering and routing using [Google Benchmark](https://github.com/google/benchmark).

## Building

```bash
cd nordlys-core

# Install dependencies
conan install . --build=missing -of=build -s compiler.cppstd=20

# Configure with benchmarks enabled
cmake -B build -DNORDLYS_BUILD_BENCHMARKS=ON -DNORDLYS_ENABLE_CUDA=ON

# Build all benchmarks
cmake --build build --target bench_cluster_cpu bench_cluster_cuda bench_e2e
```

**Executables:**
- `build/benchmarks/bench_cluster_cpu` - CPU clustering benchmarks
- `build/benchmarks/bench_cluster_cuda` - CUDA clustering benchmarks (requires CUDA)
- `build/benchmarks/bench_e2e` - End-to-end routing and checkpoint benchmarks

## Benchmark Suite

### CPU Clustering (`bench_cluster_cpu.cpp`)

Benchmarks for CPU-based clustering backend:

- `BM_ClusterAssign_CPU/<clusters>/<dim>` - Single embedding assignment
- `BM_ClusterLoadCentroids_CPU/<clusters>/<dim>` - Centroid loading performance
- `BM_ClusterBatchAssign_CPU/<batch>/<clusters>/<dim>` - Batch assignment

**Configurations:**
- Clusters: 10, 50, 100, 500, 1000
- Dimensions: 768, 1024, 1536, 3072, 4096

### CUDA Clustering (`bench_cluster_cuda.cu`)

Benchmarks for CUDA clustering backend with **CPU buffer** (includes H2D transfer) and **GPU buffer** (production scenario, pure compute):

**Single Assignment:**
- `BM_ClusterAssign_CPUBuffer/<clusters>/<dim>` - CPU memory input (includes transfer)
- `BM_ClusterAssign_GPUBuffer/<clusters>/<dim>` - GPU memory input **[PROD]**

**Batch Assignment:**
- `BM_ClusterBatchAssign_CPUBuffer/<batch>/<clusters>/<dim>` - Batch CPU memory
- `BM_ClusterBatchAssign_GPUBuffer/<batch>/<clusters>/<dim>` - Batch GPU memory **[PROD]**

**Configurations:**
- Clusters: 10, 50, 100, 500, 1000
- Dimensions: 768, 1024, 1536, 3072, 4096
- Batch sizes: 1, 8, 16, 32, 64, 128, 256, 512, 1024

### End-to-End (`bench_e2e.cpp`)

Full pipeline benchmarks including checkpoint loading and routing:

**Checkpoint:**
- `BM_CheckpointLoad_<size>` - JSON loading (Small/Medium/Large)
- `BM_RouterInit_<size>` - Router initialization from checkpoint

**Routing:**
- `BM_RouteSingle_<size>` - Single embedding routing
- `BM_RouteBatch/<batch>` - Batch routing (1-1024)
- `BM_RouteConcurrent/<threads>` - Multi-threaded routing (2/4/8 threads)
- `BM_ColdStart_Medium` - Full cold start (load + init + first route)

## Running Benchmarks

### Run All

```bash
./build/benchmarks/bench_cluster_cpu
./build/benchmarks/bench_cluster_cuda
./build/benchmarks/bench_e2e
```

### Filter Specific Benchmarks

```bash
# Compare CPU vs GPU buffer overhead
./build/benchmarks/bench_cluster_cuda --benchmark_filter="ClusterAssign.*/100/1536"

# GPU buffer only (production scenario)
./build/benchmarks/bench_cluster_cuda --benchmark_filter="GPUBuffer"

# Batch routing
./build/benchmarks/bench_e2e --benchmark_filter="RouteBatch"
```

### Save Results

```bash
./build/benchmarks/bench_cluster_cuda \
  --benchmark_format=json \
  --benchmark_out=cluster_cuda.json
```

### Other Options

```bash
# More iterations for stability
./build/benchmarks/bench_cluster_cuda --benchmark_repetitions=10

# Set minimum benchmark time
./build/benchmarks/bench_cluster_cuda --benchmark_min_time=1.0s

# Display counters as table
./build/benchmarks/bench_cluster_cuda --benchmark_counters_tabular=true
```

## Profiling with Wafer CLI

Profile CUDA kernels using [Wafer CLI](https://wafer.ai):

### Prerequisites

```bash
uv tool install wafer-cli
wafer login
```

### NSight Compute (Kernel Analysis)

```bash
# Profile GPU buffer benchmark (production scenario)
ncu --set full -o cluster_profile.ncu-rep \
  ./build/benchmarks/bench_cluster_cuda \
  --benchmark_filter="ClusterAssign_GPUBuffer/100/1536" \
  --benchmark_min_warmup_time=0.1

# Analyze with Wafer
wafer nvidia ncu analyze cluster_profile.ncu-rep

# AI-assisted analysis
wafer wevin -t trace-analyze \
  --args trace=./cluster_profile.ncu-rep \
  "Analyze kernel performance and suggest optimizations"
```

### NSight Systems (Timeline)

```bash
nsys profile -o batch_timeline.nsys-rep --force-overwrite=true \
  --trace=cuda,nvtx --cuda-memory-usage=true \
  ./build/benchmarks/bench_cluster_cuda \
  --benchmark_filter="ClusterBatchAssign_GPUBuffer/512/100/1536"

wafer nvidia nsys analyze batch_timeline.nsys-rep
```

## Expected Results

### CPU Clustering
- Single (100c/1536d): ~15-30 us
- Batch 64 (100c/1536d): ~500-1000 us

### CUDA Clustering (GPU Buffer - Production)
- Single (100c/1536d): ~20-40 us
- Batch 64 (100c/1536d): ~100-200 us

### End-to-End Routing
- Single route (Medium): ~0.3-1 us
- Batch 64: ~20-50 us
- Cold start (Medium): ~5-20 ms

*Performance varies with hardware, system load, and checkpoint size.*

## Fixtures

Synthetic checkpoints for reproducible benchmarks:

| File | Clusters | Dim | Size |
|------|----------|-----|------|
| `checkpoint_small.json` | 10 | 128 | ~17KB |
| `checkpoint_medium.json` | 100 | 512 | ~700KB |
| `checkpoint_large.json` | 1000 | 1536 | ~21MB |
| `checkpoint_xl.json` | 2000 | 1536 | ~41MB |
