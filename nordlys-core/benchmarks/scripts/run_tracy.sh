#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BENCH_DIR}/../build/Release"
TRACY_CACHE="${SCRIPT_DIR}/.tracy"
BENCHMARK_BIN="${BUILD_DIR}/benchmarks/bench_nordlys_core"

BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"
TRACY_VERSION="0.11.1"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}$1${NC}"
}

warn() {
    echo -e "${YELLOW}$1${NC}"
}

# Check if benchmark binary exists
if [[ ! -x "$BENCHMARK_BIN" ]]; then
    error "Benchmark binary not found at $BENCHMARK_BIN
Build with: cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON -DNORDLYS_ENABLE_TRACY=ON
           cmake --build --preset conan-release"
fi

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    TRACY_URL="https://github.com/wolfpld/tracy/releases/download/v${TRACY_VERSION}/tracy-linux-x64.tar.gz"
    TRACY_BINARY="tracy-profiler"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    TRACY_URL="https://github.com/wolfpld/tracy/releases/download/v${TRACY_VERSION}/tracy-macos.tar.gz"
    TRACY_BINARY="Tracy.app/Contents/MacOS/Tracy"
else
    error "Unsupported platform: $OSTYPE. Tracy supports Linux and macOS."
fi

TRACY_GUI="${TRACY_CACHE}/${TRACY_BINARY}"

# Download Tracy GUI if not present
if [[ ! -f "$TRACY_GUI" ]]; then
    info "Tracy GUI not found. Downloading..."
    mkdir -p "$TRACY_CACHE"
    
    TEMP_ARCHIVE="/tmp/tracy-${TRACY_VERSION}.tar.gz"
    
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$TRACY_URL" -O "$TEMP_ARCHIVE"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar "$TRACY_URL" -o "$TEMP_ARCHIVE"
    else
        error "Neither wget nor curl found. Please install one of them."
    fi
    
    info "Extracting Tracy GUI..."
    tar -xzf "$TEMP_ARCHIVE" -C "$TRACY_CACHE"
    rm "$TEMP_ARCHIVE"
    
    # Make binary executable
    chmod +x "$TRACY_GUI"
    
    info "Tracy GUI downloaded to ${TRACY_CACHE}/"
fi

# Verify Tracy GUI exists
if [[ ! -f "$TRACY_GUI" ]]; then
    error "Tracy GUI not found after download. Expected at: $TRACY_GUI"
fi

info "Starting Tracy profiler..."
info "Benchmark: $BENCHMARK_FILTER"
info ""

# Launch benchmark in background
info "Launching benchmark..."
"$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=10s &
BENCH_PID=$!

# Give benchmark time to start
sleep 1

# Check if benchmark is still running
if ! kill -0 $BENCH_PID 2>/dev/null; then
    error "Benchmark failed to start or exited immediately"
fi

info "Benchmark running (PID: $BENCH_PID)"
info "Opening Tracy profiler GUI..."
info ""
info "Instructions:"
info "  1. Tracy GUI will connect automatically to the benchmark"
info "  2. The benchmark will run for ~10 seconds"
info "  3. Explore the profiling data in the GUI"
info "  4. Click 'Statistics' to see zone timing breakdown"
info "  5. Click 'Memory' to see allocations"
info ""
info "Press Ctrl+C to stop both benchmark and Tracy"
info ""

# Launch Tracy GUI
"$TRACY_GUI" &
TRACY_PID=$!

# Cleanup function
cleanup() {
    info ""
    info "Shutting down..."
    
    # Kill benchmark if still running
    if kill -0 $BENCH_PID 2>/dev/null; then
        kill $BENCH_PID 2>/dev/null || true
        wait $BENCH_PID 2>/dev/null || true
    fi
    
    # Kill Tracy GUI if still running
    if kill -0 $TRACY_PID 2>/dev/null; then
        kill $TRACY_PID 2>/dev/null || true
        wait $TRACY_PID 2>/dev/null || true
    fi
    
    info "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Wait for benchmark to complete
wait $BENCH_PID 2>/dev/null || true

info ""
info "Benchmark completed! Tracy GUI is still open for analysis."
info "Close the Tracy window when done, or press Ctrl+C to exit."

# Wait for Tracy GUI
wait $TRACY_PID 2>/dev/null || true

info "Tracy profiling session complete!"
