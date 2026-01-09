#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACY_DIR="$SCRIPT_DIR/.tracy"

TRACY_VERSION="0.11.1"
CAPTURE_TIME_SEC="${TRACY_CAPTURE_TIME:-10}"
OUTPUT_DIR="${TRACY_OUTPUT_DIR:-$PROJECT_ROOT/benchmarks/tracy_captures}"

usage() {
    cat <<EOF
Usage: $0 [BENCHMARK_NAME] [OPTIONS]

Runs a benchmark with Tracy profiling in headless mode (for CI).

Arguments:
  BENCHMARK_NAME    Name of the benchmark to run (e.g., RoutingSingle_Medium)
                    If not provided, runs all benchmarks sequentially.

Options:
  -t, --time SECONDS    Capture duration in seconds (default: 10)
  -o, --output DIR      Output directory for .tracy files (default: benchmarks/tracy_captures)
  -h, --help            Show this help message

Environment Variables:
  TRACY_CAPTURE_TIME    Override capture time (seconds)
  TRACY_OUTPUT_DIR      Override output directory

Examples:
  $0 RoutingSingle_Medium
  $0 --time 30 --output /tmp/tracy_results
  $0  # Run all benchmarks

EOF
}

log_info() {
    echo "[tracy-ci] $*"
}

log_error() {
    echo "[tracy-ci] ERROR: $*" >&2
}

detect_platform() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)
            log_error "Unsupported platform: $(uname -s)"
            exit 1
            ;;
    esac
}

download_tracy_capture() {
    local platform="$1"
    local tracy_bin="$TRACY_DIR/tracy-capture"
    
    if [[ -f "$tracy_bin" ]]; then
        log_info "Tracy capture tool already exists at $tracy_bin"
        return 0
    fi
    
    log_info "Downloading Tracy capture tool v${TRACY_VERSION} for ${platform}..."
    mkdir -p "$TRACY_DIR"
    
    local download_url
    case "$platform" in
        linux)
            download_url="https://github.com/wolfpld/tracy/releases/download/v${TRACY_VERSION}/Tracy-${TRACY_VERSION}-linux.tar.gz"
            ;;
        macos)
            download_url="https://github.com/wolfpld/tracy/releases/download/v${TRACY_VERSION}/Tracy-${TRACY_VERSION}-macos.tar.gz"
            ;;
    esac
    
    local tmp_archive="$TRACY_DIR/tracy.tar.gz"
    if ! curl -L -o "$tmp_archive" "$download_url"; then
        log_error "Failed to download Tracy from $download_url"
        return 1
    fi
    
    log_info "Extracting Tracy capture tool..."
    tar -xzf "$tmp_archive" -C "$TRACY_DIR" --strip-components=1
    rm "$tmp_archive"
    
    chmod +x "$tracy_bin"
    log_info "Tracy capture tool installed to $tracy_bin"
}

find_benchmark_binary() {
    local bench_name="$1"
    
    local candidates=(
        "$PROJECT_ROOT/build/Release/benchmarks/bench_nordlys_core"
        "$PROJECT_ROOT/build/benchmarks/bench_nordlys_core"
        "$PROJECT_ROOT/build/Release/benchmarks/bench_routing_e2e"
        "$PROJECT_ROOT/build/Release/benchmarks/bench_checkpoint_e2e"
        "$PROJECT_ROOT/build/Release/benchmarks/bench_routing_cuda"
        "$PROJECT_ROOT/build/benchmarks/bench_routing_e2e"
        "$PROJECT_ROOT/build/benchmarks/bench_checkpoint_e2e"
        "$PROJECT_ROOT/build/benchmarks/bench_routing_cuda"
    )
    
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            if [[ -z "$bench_name" ]] || "$candidate" --benchmark_list 2>/dev/null | grep -q "$bench_name"; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    
    return 1
}

run_tracy_capture() {
    local bench_binary="$1"
    local bench_name="$2"
    local output_file="$3"
    
    local tracy_capture="$TRACY_DIR/tracy-capture"
    
    log_info "Starting Tracy capture (${CAPTURE_TIME_SEC}s)..."
    log_info "Output file: $output_file"
    
    "$tracy_capture" -o "$output_file" -f &
    local capture_pid=$!
    
    sleep 1
    
    log_info "Running benchmark: $bench_binary${bench_name:+ --benchmark_filter=$bench_name}"
    if [[ -n "$bench_name" ]]; then
        "$bench_binary" --benchmark_filter="$bench_name" &
    else
        "$bench_binary" &
    fi
    local bench_pid=$!
    
    log_info "Capturing for ${CAPTURE_TIME_SEC} seconds..."
    sleep "$CAPTURE_TIME_SEC"
    
    if kill -0 "$bench_pid" 2>/dev/null; then
        log_info "Stopping benchmark..."
        kill "$bench_pid" 2>/dev/null || true
        wait "$bench_pid" 2>/dev/null || true
    fi
    
    if kill -0 "$capture_pid" 2>/dev/null; then
        log_info "Stopping Tracy capture..."
        kill -INT "$capture_pid" 2>/dev/null || true
        wait "$capture_pid" 2>/dev/null || true
    fi
    
    if [[ -f "$output_file" ]]; then
        local size=$(du -h "$output_file" | cut -f1)
        log_info "Capture complete: $output_file ($size)"
        return 0
    else
        log_error "Capture file not created: $output_file"
        return 1
    fi
}

main() {
    local bench_name=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -t|--time)
                CAPTURE_TIME_SEC="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                bench_name="$1"
                shift
                ;;
        esac
    done
    
    local platform
    platform=$(detect_platform)
    
    download_tracy_capture "$platform"
    
    mkdir -p "$OUTPUT_DIR"
    
    local bench_binary
    if ! bench_binary=$(find_benchmark_binary "$bench_name"); then
        log_error "Could not find benchmark binary"
        log_error "Build benchmarks with: cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON -DNORDLYS_ENABLE_TRACY=ON"
        exit 1
    fi
    
    log_info "Found benchmark binary: $bench_binary"
    
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local output_file="$OUTPUT_DIR/${bench_name:-all}_${timestamp}.tracy"
    
    if run_tracy_capture "$bench_binary" "$bench_name" "$output_file"; then
        log_info "Success! Tracy capture saved to: $output_file"
        exit 0
    else
        log_error "Tracy capture failed"
        exit 1
    fi
}

main "$@"
