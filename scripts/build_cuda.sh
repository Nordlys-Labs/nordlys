#!/bin/bash
# Build script for CUDA package (Linux with CUDA only)
# Run this on a Linux system with CUDA installed

set -e

echo "Building adaptive-router-core-cu12..."

# Change to the CUDA package directory
cd adaptive_router_core_cu12

# Build the package
uv build

echo "CUDA package built successfully!"
echo "Wheel available in: adaptive_router_core_cu12/dist/"