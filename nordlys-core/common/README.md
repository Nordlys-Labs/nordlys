# Common Module

Shared types and utilities used across all Nordlys Core modules.

## Overview

This is a **header-only** module providing foundational types:

- `Matrix<T>` - 2D matrix with memory management
- `Device` - Compute device abstraction (CPU/CUDA)
- `Result` - Routing result types

## Headers

```cpp
#include <nordlys/common/matrix.hpp>      // Matrix<T> class
#include <nordlys/common/device.hpp>      // Device enum and utilities
#include <nordlys/common/result.hpp>      // RouteResult, ModelScore types
```

## CMake

```cmake
target_link_libraries(your_target PRIVATE Nordlys::Common)
```

## Usage

### Matrix

```cpp
#include <nordlys/common/matrix.hpp>

// Create a 100x128 float matrix
Matrix<float> mat(100, 128);

// Access elements
mat(0, 0) = 1.0f;
float val = mat(0, 0);

// Get dimensions
size_t rows = mat.rows();
size_t cols = mat.cols();

// Get raw pointer
float* data = mat.data();
```

### Device

```cpp
#include <nordlys/common/device.hpp>

// Check CUDA availability
bool has_cuda = is_cuda_available();

// Device selection
Device device = Device::CPU;
if (has_cuda) {
    device = Device::CUDA;
}
```

## Dependencies

None - this is a standalone header-only module.
