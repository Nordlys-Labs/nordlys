# C FFI Bindings

C-compatible API for integrating Nordlys with other languages.

## Overview

The C FFI provides a C-compatible interface to the Nordlys C++ core, making it easy to integrate with:
- **Rust** - via FFI bindings
- **Go** - via cgo
- **Java** - via JNI
- **Node.js** - via native addons
- **Other languages** - via C interop

## API Structure

### Headers

```c
#include "nordlys.h"
```

### Types

```c
// Router handle (opaque pointer)
typedef struct NordlysRouter NordlysRouter;

// Device selection
typedef enum {
  NORDLYS_DEVICE_CPU,
  NORDLYS_DEVICE_CUDA
} NordlysDevice;

// Result types
typedef struct {
  char* selected_model;
  char** alternatives;
  size_t alternatives_count;
  int cluster_id;
  float cluster_distance;
} NordlysRouteResult32;

typedef struct {
  char* selected_model;
  char** alternatives;
  size_t alternatives_count;
  int cluster_id;
  double cluster_distance;
} NordlysRouteResult64;
```

## Usage

### Creating a Router

```c
#include "nordlys.h"

// Load checkpoint and create router
NordlysErrorCode error;
NordlysRouter* router = nordlys_router_create_from_file(
    "checkpoint.json",
    NORDLYS_DEVICE_CPU,
    &error
);

if (error != NORDLYS_ERROR_NONE) {
    // Handle error
    nordlys_string_free(nordlys_error_message(error));
    return;
}
```

### Routing

```c
// Route with float32 embedding
float embedding[] = {0.1f, 0.2f, 0.3f, ...};
size_t embedding_size = 512;

NordlysRouteResult32* result = nordlys_router_route_f32(
    router,
    embedding,
    embedding_size,
    0.5f,  // cost_bias
    &error
);

if (error == NORDLYS_ERROR_NONE) {
    printf("Selected model: %s\n", result->selected_model);
    printf("Cluster ID: %d\n", result->cluster_id);
    
    // Access alternatives
    for (size_t i = 0; i < result->alternatives_count; i++) {
        printf("Alternative: %s\n", result->alternatives[i]);
    }
}

// Cleanup
nordlys_route_result_free_f32(result);
nordlys_router_destroy(router);
```

### Batch Routing

```c
float embeddings[100][512];  // 100 embeddings of size 512
size_t count = 100;
size_t dim = 512;

NordlysRouteResult32** results = nordlys_router_route_batch_f32(
    router,
    (float*)embeddings,
    count,
    dim,
    0.5f,
    &error
);

if (error == NORDLYS_ERROR_NONE) {
    for (size_t i = 0; i < count; i++) {
        printf("Result %zu: %s\n", i, results[i]->selected_model);
    }
}

// Cleanup
nordlys_route_batch_free_f32(results, count);
```

## Memory Management

All strings returned by the API must be freed:

```c
// Free a single string
nordlys_string_free(char* str);

// Free route result (frees all strings)
nordlys_route_result_free_f32(NordlysRouteResult32* result);
nordlys_route_result_free_f64(NordlysRouteResult64* result);

// Free batch results
nordlys_route_batch_free_f32(NordlysRouteResult32** results, size_t count);
nordlys_route_batch_free_f64(NordlysRouteResult64** results, size_t count);
```

## Error Handling

```c
typedef enum {
  NORDLYS_ERROR_NONE,
  NORDLYS_ERROR_INVALID_CHECKPOINT,
  NORDLYS_ERROR_DIMENSION_MISMATCH,
  NORDLYS_ERROR_INVALID_CLUSTER,
  NORDLYS_ERROR_MEMORY_ALLOCATION
} NordlysErrorCode;

// Get error message
const char* message = nordlys_error_message(error);
// ... use message ...
nordlys_string_free(message);
```

## Thread Safety

The router is **thread-safe** and can be used concurrently:

```c
// Safe to use from multiple threads
#pragma omp parallel for
for (int i = 0; i < 100; i++) {
    NordlysRouteResult32* result = nordlys_router_route_f32(
        router, embeddings[i], dim, 0.5f, &error
    );
    // ... use result ...
    nordlys_route_result_free_f32(result);
}
```

## Building

```bash
# Build C FFI library
cmake --preset conan-release -DNORDLYS_BUILD_C=ON
cmake --build --preset conan-release --target nordlys_c

# Output: libnordlys_c.so (Linux) or libnordlys_c.dylib (macOS)
```

## Testing

```bash
# Run C FFI tests
ctest -R test_c_ffi --output-on-failure
```

## Integration Examples

### Rust

```rust
use std::ffi::{CString, CStr};
use std::os::raw::c_char;

#[link(name = "nordlys_c")]
extern "C" {
    fn nordlys_router_create_from_file(
        path: *const c_char,
        device: u32,
        error: *mut u32
    ) -> *mut std::ffi::c_void;
    // ... other functions
}
```

### Go

```go
/*
#cgo LDFLAGS: -lnordlys_c
#include "nordlys.h"
*/
import "C"

func Route(router *C.NordlysRouter, embedding []float32) {
    result := C.nordlys_router_route_f32(
        router,
        (*C.float)(&embedding[0]),
        C.size_t(len(embedding)),
        0.5,
        nil,
    )
    // ... use result
}
```

## See Also

- [Main README](../../README.md) - Build and usage guide
- [Core Library](../../core/README.md) - C++ implementation
- [Python Bindings](../python/README.md) - Python API
