# Language Bindings

Language bindings for the Nordlys C++ core library.

## Overview

The bindings provide native APIs for different languages, allowing integration with various ecosystems while maintaining the performance of the C++ core.

## Available Bindings

### Python (`python/`)
High-level Python API using nanobind for seamless integration.

**Features:**
- Native Python types (lists, dicts, strings)
- Type stubs for IDE support
- NumPy array compatibility
- Used by the main `nordlys` Python package

**See:** [python/README.md](python/README.md)

### C FFI (`c/`)
C-compatible API for integration with other languages.

**Features:**
- C-compatible types and functions
- Ideal for Rust, Go, Java, and other systems
- Manual memory management
- Thread-safe operations

**See:** [c/README.md](c/README.md)

## Building

Bindings are built automatically when enabled:

```bash
# Build Python bindings (default: ON)
cmake --preset conan-release -DNORDLYS_BUILD_PYTHON=ON

# Build C FFI bindings (default: OFF)
cmake --preset conan-release -DNORDLYS_BUILD_C=ON

# Build both
cmake --preset conan-release -DNORDLYS_BUILD_PYTHON=ON -DNORDLYS_BUILD_C=ON
```

## Testing

```bash
# Test Python bindings
ctest -R python_bindings

# Test C FFI
ctest -R test_c_ffi
```

## See Also

- [Main README](../README.md) - Build and usage guide
- [Core Library](../core/README.md) - C++ core implementation
