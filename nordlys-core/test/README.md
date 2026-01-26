# Integration Tests

Cross-module integration tests that verify end-to-end functionality.

## Test Suites

- `test_checkpoint_integration.cpp` - Checkpoint loading and round-trip tests
- `test_nordlys_integration.cpp` - Full routing pipeline tests
- `test_end_to_end_integration.cpp` - Complete workflow tests

## Running Tests

```bash
# Build with tests enabled
cmake --preset conan-release -DNORDLYS_BUILD_TESTS=ON
cmake --build --preset conan-release

# Run all integration tests
ctest --preset conan-release -L integration

# Run specific integration test
ctest --preset conan-release -R CheckpointIntegration
```

## Test Fixtures

Test fixtures are located in `fixtures/`:

- `valid_checkpoint_f32.json` - Valid checkpoint with float32 centroids
- `valid_checkpoint_f64.json` - Valid checkpoint with float64 centroids
- `invalid_checkpoint.json` - Malformed checkpoint for error handling tests
