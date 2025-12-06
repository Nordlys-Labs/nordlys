# =============================================================================
# Adaptive Router Core - Test Suite
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_adaptive_core)

target_sources(test_adaptive_core PRIVATE
  test_router.cpp
  test_cluster.cpp
  test_scorer.cpp
  test_c_ffi.cpp
)

target_link_libraries(test_adaptive_core
  PRIVATE
    adaptive_core
    adaptive_c
    GTest::gtest
    GTest::gtest_main
)

target_compile_features(test_adaptive_core PRIVATE cxx_std_20)

# Relaxed warnings for test code
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_adaptive_core PRIVATE
    -Wall
    -Wextra
    -Wno-unused-result
  )
endif()

# Test discovery and execution
include(GoogleTest)
gtest_discover_tests(test_adaptive_core
  PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)
