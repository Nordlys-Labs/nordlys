# =============================================================================
# Nordlys Core - Core Tests
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_nordlys_core
  test_nordlys.cpp
  test_cluster.cpp
  test_scorer.cpp
  test_checkpoint.cpp
)

add_subdirectory(integration)

target_link_libraries(test_nordlys_core
  PRIVATE
    nordlys::core
    GTest::gtest
    GTest::gtest_main
)

# Suppress nodiscard warnings in tests globally
target_compile_options(test_nordlys_core PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_nordlys_core
  PROPERTIES TIMEOUT 30 LABELS "core"
)
