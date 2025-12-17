# =============================================================================
# Adaptive Router Core - Core Tests
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_adaptive_core
  test_router.cpp
  test_cluster.cpp
  test_scorer.cpp
  test_profile.cpp
)

target_link_libraries(test_adaptive_core
  PRIVATE
    adaptive::core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_adaptive_core
  PROPERTIES TIMEOUT 30 LABELS "core"
)