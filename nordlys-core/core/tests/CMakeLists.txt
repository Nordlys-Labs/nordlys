# =============================================================================
# Nordlys Core - Core Tests
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_nordlys_core
  test_router.cpp
  test_cluster.cpp
  test_scorer.cpp
  test_profile.cpp
)

target_link_libraries(test_nordlys_core
  PRIVATE
    nordlys::core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_nordlys_core
  PROPERTIES TIMEOUT 30 LABELS "core"
)