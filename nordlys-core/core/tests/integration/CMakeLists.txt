# =============================================================================
# Nordlys Core - Integration Tests
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_nordlys_integration
  test_nordlys_integration.cpp
  test_checkpoint_integration.cpp
)

target_link_libraries(test_nordlys_integration
  PRIVATE
    nordlys::core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_nordlys_integration
  PROPERTIES TIMEOUT 60 LABELS "integration"
)
