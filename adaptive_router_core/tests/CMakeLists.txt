find_package(GTest REQUIRED)

add_executable(test_adaptive_core
    test_router.cpp
    test_cluster.cpp
    test_scorer.cpp
)

target_link_libraries(test_adaptive_core PRIVATE
    adaptive_core
    GTest::gtest
    GTest::gtest_main
)

# C++26 for tests
target_compile_features(test_adaptive_core PRIVATE cxx_std_26)

# Enable parallel test execution
include(GoogleTest)
gtest_discover_tests(test_adaptive_core
    PROPERTIES
        TIMEOUT 30
        PARALLEL_LEVEL 4
)
