# =============================================================================
# Nordlys Core - C FFI Tests
# =============================================================================

find_package(GTest REQUIRED)

add_executable(test_c_ffi
  test_c_ffi.cpp
)

target_link_libraries(test_c_ffi
  PRIVATE
    nordlys::c
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_c_ffi
  PROPERTIES TIMEOUT 30 LABELS "c_ffi"
)