find_package(GTest REQUIRED)

add_executable(test_common
  test_matrix.cpp
)

target_link_libraries(test_common PRIVATE
  Nordlys::Common
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_common PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_common
  PROPERTIES TIMEOUT 30 LABELS "common"
)
