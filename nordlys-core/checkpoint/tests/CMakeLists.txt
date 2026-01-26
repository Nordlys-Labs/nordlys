find_package(GTest REQUIRED)

add_executable(test_checkpoint
  test_checkpoint.cpp
)

target_link_libraries(test_checkpoint PRIVATE
  Nordlys::Checkpoint
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_checkpoint PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_checkpoint
  PROPERTIES TIMEOUT 30 LABELS "checkpoint"
)
