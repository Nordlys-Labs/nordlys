find_package(GTest REQUIRED)

add_executable(test_integration
  integration/test_checkpoint_integration.cpp
  integration/test_end_to_end_integration.cpp
  integration/test_nordlys_integration.cpp
)

target_link_libraries(test_integration PRIVATE
  Nordlys::Core
  GTest::gtest
  GTest::gtest_main
)

target_include_directories(test_integration PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(test_integration PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_integration
  PROPERTIES TIMEOUT 60 LABELS "integration"
)
