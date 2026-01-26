find_package(GTest REQUIRED)

add_executable(test_routing
  test_nordlys.cpp
)

target_link_libraries(test_routing PRIVATE
  Nordlys::Routing
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_routing PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_routing
  PROPERTIES TIMEOUT 30 LABELS "routing"
)
