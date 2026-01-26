find_package(GTest REQUIRED)

add_executable(test_scoring
  test_scorer.cpp
)

target_link_libraries(test_scoring PRIVATE
  Nordlys::Scoring
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_scoring PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_scoring
  PROPERTIES TIMEOUT 30 LABELS "scoring"
)
