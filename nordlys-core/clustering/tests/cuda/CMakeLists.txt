find_package(GTest REQUIRED)

add_executable(test_cuda_common
  test_cuda_common.cu
)

target_link_libraries(test_cuda_common PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_common PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_common
  PROPERTIES TIMEOUT 30 LABELS "cuda"
)

add_executable(test_cuda_config
  test_cuda_config.cu
)

target_link_libraries(test_cuda_config PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_config PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_config
  PROPERTIES TIMEOUT 30 LABELS "cuda"
)

add_executable(test_cuda_distance
  test_cuda_distance.cu
)

target_link_libraries(test_cuda_distance PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_distance PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_distance
  PROPERTIES TIMEOUT 60 LABELS "cuda"
)

add_executable(test_cuda_memory
  test_cuda_memory.cu
)

target_link_libraries(test_cuda_memory PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_memory PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_memory
  PROPERTIES TIMEOUT 30 LABELS "cuda"
)

add_executable(test_cuda_reduce
  test_cuda_reduce.cu
)

target_link_libraries(test_cuda_reduce PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_reduce PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_reduce
  PROPERTIES TIMEOUT 30 LABELS "cuda"
)

add_executable(test_cuda_utils
  test_cuda_utils.cu
)

target_link_libraries(test_cuda_utils PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cuda_utils PRIVATE -Wno-unused-result)

gtest_discover_tests(test_cuda_utils
  PROPERTIES TIMEOUT 30 LABELS "cuda"
)
