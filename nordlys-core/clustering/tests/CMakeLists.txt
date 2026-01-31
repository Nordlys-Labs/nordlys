find_package(GTest REQUIRED)

add_executable(test_cluster_cpu
  test_cluster_cpu.cpp
)

target_link_libraries(test_cluster_cpu PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_cluster_cpu PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_cluster_cpu
  PROPERTIES TIMEOUT 30 LABELS "clustering"
)

if(NORDLYS_ENABLE_CUDA)
  add_executable(test_cluster_cuda
    test_cluster_cuda.cu
  )

  target_link_libraries(test_cluster_cuda PRIVATE
    Nordlys::Clustering
    GTest::gtest
    GTest::gtest_main
  )

  target_compile_options(test_cluster_cuda PRIVATE -Wno-unused-result)

  gtest_discover_tests(test_cluster_cuda
    PROPERTIES TIMEOUT 60 LABELS "cuda"
  )

  add_subdirectory(cuda)
endif()
