find_package(GTest REQUIRED)

add_executable(test_clustering
  test_cluster.cpp
)

target_link_libraries(test_clustering PRIVATE
  Nordlys::Clustering
  GTest::gtest
  GTest::gtest_main
)

target_compile_options(test_clustering PRIVATE -Wno-unused-result)

include(GoogleTest)
gtest_discover_tests(test_clustering
  PROPERTIES TIMEOUT 30 LABELS "clustering"
)

# CUDA memory tests (only if CUDA enabled)
if(NORDLYS_ENABLE_CUDA)
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
endif()
