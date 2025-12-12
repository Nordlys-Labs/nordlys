# =============================================================================
# Adaptive Router Core - C++ Library
# =============================================================================

add_library(adaptive_core STATIC
  src/router.cpp
  src/cluster_backend.cpp
  src/scorer.cpp
  src/profile.cpp
)
add_library(adaptive::core ALIAS adaptive_core)

target_include_directories(adaptive_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(adaptive_core
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    msgpack-cxx
)

adaptive_target_compile_options(adaptive_core)

# CUDA acceleration (optional)
if(ADAPTIVE_ENABLE_CUDA)
  add_subdirectory(src/cuda)
  target_link_libraries(adaptive_core PUBLIC adaptive::cuda)
  target_compile_definitions(adaptive_core PUBLIC ADAPTIVE_HAS_CUDA=1)
endif()

# Tests
if(ADAPTIVE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Installation
install(TARGETS adaptive_core
  EXPORT adaptive_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/adaptive_core
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)