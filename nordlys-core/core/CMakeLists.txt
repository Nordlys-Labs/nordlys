# =============================================================================
# Nordlys Core - C++ Library
# =============================================================================

add_library(nordlys_core STATIC
  src/cluster_backend.cpp
  src/scorer.cpp
  src/profile.cpp
)
add_library(nordlys::core ALIAS nordlys_core)

target_include_directories(nordlys_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nordlys_core
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    msgpack-cxx
)

nordlys_target_compile_options(nordlys_core)

# CUDA acceleration (optional)
if(NORDLYS_ENABLE_CUDA)
  add_subdirectory(src/cuda)
  target_link_libraries(nordlys_core PUBLIC nordlys::cuda)
  target_compile_definitions(nordlys_core PUBLIC NORDLYS_HAS_CUDA=1)
endif()

# Tests
if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Installation
install(TARGETS nordlys_core
  EXPORT nordlys_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/nordlys_core
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)