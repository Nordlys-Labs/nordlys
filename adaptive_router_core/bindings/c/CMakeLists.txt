# =============================================================================
# Adaptive Router Core - C FFI Bindings
# =============================================================================

add_library(adaptive_c SHARED
  src/adaptive_c.cpp
)
add_library(adaptive::c ALIAS adaptive_c)

target_include_directories(adaptive_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(adaptive_c PRIVATE adaptive::core)
target_compile_definitions(adaptive_c PRIVATE ADAPTIVE_C_EXPORTS)
adaptive_target_compile_options(adaptive_c)

set_target_properties(adaptive_c PROPERTIES
  OUTPUT_NAME "adaptive_c"
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Tests
if(ADAPTIVE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Installation
install(TARGETS adaptive_c
  EXPORT adaptive_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/adaptive.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)