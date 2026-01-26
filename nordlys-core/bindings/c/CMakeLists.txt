# =============================================================================
# Nordlys Core - C FFI Bindings
# =============================================================================

add_library(nordlys_c SHARED
  src/nordlys_c.cpp
)
add_library(nordlys::c ALIAS nordlys_c)

target_include_directories(nordlys_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nordlys_c PRIVATE Nordlys::Core)
target_compile_definitions(nordlys_c PRIVATE NORDLYS_C_EXPORTS)
nordlys_target_compile_options(nordlys_c)

set_target_properties(nordlys_c PROPERTIES
  OUTPUT_NAME "nordlys_c"
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Tests
if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Installation
install(TARGETS nordlys_c
  EXPORT nordlys_coreTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/nordlys.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)