add_library(nordlys_common INTERFACE)
add_library(Nordlys::Common ALIAS nordlys_common)

target_include_directories(nordlys_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS nordlys_common
  EXPORT NordlysTargets
)

install(DIRECTORY include/nordlys
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
