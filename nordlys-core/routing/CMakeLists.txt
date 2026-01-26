add_library(nordlys_routing INTERFACE)
add_library(Nordlys::Routing ALIAS nordlys_routing)

target_include_directories(nordlys_routing INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nordlys_routing INTERFACE
  Nordlys::Clustering
  Nordlys::Scoring
  Nordlys::Checkpoint
  Nordlys::Common
)

if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS nordlys_routing
  EXPORT NordlysTargets
)

install(DIRECTORY include/nordlys
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
