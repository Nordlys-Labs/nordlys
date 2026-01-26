add_library(nordlys_scoring STATIC
  src/scorer.cpp
)
add_library(Nordlys::Scoring ALIAS nordlys_scoring)

target_include_directories(nordlys_scoring
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nordlys_scoring
  PUBLIC
    Nordlys::Common
)

nordlys_target_compile_options(nordlys_scoring)

if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS nordlys_scoring
  EXPORT NordlysTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/nordlys
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
