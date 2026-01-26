add_library(nordlys_checkpoint STATIC
  src/checkpoint.cpp
)
add_library(Nordlys::Checkpoint ALIAS nordlys_checkpoint)

target_include_directories(nordlys_checkpoint
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nordlys_checkpoint
  PUBLIC
    Nordlys::Common
    Nordlys::Scoring
    nlohmann_json::nlohmann_json
    msgpack-cxx
    simdjson::simdjson
)

nordlys_target_compile_options(nordlys_checkpoint)

if(NORDLYS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

install(TARGETS nordlys_checkpoint
  EXPORT NordlysTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/nordlys
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
