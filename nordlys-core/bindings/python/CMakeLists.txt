# =============================================================================
# Nordlys Core - Python Bindings
# =============================================================================

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(nordlys_core_ext NB_STATIC
  src/nordlys_core_py.cpp
)

target_link_libraries(nordlys_core_ext PRIVATE nordlys::core)
target_compile_definitions(nordlys_core_ext PRIVATE
  NORDLYS_VERSION="${PROJECT_VERSION}"
)

# Install for wheel
install(TARGETS nordlys_core_ext LIBRARY DESTINATION .)

# Python tests via CTest
if(NORDLYS_BUILD_TESTS)
  find_package(Python REQUIRED COMPONENTS Interpreter)

   add_test(
     NAME python_bindings
     COMMAND uv run --no-project pytest
       ${CMAKE_CURRENT_SOURCE_DIR}/tests -v --tb=short
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
   )

  # Set PYTHONPATH so tests can find the built module
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nordlys_core_ext>"
    LABELS "python"
  )
endif()