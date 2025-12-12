# =============================================================================
# Adaptive Router Core - Python Bindings
# =============================================================================

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(adaptive_core_ext NB_STATIC
  src/adaptive_core_py.cpp
)

target_link_libraries(adaptive_core_ext PRIVATE adaptive::core)
target_compile_definitions(adaptive_core_ext PRIVATE
  ADAPTIVE_VERSION="${PROJECT_VERSION}"
)

# Install for wheel
install(TARGETS adaptive_core_ext LIBRARY DESTINATION .)

# Python tests via CTest
if(ADAPTIVE_BUILD_TESTS)
  find_package(Python REQUIRED COMPONENTS Interpreter)

   add_test(
     NAME python_bindings
     COMMAND uv run --no-project pytest
       ${CMAKE_CURRENT_SOURCE_DIR}/tests -v --tb=short
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
   )

  # Set PYTHONPATH so tests can find the built module
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:adaptive_core_ext>"
    LABELS "python"
  )
endif()