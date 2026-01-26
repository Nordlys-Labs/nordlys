# =============================================================================
# Nordlys Core - Python Bindings
# =============================================================================

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC
  src/nordlys_core/module.cpp
  src/nordlys_core/helpers.cpp
  src/nordlys_core/types.cpp
  src/nordlys_core/results.cpp
  src/nordlys_core/checkpoint.cpp
  src/nordlys_core/nordlys.cpp
)

target_link_libraries(_core PRIVATE Nordlys::Core)
target_compile_definitions(_core PRIVATE NORDLYS_VERSION="${PROJECT_VERSION}")

install(TARGETS _core LIBRARY DESTINATION .)
