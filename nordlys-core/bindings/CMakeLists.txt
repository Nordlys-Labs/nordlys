# =============================================================================
# Nordlys Core - Bindings
# =============================================================================

if(NORDLYS_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(NORDLYS_BUILD_C)
  add_subdirectory(c)
endif()