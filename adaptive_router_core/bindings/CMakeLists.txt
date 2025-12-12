# =============================================================================
# Adaptive Router Core - Bindings
# =============================================================================

if(ADAPTIVE_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(ADAPTIVE_BUILD_C)
  add_subdirectory(c)
endif()