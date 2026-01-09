# =============================================================================
# Nordlys Core - Shared Compile Options
# =============================================================================

function(nordlys_target_compile_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -Wnon-virtual-dtor -Woverloaded-virtual
        -Wold-style-cast -Wcast-qual
        -Wformat=2 -Wimplicit-fallthrough
      >
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${NORDLYS_WARNINGS_AS_ERRORS}>>:-Werror -Wno-error=cpp>
    )

    # Release optimizations
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:
        -O3
        -ffast-math
        -funroll-loops
        -ftree-vectorize
      >
    )

    # LTO for Release builds (significant performance improvement)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

    # Sanitizers for Debug builds (works with multi-config generators like VS/Xcode/Ninja Multi-Config)
    target_compile_options(${target} PRIVATE
      $<$<AND:$<BOOL:${NORDLYS_ENABLE_SANITIZERS}>,$<CONFIG:Debug>>:
        -fsanitize=address,undefined -fno-omit-frame-pointer
      >
    )
    target_link_options(${target} PRIVATE
      $<$<AND:$<BOOL:${NORDLYS_ENABLE_SANITIZERS}>,$<CONFIG:Debug>>:
        -fsanitize=address,undefined
      >
    )

  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE
      /W4 /permissive- /Zc:__cplusplus
      $<$<BOOL:${NORDLYS_WARNINGS_AS_ERRORS}>:/WX>
    )

    # Release optimizations for MSVC
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:
        /O2
        /Ob3
        /fp:fast
        /GL
      >
    )
    target_link_options(${target} PRIVATE
      $<$<CONFIG:Release>:/LTCG>
    )
  endif()
endfunction()