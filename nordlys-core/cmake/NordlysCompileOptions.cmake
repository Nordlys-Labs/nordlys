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
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${NORDLYS_WARNINGS_AS_ERRORS}>>:-Werror>
    )

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
  endif()
endfunction()