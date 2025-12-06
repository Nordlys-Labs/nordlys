find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find nanobind - provided by scikit-build-core from pyproject.toml build requirements
# or via pip install nanobind
find_package(nanobind CONFIG REQUIRED)

# Create Python extension module
nanobind_add_module(adaptive_core_ext
    NB_STATIC  # Statically link nanobind
    adaptive_core_py.cpp
)

target_link_libraries(adaptive_core_ext PRIVATE adaptive_core)

# Set C++23 for cross-platform compatibility
target_compile_features(adaptive_core_ext PRIVATE cxx_std_23)

# Install module
install(TARGETS adaptive_core_ext LIBRARY DESTINATION .)
