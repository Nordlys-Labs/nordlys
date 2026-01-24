#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations for registration functions
void register_types(nb::module_& m);
void register_results(nb::module_& m);
void register_checkpoint(nb::module_& m);
void register_nordlys(nb::module_& m);
