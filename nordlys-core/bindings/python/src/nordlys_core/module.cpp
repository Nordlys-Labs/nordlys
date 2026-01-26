#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <nordlys/checkpoint/checkpoint.hpp>

#include "bindings.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
  m.doc() = "Nordlys Core - High-performance routing engine";

  // Register all bindings
  register_types(m);
  register_results(m);
  register_checkpoint(m);
  register_nordlys(m);

  // Convenience factory function
  m.def(
      "load_checkpoint",
      [](const std::string& path) {
        // Detect format based on extension
        if (path.ends_with(".msgpack") || path.ends_with(".bin")) {
          return NordlysCheckpoint::from_msgpack(path);
        } else {
          return NordlysCheckpoint::from_json(path);
        }
      },
      "path"_a,
      "Load checkpoint from file (auto-detects format)\n\n"
      "Args:\n"
      "    path: Path to checkpoint file (.json or .msgpack)\n\n"
      "Returns:\n"
      "    NordlysCheckpoint instance");

  m.attr("__version__") = NORDLYS_VERSION;
}
