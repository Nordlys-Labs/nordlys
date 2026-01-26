#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys/routing/nordlys.hpp>

#include "bindings.h"

namespace nb = nanobind;

void register_results(nb::module_& m) {
  // Result type
  nb::class_<RouteResult>(m, "RouteResult", "Routing result")
      .def_ro("selected_model", &RouteResult::selected_model, "Selected model ID")
      .def_ro("alternatives", &RouteResult::alternatives, "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResult::cluster_id, "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResult::cluster_distance,
              "Distance to cluster center")
      .def("__repr__", [](const RouteResult& r) {
        return "<RouteResult model='" + r.selected_model
               + "' cluster=" + std::to_string(r.cluster_id) + ">";
      });
}
