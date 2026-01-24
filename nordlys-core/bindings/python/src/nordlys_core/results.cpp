#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/nordlys.hpp>

#include "bindings.h"

namespace nb = nanobind;

void register_results(nb::module_& m) {
  // Result types
  nb::class_<RouteResult<float>>(m, "RouteResult32", "Routing result for float32 precision")
      .def_ro("selected_model", &RouteResult<float>::selected_model, "Selected model ID")
      .def_ro("alternatives", &RouteResult<float>::alternatives, "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResult<float>::cluster_id, "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResult<float>::cluster_distance,
              "Distance to cluster center")
      .def("__repr__", [](const RouteResult<float>& r) {
        return "<RouteResult32 model='" + r.selected_model
               + "' cluster=" + std::to_string(r.cluster_id) + ">";
      });

  nb::class_<RouteResult<double>>(m, "RouteResult64", "Routing result for float64 precision")
      .def_ro("selected_model", &RouteResult<double>::selected_model, "Selected model ID")
      .def_ro("alternatives", &RouteResult<double>::alternatives, "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResult<double>::cluster_id, "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResult<double>::cluster_distance,
              "Distance to cluster center")
      .def("__repr__", [](const RouteResult<double>& r) {
        return "<RouteResult64 model='" + r.selected_model
               + "' cluster=" + std::to_string(r.cluster_id) + ">";
      });
}
