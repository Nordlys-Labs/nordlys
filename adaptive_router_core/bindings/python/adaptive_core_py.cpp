#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "router.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(adaptive_core_ext, m) {
    m.doc() = "Adaptive Router C++ Core - High-performance routing engine";

    // RouteResponse struct
    nb::class_<RouteResponse>(m, "RouteResponse")
        .def_ro("selected_model", &RouteResponse::selected_model,
                "ID of the selected model")
        .def_ro("alternatives", &RouteResponse::alternatives,
                "List of alternative model IDs")
        .def_ro("cluster_id", &RouteResponse::cluster_id,
                "Assigned cluster ID")
        .def_ro("cluster_distance", &RouteResponse::cluster_distance,
                "Distance to cluster centroid");

    // Router class
    nb::class_<Router>(m, "Router")
        // Factory methods
        .def_static("from_file", &Router::from_file,
                    "path"_a,
                    "Load router from JSON profile file")
        .def_static("from_json_string", &Router::from_json_string,
                    "json_str"_a,
                    "Load router from JSON string")
        .def_static("from_binary", &Router::from_binary,
                    "path"_a,
                    "Load router from binary MessagePack profile")

        // Single route method - accepts both numpy arrays and Python lists
        // nanobind automatically converts numpy arrays to std::vector<float>
        .def("route", [](Router& self,
                        const std::vector<float>& embedding,
                        float cost_bias) {
            return self.route(embedding.data(), embedding.size(), cost_bias);
        }, "embedding"_a, "cost_bias"_a = 0.5f,
           "Route using pre-computed embedding vector (numpy array or Python list)")

        // Introspection
        .def("get_supported_models", &Router::get_supported_models,
             "Get list of all supported model IDs")
        .def("get_n_clusters", &Router::get_n_clusters,
             "Get number of clusters")
        .def("get_embedding_dim", &Router::get_embedding_dim,
             "Get expected embedding dimension");

    // Module-level version info
    m.attr("__version__") = "0.1.0";
}
