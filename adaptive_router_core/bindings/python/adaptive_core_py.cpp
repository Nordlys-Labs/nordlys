#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "router.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(adaptive_core_ext, m) {
  m.doc() = "Adaptive Router C++ Core - High-performance routing engine";

  // RouteResponse struct
  nb::class_<RouteResponse>(m, "RouteResponse")
      .def_ro("selected_model", &RouteResponse::selected_model, "ID of the selected model")
      .def_ro("alternatives", &RouteResponse::alternatives, "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResponse::cluster_id, "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResponse::cluster_distance, "Distance to cluster centroid");

  // Router class
  nb::class_<Router>(m, "Router")
      // Factory methods
      .def_static("from_file", &Router::from_file, "path"_a, "Load router from JSON profile file")
      .def_static("from_json_string", &Router::from_json_string, "json_str"_a,
                  "Load router from JSON string")
      .def_static("from_binary", &Router::from_binary, "path"_a,
                  "Load router from binary MessagePack profile")

      // Route - accepts any floating point numpy array (float32, float64, etc.)
      .def(
          "route",
          [](Router& self, nb::ndarray<nb::numpy, nb::ndim<1>> embedding, float cost_bias) {
            size_t size = embedding.shape(0);

            // Handle different dtypes by calling the appropriate C++ template
            if (embedding.dtype() == nb::dtype<float>()) {
              // Check for contiguity: stride(0) should equal sizeof(float) for contiguous elements
              if (embedding.stride(0) != sizeof(float)) {
                throw std::invalid_argument(
                    "Input array must be contiguous. "
                    "Use numpy.ascontiguousarray() to convert non-contiguous arrays.");
              }
              const float* data = static_cast<const float*>(embedding.data());
              return self.route(data, size, cost_bias);
            } else if (embedding.dtype() == nb::dtype<double>()) {
              // Check for contiguity: stride(0) should equal sizeof(double) for contiguous elements
              if (embedding.stride(0) != sizeof(double)) {
                throw std::invalid_argument(
                    "Input array must be contiguous. "
                    "Use numpy.ascontiguousarray() to convert non-contiguous arrays.");
              }
              const double* data = static_cast<const double*>(embedding.data());
              return self.route(data, size, cost_bias);  // Uses templated method
            } else {
              throw std::invalid_argument("Unsupported dtype: only float32 and float64 are supported");
            }
          },
          "embedding"_a, "cost_bias"_a = 0.5f,
          "Route using pre-computed embedding vector (numpy array)")

      // Batch route - accepts any floating point 2D numpy array
      .def(
          "route_batch",
          [](Router& self, nb::ndarray<nb::numpy, nb::ndim<2>> embeddings, float cost_bias) {
            size_t n_embeddings = embeddings.shape(0);
            size_t embedding_dim = embeddings.shape(1);

            if (embedding_dim != static_cast<size_t>(self.get_embedding_dim())) {
              throw std::invalid_argument(
                  "Embedding dimension mismatch: expected " +
                  std::to_string(self.get_embedding_dim()) +
                  ", got " + std::to_string(embedding_dim));
            }

            std::vector<RouteResponse> results;
            results.reserve(n_embeddings);

            // Handle different dtypes
            if (embeddings.dtype() == nb::dtype<float>()) {
              const float* data = static_cast<const float*>(embeddings.data());

              // Check for C-contiguity: stride(1) should equal sizeof(float) for contiguous row elements
              if (embeddings.stride(1) != sizeof(float)) {
                throw std::invalid_argument(
                    "Input array must be C-contiguous (row-major). "
                    "Use numpy.ascontiguousarray() to convert non-contiguous arrays.");
              }

              size_t row_stride = embeddings.stride(0) / sizeof(float);

              for (size_t i = 0; i < n_embeddings; ++i) {
                results.push_back(
                    self.route(data + i * row_stride, embedding_dim, cost_bias)
                );
              }
            } else if (embeddings.dtype() == nb::dtype<double>()) {
              const double* data = static_cast<const double*>(embeddings.data());

              // Check for C-contiguity: stride(1) should equal sizeof(double) for contiguous row elements
              if (embeddings.stride(1) != sizeof(double)) {
                throw std::invalid_argument(
                    "Input array must be C-contiguous (row-major). "
                    "Use numpy.ascontiguousarray() to convert non-contiguous arrays.");
              }

              size_t row_stride = embeddings.stride(0) / sizeof(double);

              for (size_t i = 0; i < n_embeddings; ++i) {
                results.push_back(
                    self.route(data + i * row_stride, embedding_dim, cost_bias)  // Uses templated method
                );
              }
            } else {
              throw std::invalid_argument("Unsupported dtype: only float32 and float64 are supported");
            }

            return results;
          },
          "embeddings"_a, "cost_bias"_a = 0.5f,
          "Batch route multiple embeddings (N×D numpy array)")

      // Introspection
      .def("get_supported_models", &Router::get_supported_models,
           "Get list of all supported model IDs")
      .def("get_n_clusters", &Router::get_n_clusters, "Get number of clusters")
      .def("get_embedding_dim", &Router::get_embedding_dim, "Get expected embedding dimension");

  // Module-level version info
  m.attr("__version__") = "0.1.0";
}
