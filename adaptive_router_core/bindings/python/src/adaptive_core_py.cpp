#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <adaptive_core/router.hpp>

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
      // Factory methods - wrap to convert Result<Router, string> errors to exceptions
      .def_static(
          "from_json_file",
          [](const std::string& path) {
            auto result = Router::from_file(path);
            if (!result) {
              throw std::runtime_error(result.error());
            }
            return std::move(result.value());
          },
          "path"_a, "Load router from JSON profile file")
      .def_static(
          "from_json_string",
          [](const std::string& json_str) {
            auto result = Router::from_json_string(json_str);
            if (!result) {
              throw std::runtime_error(result.error());
            }
            return std::move(result.value());
          },
          "json_str"_a, "Load router from JSON string (in-memory)")
      .def_static(
          "from_msgpack_file",
          [](const std::string& path) {
            auto result = Router::from_binary(path);
            if (!result) {
              throw std::runtime_error(result.error());
            }
            return std::move(result.value());
          },
          "path"_a, "Load router from MessagePack binary profile file")

      // Route - float32 version (most common case)
      .def(
          "route",
          [](Router& self, nb::ndarray<float, nb::ndim<1>, nb::c_contig> embedding, float cost_bias, nb::stl::vector<std::string> models) {
            if (embedding.shape(0) != static_cast<size_t>(self.get_embedding_dim())) {
              throw std::invalid_argument(
                  "Embedding dimension mismatch: expected " +
                  std::to_string(self.get_embedding_dim()) +
                  ", got " + std::to_string(embedding.shape(0)));
            }
            return self.route(embedding.data(), embedding.shape(0), cost_bias, models);
          },
          "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = nb::stl::vector<std::string>{},
          "Route using pre-computed embedding vector (float32 numpy array)")

      // Route - float64 version
      .def(
          "route",
          [](Router& self, nb::ndarray<double, nb::ndim<1>, nb::c_contig> embedding, float cost_bias, nb::stl::vector<std::string> models) {
            if (embedding.shape(0) != static_cast<size_t>(self.get_embedding_dim())) {
              throw std::invalid_argument(
                  "Embedding dimension mismatch: expected " +
                  std::to_string(self.get_embedding_dim()) +
                  ", got " + std::to_string(embedding.shape(0)));
            }
            return self.route(embedding.data(), embedding.shape(0), cost_bias, models);
          },
          "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = nb::stl::vector<std::string>{},
          "Route using pre-computed embedding vector (float64 numpy array)")

      // Batch route - float32 version
      .def(
          "route_batch",
          [](Router& self, nb::ndarray<float, nb::ndim<2>, nb::c_contig> embeddings, float cost_bias, nb::stl::vector<std::string> models) {
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

            const float* data = embeddings.data();
            for (size_t i = 0; i < n_embeddings; ++i) {
              results.push_back(self.route(data + i * embedding_dim, embedding_dim, cost_bias, models));
            }

            return results;
          },
          "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = nb::stl::vector<std::string>{},
          "Batch route multiple embeddings (N×D float32 numpy array)")

      // Batch route - float64 version
      .def(
          "route_batch",
          [](Router& self, nb::ndarray<double, nb::ndim<2>, nb::c_contig> embeddings, float cost_bias, nb::stl::vector<std::string> models) {
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

            const double* data = embeddings.data();
            for (size_t i = 0; i < n_embeddings; ++i) {
              results.push_back(self.route(data + i * embedding_dim, embedding_dim, cost_bias, models));
            }

            return results;
          },
          "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = nb::stl::vector<std::string>{},
          "Batch route multiple embeddings (N×D float64 numpy array)")

      // Introspection
      .def("get_supported_models", &Router::get_supported_models,
           "Get list of all supported model IDs")
      .def("get_n_clusters", &Router::get_n_clusters, "Get number of clusters")
      .def("get_embedding_dim", &Router::get_embedding_dim, "Get expected embedding dimension");

   // Module-level version info
    m.attr("__version__") = ADAPTIVE_VERSION;
}
