#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/nordlys.hpp>
#include <nordlys_core/checkpoint.hpp>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(nordlys_core_ext, m) {
  m.doc() = "Nordlys Core - High-performance routing engine";

  // Result types
  nb::class_<RouteResult<float>>(m, "RouteResult32",
      "Routing result for float32 precision")
      .def_ro("selected_model", &RouteResult<float>::selected_model,
          "Selected model ID")
      .def_ro("alternatives", &RouteResult<float>::alternatives,
          "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResult<float>::cluster_id,
          "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResult<float>::cluster_distance,
          "Distance to cluster center")
      .def("__repr__", [](const RouteResult<float>& r) {
          return "<RouteResult32 model='" + r.selected_model + "' cluster=" + 
                 std::to_string(r.cluster_id) + ">";
      });

  nb::class_<RouteResult<double>>(m, "RouteResult64",
      "Routing result for float64 precision")
      .def_ro("selected_model", &RouteResult<double>::selected_model,
          "Selected model ID")
      .def_ro("alternatives", &RouteResult<double>::alternatives,
          "List of alternative model IDs")
      .def_ro("cluster_id", &RouteResult<double>::cluster_id,
          "Assigned cluster ID")
      .def_ro("cluster_distance", &RouteResult<double>::cluster_distance,
          "Distance to cluster center")
      .def("__repr__", [](const RouteResult<double>& r) {
          return "<RouteResult64 model='" + r.selected_model + "' cluster=" + 
                 std::to_string(r.cluster_id) + ">";
      });

  // Checkpoint
  nb::class_<NordlysCheckpoint>(m, "NordlysCheckpoint",
      "Serialized Nordlys model checkpoint containing cluster centers and model metadata")
      // Loading methods
      .def_static("from_json_file", &NordlysCheckpoint::from_json,
          "path"_a,
          "Load checkpoint from JSON file\n\n"
          "Args:\n"
          "    path: Path to JSON file\n\n"
          "Returns:\n"
          "    NordlysCheckpoint instance")
      .def_static("from_json_string", &NordlysCheckpoint::from_json_string,
          "json_str"_a,
          "Load checkpoint from JSON string")
      .def_static("from_msgpack_file", &NordlysCheckpoint::from_msgpack,
          "path"_a,
          "Load checkpoint from MessagePack file")
      .def_static("from_msgpack_bytes", [](nb::bytes data) {
          return NordlysCheckpoint::from_msgpack_string(std::string(data.c_str(), data.size()));
      }, "data"_a,
         "Load checkpoint from MessagePack bytes")

      // Serialization methods
      .def("to_json_string", &NordlysCheckpoint::to_json_string,
          "Serialize checkpoint to JSON string")
      .def("to_json_file", &NordlysCheckpoint::to_json,
          "path"_a,
          "Write checkpoint to JSON file")
      .def("to_msgpack_bytes", [](const NordlysCheckpoint& c) {
          std::string data = c.to_msgpack_string();
          return nb::bytes(data.data(), data.size());
      }, "Serialize checkpoint to MessagePack bytes")
      .def("to_msgpack_file", &NordlysCheckpoint::to_msgpack,
          "path"_a,
          "Write checkpoint to MessagePack file")

      // Validation
      .def("validate", &NordlysCheckpoint::validate,
          "Validate checkpoint data integrity")

      // Properties
      .def_prop_ro("n_clusters", [](const NordlysCheckpoint& c) { 
          return c.metadata.n_clusters; 
      }, "Number of clusters")
      .def_prop_ro("embedding_model", [](const NordlysCheckpoint& c) { 
          return c.metadata.embedding_model; 
      }, "Embedding model name")
      .def_prop_ro("dtype", [](const NordlysCheckpoint& c) { 
          return c.metadata.dtype; 
      }, "Data type ('float32' or 'float64')")
      .def_prop_ro("silhouette_score", [](const NordlysCheckpoint& c) { 
          return c.metadata.silhouette_score; 
      }, "Silhouette score")
      .def_prop_ro("is_float32", &NordlysCheckpoint::is_float32,
          "True if checkpoint uses float32 precision")
      .def_prop_ro("is_float64", &NordlysCheckpoint::is_float64,
          "True if checkpoint uses float64 precision");

  // Nordlys32 (float32)
  nb::class_<Nordlys<float>>(m, "Nordlys32",
      "High-performance routing engine with float32 precision\n\n"
      "This class provides intelligent model selection based on prompt clustering.\n"
      "Use Nordlys32.from_checkpoint() to load a trained model.")
      .def_static("from_checkpoint",
          [](NordlysCheckpoint checkpoint) {
              auto result = Nordlys<float>::from_checkpoint(std::move(checkpoint));
              if (!result) {
                  throw nb::value_error(result.error());
              }
              return std::move(result.value());
          },
          "checkpoint"_a,
          "Load engine from checkpoint\n\n"
          "Args:\n"
          "    checkpoint: NordlysCheckpoint instance with float32 dtype\n\n"
          "Returns:\n"
          "    Nordlys32 engine instance\n\n"
          "Raises:\n"
          "    ValueError: If checkpoint dtype doesn't match float32")
      .def("route",
          [](Nordlys<float>& self,
             nb::ndarray<float, nb::ndim<1>, nb::c_contig> embedding,
             float cost_bias,
             const std::vector<std::string>& models) {
              return self.route(embedding.data(), embedding.shape(0), cost_bias, models);
          },
          "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
          "Route an embedding to the best model\n\n"
          "Args:\n"
          "    embedding: 1D numpy array of float32\n"
          "    cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    RouteResult32 with selected model and alternatives")
      .def("route_batch",
          [](Nordlys<float>& self,
             nb::ndarray<float, nb::ndim<2>, nb::c_contig> embeddings,
             float cost_bias,
             const std::vector<std::string>& models) {
              size_t n = embeddings.shape(0), d = embeddings.shape(1);
              std::vector<RouteResult<float>> results;
              results.reserve(n);
              const float* ptr = embeddings.data();
              for (size_t i = 0; i < n; ++i) {
                  results.push_back(self.route(ptr + i * d, d, cost_bias, models));
              }
              return results;
          },
          "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
          "Batch route multiple embeddings\n\n"
          "Args:\n"
          "    embeddings: 2D numpy array of float32, shape (n_samples, embedding_dim)\n"
          "    cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    List of RouteResult32")
      .def("get_supported_models", &Nordlys<float>::get_supported_models,
          "Get list of all supported model IDs")
      .def_prop_ro("n_clusters", &Nordlys<float>::get_n_clusters,
          "Number of clusters in the model")
      .def_prop_ro("embedding_dim", &Nordlys<float>::get_embedding_dim,
          "Expected embedding dimensionality")
      .def_prop_ro("dtype", [](const Nordlys<float>&) { return "float32"; },
          "Data type of the engine");

  // Nordlys64 (float64)
  nb::class_<Nordlys<double>>(m, "Nordlys64",
      "High-performance routing engine with float64 precision\n\n"
      "This class provides intelligent model selection based on prompt clustering.\n"
      "Use Nordlys64.from_checkpoint() to load a trained model.")
      .def_static("from_checkpoint",
          [](NordlysCheckpoint checkpoint) {
              auto result = Nordlys<double>::from_checkpoint(std::move(checkpoint));
              if (!result) {
                  throw nb::value_error(result.error());
              }
              return std::move(result.value());
          },
          "checkpoint"_a,
          "Load engine from checkpoint\n\n"
          "Args:\n"
          "    checkpoint: NordlysCheckpoint instance with float64 dtype\n\n"
          "Returns:\n"
          "    Nordlys64 engine instance\n\n"
          "Raises:\n"
          "    ValueError: If checkpoint dtype doesn't match float64")
      .def("route",
          [](Nordlys<double>& self,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> embedding,
             float cost_bias,
             const std::vector<std::string>& models) {
              return self.route(embedding.data(), embedding.shape(0), cost_bias, models);
          },
          "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
          "Route an embedding to the best model\n\n"
          "Args:\n"
          "    embedding: 1D numpy array of float64\n"
          "    cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    RouteResult64 with selected model and alternatives")
      .def("route_batch",
          [](Nordlys<double>& self,
             nb::ndarray<double, nb::ndim<2>, nb::c_contig> embeddings,
             float cost_bias,
             const std::vector<std::string>& models) {
              size_t n = embeddings.shape(0), d = embeddings.shape(1);
              std::vector<RouteResult<double>> results;
              results.reserve(n);
              const double* ptr = embeddings.data();
              for (size_t i = 0; i < n; ++i) {
                  results.push_back(self.route(ptr + i * d, d, cost_bias, models));
              }
              return results;
          },
          "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
          "Batch route multiple embeddings\n\n"
          "Args:\n"
          "    embeddings: 2D numpy array of float64, shape (n_samples, embedding_dim)\n"
          "    cost_bias: Cost preference (0.0=cheapest, 1.0=highest quality)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    List of RouteResult64")
      .def("get_supported_models", &Nordlys<double>::get_supported_models,
          "Get list of all supported model IDs")
      .def_prop_ro("n_clusters", &Nordlys<double>::get_n_clusters,
          "Number of clusters in the model")
      .def_prop_ro("embedding_dim", &Nordlys<double>::get_embedding_dim,
          "Expected embedding dimensionality")
      .def_prop_ro("dtype", [](const Nordlys<double>&) { return "float64"; },
          "Data type of the engine");

  // Convenience factory function
  m.def("load_checkpoint",
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
