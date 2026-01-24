#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/checkpoint.hpp>
#include <nordlys_core/nordlys.hpp>

#include "helpers.h"
#include "bindings.h"

namespace nb = nanobind;
using namespace nb::literals;

void register_nordlys(nb::module_& m) {
  // Nordlys32 (float32)
  nb::class_<Nordlys<float>>(
      m, "Nordlys32",
      "High-performance routing engine with float32 precision\n\n"
      "This class provides intelligent model selection based on prompt clustering.\n"
      "Use Nordlys32.from_checkpoint() to load a trained model.")
      .def_static(
          "from_checkpoint",
          [](NordlysCheckpoint checkpoint, const std::string& device) {
            auto device_type = device_string_to_enum(device);
            auto result = Nordlys<float>::from_checkpoint(std::move(checkpoint), device_type);
            if (!result) {
              throw nb::value_error(result.error().c_str());
            }
            return std::move(result.value());
          },
          "checkpoint"_a, "device"_a = "cpu",
          "Load engine from checkpoint\n\n"
          "Args:\n"
          "    checkpoint: NordlysCheckpoint instance with float32 dtype\n"
          "    device: Device to use ('cpu' or 'cuda'), defaults to 'cpu'\n\n"
          "Returns:\n"
          "    Nordlys32 engine instance\n\n"
          "Raises:\n"
          "    ValueError: If checkpoint dtype doesn't match float32 or device is invalid")
      .def(
          "route",
          [](Nordlys<float>& self, nb::ndarray<float, nb::ndim<1>, nb::c_contig> embedding,
             float cost_bias, const std::vector<std::string>& models) {
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
      .def(
          "route_batch",
          [](Nordlys<float>& self, nb::ndarray<float, nb::ndim<2>, nb::c_contig> embeddings,
             float cost_bias, const std::vector<std::string>& models) {
            return self.route_batch(embeddings.data(), embeddings.shape(0), embeddings.shape(1),
                                    cost_bias, models);
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
      .def_prop_ro("n_clusters", &Nordlys<float>::get_n_clusters, "Number of clusters in the model")
      .def_prop_ro("embedding_dim", &Nordlys<float>::get_embedding_dim,
                   "Expected embedding dimensionality")
      .def_prop_ro(
          "dtype", [](const Nordlys<float>&) { return "float32"; }, "Data type of the engine");

  // Nordlys64 (float64)
  nb::class_<Nordlys<double>>(
      m, "Nordlys64",
      "High-performance routing engine with float64 precision\n\n"
      "This class provides intelligent model selection based on prompt clustering.\n"
      "Use Nordlys64.from_checkpoint() to load a trained model.")
      .def_static(
          "from_checkpoint",
          [](NordlysCheckpoint checkpoint, const std::string& device) {
            auto device_type = device_string_to_enum(device);
            auto result = Nordlys<double>::from_checkpoint(std::move(checkpoint), device_type);
            if (!result) {
              throw nb::value_error(result.error().c_str());
            }
            return std::move(result.value());
          },
          "checkpoint"_a, "device"_a = "cpu",
          "Load engine from checkpoint\n\n"
          "Args:\n"
          "    checkpoint: NordlysCheckpoint instance with float64 dtype\n"
          "    device: Device to use ('cpu' or 'cuda'), defaults to 'cpu'\n\n"
          "Returns:\n"
          "    Nordlys64 engine instance\n\n"
          "Raises:\n"
          "    ValueError: If checkpoint dtype doesn't match float64 or device is invalid")
      .def(
          "route",
          [](Nordlys<double>& self, nb::ndarray<double, nb::ndim<1>, nb::c_contig> embedding,
             float cost_bias, const std::vector<std::string>& models) {
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
      .def(
          "route_batch",
          [](Nordlys<double>& self, nb::ndarray<double, nb::ndim<2>, nb::c_contig> embeddings,
             float cost_bias, const std::vector<std::string>& models) {
            return self.route_batch(embeddings.data(), embeddings.shape(0), embeddings.shape(1),
                                    cost_bias, models);
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
      .def_prop_ro(
          "dtype", [](const Nordlys<double>&) { return "float64"; }, "Data type of the engine");
}
