#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys/checkpoint/checkpoint.hpp>
#include <nordlys/common/device.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/routing/nordlys.hpp>

#include "helpers.h"
#include "bindings.h"

namespace nb = nanobind;
using namespace nb::literals;

void register_nordlys(nb::module_& m) {
  // Single Nordlys class (float32)
  nb::class_<Nordlys>(
      m, "Nordlys",
      "High-performance routing engine\n\n"
      "This class provides intelligent model selection based on prompt clustering.\n"
      "Use Nordlys.from_checkpoint() to load a trained model.")
      .def_static(
          "from_checkpoint",
          [](NordlysCheckpoint checkpoint, const std::string& device) {
            auto dev = device_string_to_device(device);
            auto result = Nordlys::from_checkpoint(std::move(checkpoint), dev);
            if (!result) {
              throw nb::value_error(result.error().c_str());
            }
            return std::move(result.value());
          },
          "checkpoint"_a, "device"_a = "cpu",
          "Load engine from checkpoint\n\n"
          "Args:\n"
          "    checkpoint: NordlysCheckpoint instance\n"
          "    device: Device to use ('cpu' or 'cuda'), defaults to 'cpu'\n\n"
          "Returns:\n"
          "    Nordlys engine instance\n\n"
          "Raises:\n"
          "    ValueError: If device is invalid")
      .def(
          "route",
          [](Nordlys& self, nb::ndarray<float, nb::ndim<1>> embedding,
             const std::vector<std::string>& models) {
            Device device;
            if (embedding.device_type() == nb::device::cuda::value) {
              device = Device{CudaDevice{embedding.device_id()}};
            } else {
              device = Device{CpuDevice{}};
            }
            
            EmbeddingView view{
                static_cast<const float*>(embedding.data()),
                embedding.shape(0),
                device
            };
            
            return self.route(view, models);
          },
          "embedding"_a, "models"_a = std::vector<std::string>{},
          "Route an embedding to the best model\n\n"
          "Args:\n"
          "    embedding: 1D numpy array of float32 (CPU or GPU)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    RouteResult with selected model and alternatives")
      .def(
          "route_batch",
          [](Nordlys& self, nb::ndarray<float, nb::ndim<2>> embeddings,
             const std::vector<std::string>& models) {
            Device device;
            if (embeddings.device_type() == nb::device::cuda::value) {
              device = Device{CudaDevice{embeddings.device_id()}};
            } else {
              device = Device{CpuDevice{}};
            }
            
            EmbeddingBatchView view{
                static_cast<const float*>(embeddings.data()),
                embeddings.shape(0),
                embeddings.shape(1),
                device
            };
            
            return self.route_batch(view, models);
          },
          "embeddings"_a, "models"_a = std::vector<std::string>{},
          "Batch route multiple embeddings\n\n"
          "Args:\n"
          "    embeddings: 2D numpy array of float32, shape (n_samples, embedding_dim) (CPU or GPU)\n"
          "    models: Optional list of model IDs to consider\n\n"
          "Returns:\n"
          "    List of RouteResult")
      .def("get_supported_models", &Nordlys::get_supported_models,
           "Get list of all supported model IDs")
      .def_prop_ro("n_clusters", &Nordlys::get_n_clusters, "Number of clusters in the model")
      .def_prop_ro("embedding_dim", &Nordlys::get_embedding_dim,
                   "Expected embedding dimensionality")
      .def_prop_ro(
          "dtype", [](const Nordlys&) { return "float32"; }, "Data type of the engine (always float32)");
}
