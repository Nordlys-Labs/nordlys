#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/checkpoint.hpp>

#include "bindings.h"

namespace nb = nanobind;
using namespace nb::literals;

void register_checkpoint(nb::module_& m) {
  // Checkpoint
  nb::class_<NordlysCheckpoint>(
      m, "NordlysCheckpoint",
      "Serialized Nordlys model checkpoint containing cluster centers and model metadata")
      // Loading methods
      .def_static("from_json_file", &NordlysCheckpoint::from_json, "path"_a,
                  "Load checkpoint from JSON file\n\n"
                  "Args:\n"
                  "    path: Path to JSON file\n\n"
                  "Returns:\n"
                  "    NordlysCheckpoint instance")
      .def_static("from_json_string", &NordlysCheckpoint::from_json_string, "json_str"_a,
                  "Load checkpoint from JSON string")
      .def_static("from_msgpack_file", &NordlysCheckpoint::from_msgpack, "path"_a,
                  "Load checkpoint from MessagePack file")
      .def_static(
          "from_msgpack_bytes",
          [](nb::bytes data) {
            return NordlysCheckpoint::from_msgpack_string(std::string(data.c_str(), data.size()));
          },
          "data"_a, "Load checkpoint from MessagePack bytes")

      // Serialization methods
      .def("to_json_string", &NordlysCheckpoint::to_json_string,
           "Serialize checkpoint to JSON string")
      .def("to_json_file", &NordlysCheckpoint::to_json, "path"_a, "Write checkpoint to JSON file")
      .def(
          "to_msgpack_bytes",
          [](const NordlysCheckpoint& c) {
            std::string data = c.to_msgpack_string();
            return nb::bytes(data.data(), data.size());
          },
          "Serialize checkpoint to MessagePack bytes")
      .def("to_msgpack_file", &NordlysCheckpoint::to_msgpack, "path"_a,
           "Write checkpoint to MessagePack file")

      // Validation
      .def("validate", &NordlysCheckpoint::validate, "Validate checkpoint data integrity")

      // Version
      .def_ro("version", &NordlysCheckpoint::version, "Checkpoint format version")

      // Core data - convert Matrix to ndarray for Python
      .def_prop_ro(
          "cluster_centers",
          [](const NordlysCheckpoint& c) {
            size_t shape[2] = {c.cluster_centers.rows(), c.cluster_centers.cols()};
            return nb::cast(nb::ndarray<nb::numpy, const float, nb::ndim<2>>(
                c.cluster_centers.data(), 2, shape, nb::handle()));
          },
          nb::rv_policy::reference_internal,
          "Cluster centers as numpy array (float32)")
      .def_ro("models", &NordlysCheckpoint::models, "List of model configurations")

      // Configuration structs
      .def_ro("embedding", &NordlysCheckpoint::embedding, "Embedding configuration")
      .def_ro("clustering", &NordlysCheckpoint::clustering, "Clustering configuration")
      .def_ro("metrics", &NordlysCheckpoint::metrics, "Training metrics (optional fields)")

      // Computed properties
      .def_prop_ro("n_clusters", &NordlysCheckpoint::n_clusters, "Number of clusters (computed)")
      .def_prop_ro("feature_dim", &NordlysCheckpoint::feature_dim,
                   "Feature dimensionality (computed)")

      // Convenience accessors (aliases)
      .def_prop_ro("embedding_model", &NordlysCheckpoint::embedding_model, "Embedding model ID")
      .def_prop_ro("random_state", &NordlysCheckpoint::random_state, "Random state")
      .def_prop_ro("allow_trust_remote_code", &NordlysCheckpoint::allow_trust_remote_code,
                   "Trust remote code flag")
      .def_prop_ro("silhouette_score", &NordlysCheckpoint::silhouette_score,
                   "Silhouette score (-1.0 if not available)");
}
