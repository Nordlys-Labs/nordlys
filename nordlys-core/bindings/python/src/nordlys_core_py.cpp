#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <nordlys_core/router.hpp>
#include <nordlys_core/profile.hpp>
#include <variant>

namespace nb = nanobind;
using namespace nb::literals;

// Type-erased router wrapper for Python
class RouterWrapper {
public:
  using RouterVariant = std::variant<RouterT<float>, RouterT<double>>;

  RouterVariant router;
  bool is_float64;

  RouterWrapper(RouterT<float> r) : router(std::move(r)), is_float64(false) {}
  RouterWrapper(RouterT<double> r) : router(std::move(r)), is_float64(true) {}

  static RouterWrapper from_profile(RouterProfile profile) {
    if (profile.is_float64()) {
      auto r = RouterT<double>::from_profile(std::move(profile));
      if (!r) throw std::runtime_error(r.error());
      return RouterWrapper(std::move(r.value()));
    } else {
      auto r = RouterT<float>::from_profile(std::move(profile));
      if (!r) throw std::runtime_error(r.error());
      return RouterWrapper(std::move(r.value()));
    }
  }

  RouteResponseT<float> route_f32(nb::ndarray<float, nb::ndim<1>, nb::c_contig> emb,
                               float bias, const std::vector<std::string>& models) {
    if (is_float64) throw std::invalid_argument("Router expects float64, got float32");
    return std::get<RouterT<float>>(router).route(emb.data(), emb.shape(0), bias, models);
  }

  RouteResponseT<double> route_f64(nb::ndarray<double, nb::ndim<1>, nb::c_contig> emb,
                                float bias, const std::vector<std::string>& models) {
    if (!is_float64) throw std::invalid_argument("Router expects float32, got float64");
    return std::get<RouterT<double>>(router).route(emb.data(), emb.shape(0), bias, models);
  }

  std::vector<RouteResponseT<float>> route_batch_f32(nb::ndarray<float, nb::ndim<2>, nb::c_contig> embs,
                                                   float bias, const std::vector<std::string>& models) {
    if (is_float64) throw std::invalid_argument("Router expects float64, got float32");
    auto& r = std::get<RouterT<float>>(router);
    size_t n = embs.shape(0), d = embs.shape(1);
    std::vector<RouteResponseT<float>> out;
    out.reserve(n);
    const float* ptr = embs.data();
    for (size_t i = 0; i < n; ++i)
      out.push_back(r.route(ptr + i * d, d, bias, models));
    return out;
  }

  std::vector<RouteResponseT<double>> route_batch_f64(nb::ndarray<double, nb::ndim<2>, nb::c_contig> embs,
                                                   float bias, const std::vector<std::string>& models) {
    if (!is_float64) throw std::invalid_argument("Router expects float32, got float64");
    auto& r = std::get<RouterT<double>>(router);
    size_t n = embs.shape(0), d = embs.shape(1);
    std::vector<RouteResponseT<double>> out;
    out.reserve(n);
    const double* ptr = embs.data();
    for (size_t i = 0; i < n; ++i)
      out.push_back(r.route(ptr + i * d, d, bias, models));
    return out;
  }

  std::vector<std::string> get_supported_models() const {
    return std::visit([](const auto& r) { return r.get_supported_models(); }, router);
  }

  int get_n_clusters() const {
    return std::visit([](const auto& r) { return r.get_n_clusters(); }, router);
  }

  int get_embedding_dim() const {
    return std::visit([](const auto& r) { return r.get_embedding_dim(); }, router);
  }

  std::string get_dtype() const { return is_float64 ? "float64" : "float32"; }

  // Explicit cleanup to help prevent nanobind memory leaks on shutdown
  void cleanup() {
    // Reset the variant to default state, triggering destructor of held router
    if (is_float64) {
      router = RouterT<double>();
    } else {
      router = RouterT<float>();
    }
  }
};

NB_MODULE(nordlys_core_ext, m) {
  m.doc() = "Nordlys Core - High-performance routing engine";

  nb::class_<RouteResponseT<float>>(m, "RouteResponse32")
      .def_ro("selected_model", &RouteResponseT<float>::selected_model)
      .def_ro("alternatives", &RouteResponseT<float>::alternatives)
      .def_ro("cluster_id", &RouteResponseT<float>::cluster_id)
      .def_ro("cluster_distance", &RouteResponseT<float>::cluster_distance);

  nb::class_<RouteResponseT<double>>(m, "RouteResponse64")
      .def_ro("selected_model", &RouteResponseT<double>::selected_model)
      .def_ro("alternatives", &RouteResponseT<double>::alternatives)
      .def_ro("cluster_id", &RouteResponseT<double>::cluster_id)
      .def_ro("cluster_distance", &RouteResponseT<double>::cluster_distance);

  nb::class_<RouterProfile>(m, "RouterProfile")
      // Static factory methods for loading profiles
      .def_static("from_json_file", &RouterProfile::from_json,
          "path"_a, "Load profile from JSON file")
      .def_static("from_json_string", &RouterProfile::from_json_string,
          "json_str"_a, "Load profile from JSON string")
      .def_static("from_msgpack_file", &RouterProfile::from_binary,
          "path"_a, "Load profile from MessagePack binary file")
      .def_static("from_msgpack_bytes", [] (nb::bytes data) {
          return RouterProfile::from_binary_string(std::string(data.c_str(), data.size()));
      }, "data"_a, "Load profile from MessagePack binary data")

      // Serialization methods
      .def("to_json_string", &RouterProfile::to_json_string,
          "Serialize profile to JSON string")
      .def("to_json_file", &RouterProfile::to_json,
          "path"_a, "Write profile to JSON file")
      .def("to_msgpack_bytes", [] (const RouterProfile& p) {
          std::string data = p.to_binary_string();
          return nb::bytes(data.data(), data.size());
      }, "Serialize profile to MessagePack binary data")
      .def("to_msgpack_file", &RouterProfile::to_binary,
          "path"_a, "Write profile to MessagePack binary file")

      // Validation
      .def("validate", &RouterProfile::validate,
          "Validate profile data")

      // Properties
      .def_prop_ro("n_clusters", [](const RouterProfile& p) { return p.metadata.n_clusters; })
      .def_prop_ro("embedding_model", [](const RouterProfile& p) { return p.metadata.embedding_model; })
      .def_prop_ro("dtype", [](const RouterProfile& p) { return p.metadata.dtype; })
      .def_prop_ro("silhouette_score", [](const RouterProfile& p) { return p.metadata.silhouette_score; })
      .def_prop_ro("allow_trust_remote_code", [](const RouterProfile& p) { return p.metadata.allow_trust_remote_code; })
      .def_prop_ro("default_cost_preference", [](const RouterProfile& p) { return p.metadata.routing.default_cost_preference; })
      .def_prop_ro("lambda_min", [](const RouterProfile& p) { return p.metadata.routing.lambda_min; })
      .def_prop_ro("lambda_max", [](const RouterProfile& p) { return p.metadata.routing.lambda_max; })
      .def_prop_ro("max_alternatives", [](const RouterProfile& p) { return p.metadata.routing.max_alternatives; })
      .def_prop_ro("max_iter", [](const RouterProfile& p) { return p.metadata.clustering.max_iter; })
      .def_prop_ro("random_state", [](const RouterProfile& p) { return p.metadata.clustering.random_state; })
      .def_prop_ro("n_init", [](const RouterProfile& p) { return p.metadata.clustering.n_init; })
      .def_prop_ro("algorithm", [](const RouterProfile& p) { return p.metadata.clustering.algorithm; })
      .def_prop_ro("normalization_strategy", [](const RouterProfile& p) { return p.metadata.clustering.normalization_strategy; })
      .def_prop_ro("is_float32", &RouterProfile::is_float32)
      .def_prop_ro("is_float64", &RouterProfile::is_float64);

  nb::class_<RouterWrapper>(m, "Router")
      .def_static("from_json_file", [](const std::string& path) {
        return RouterWrapper::from_profile(RouterProfile::from_json(path));
      }, "path"_a, "Load router from JSON profile file")

      .def_static("from_json_string", [](const std::string& json) {
        return RouterWrapper::from_profile(RouterProfile::from_json_string(json));
      }, "json_str"_a, "Load router from JSON string")

      .def_static("from_msgpack_file", [](const std::string& path) {
        return RouterWrapper::from_profile(RouterProfile::from_binary(path));
      }, "path"_a, "Load router from MessagePack binary file")

      // Overloaded route - nanobind dispatches based on array dtype
      .def("route", &RouterWrapper::route_f32,
           "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
           "Route using embedding (float32)")
      .def("route", &RouterWrapper::route_f64,
           "embedding"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
           "Route using embedding (float64)")

      .def("route_batch", &RouterWrapper::route_batch_f32,
           "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
           "Batch route embeddings (float32)")
      .def("route_batch", &RouterWrapper::route_batch_f64,
           "embeddings"_a, "cost_bias"_a = 0.5f, "models"_a = std::vector<std::string>{},
           "Batch route embeddings (float64)")

      .def("get_supported_models", &RouterWrapper::get_supported_models)
      .def("get_n_clusters", &RouterWrapper::get_n_clusters)
      .def("get_embedding_dim", &RouterWrapper::get_embedding_dim)
      .def("cleanup", &RouterWrapper::cleanup, "Explicit cleanup to prevent memory leaks")
      .def_prop_ro("dtype", &RouterWrapper::get_dtype);

  m.attr("__version__") = NORDLYS_VERSION;
}
