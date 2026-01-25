#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <nordlys_core/checkpoint.hpp>
#include <nordlys_core/device.hpp>
#include <nordlys_core/embedding_view.hpp>
#include <nordlys_core/nordlys.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <variant>

#include "nordlys.h"

// Type-erased router: either float or double precision
using RouterVariant = std::variant<Nordlys<float>, Nordlys<double>>;

// Internal helper to convert std::string to C string
static char* str_duplicate(const std::string& str) {
  char* result = static_cast<char*>(malloc(str.length() + 1));
  if (result) {
    std::ranges::copy(str, result);
    result[str.length()] = '\0';
  }
  return result;
}

// Internal helper to clean up route result contents (but not the struct itself)
template <typename ResultT> static void cleanup_route_result_contents(ResultT* result) {
  if (!result) return;

  free(result->selected_model);
  result->selected_model = nullptr;

  if (result->alternatives) {
    // Use ranges to free alternatives
    auto alternatives_span = std::span(result->alternatives, result->alternatives_count);
    std::ranges::for_each(alternatives_span, [](char* str) { free(str); });
    free(result->alternatives);
    result->alternatives = nullptr;
  }
  result->alternatives_count = 0;
}

// Get RouterVariant pointer from opaque handle
static RouterVariant* get_router(NordlysRouter* router) {
  return router ? reinterpret_cast<RouterVariant*>(router) : nullptr;
}

// Helper to convert NordlysDevice to Device variant
static Device device_to_device(NordlysDevice device) {
  switch (device) {
    case NORDLYS_DEVICE_CPU:
      return Device{CpuDevice{}};
    case NORDLYS_DEVICE_CUDA:
      return Device{CudaDevice{0}};
    default:
      return Device{CpuDevice{}};
  }
}

// Factory: creates correct router type based on profile dtype
static std::optional<RouterVariant> create_router_variant(NordlysCheckpoint profile, Device device) {
  if (profile.dtype() == "float64") {
    auto result = Nordlys<double>::from_checkpoint(std::move(profile), device);
    if (!result) return std::nullopt;
    return RouterVariant{std::in_place_type<Nordlys<double>>, std::move(result.value())};
  } else {
    auto result = Nordlys<float>::from_checkpoint(std::move(profile), device);
    if (!result) return std::nullopt;
    return RouterVariant{std::in_place_type<Nordlys<float>>, std::move(result.value())};
  }
}

// Helper to build precision-specific result from RouteResult<Scalar>
template <typename Scalar, typename ResultT>
static ResultT* build_route_result(const RouteResult<Scalar>& response) {
  auto* result = static_cast<ResultT*>(malloc(sizeof(ResultT)));
  if (!result) return nullptr;

  result->selected_model = str_duplicate(response.selected_model);
  if (!result->selected_model) {
    free(result);
    return nullptr;
  }

  result->cluster_id = response.cluster_id;
  result->cluster_distance = response.cluster_distance;

  result->alternatives_count = response.alternatives.size();
  if (result->alternatives_count > 0) {
    result->alternatives = static_cast<char**>(malloc(sizeof(char*) * result->alternatives_count));
    if (!result->alternatives) {
      free(result->selected_model);
      free(result);
      return nullptr;
    }

    for (size_t i = 0; i < result->alternatives_count; ++i) {
      result->alternatives[i] = str_duplicate(response.alternatives[i]);
      if (!result->alternatives[i]) {
        for (size_t j = 0; j < i; ++j) free(result->alternatives[j]);
        free(result->alternatives);
        free(result->selected_model);
        free(result);
        return nullptr;
      }
    }
  } else {
    result->alternatives = nullptr;
  }

  return result;
}

// C API implementation
extern "C" {

NordlysRouter* nordlys_router_create(const char* profile_path, NordlysDevice device) {
  if (!profile_path) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_json(profile_path);
    auto dev = device_to_device(device);
    auto variant = create_router_variant(std::move(profile), dev);
    if (!variant) return nullptr;
    return reinterpret_cast<NordlysRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

NordlysRouter* nordlys_router_create_from_json(const char* json_str, NordlysDevice device) {
  if (!json_str) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_json_string(json_str);
    auto dev = device_to_device(device);
    auto variant = create_router_variant(std::move(profile), dev);
    if (!variant) return nullptr;
    return reinterpret_cast<NordlysRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

NordlysRouter* nordlys_router_create_from_msgpack(const char* path, NordlysDevice device) {
  if (!path) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_msgpack(path);
    auto dev = device_to_device(device);
    auto variant = create_router_variant(std::move(profile), dev);
    if (!variant) return nullptr;
    return reinterpret_cast<NordlysRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

void nordlys_router_destroy(NordlysRouter* router) { delete get_router(router); }

NordlysRouteResult32* nordlys_router_route_f32(NordlysRouter* router, const float* embedding,
                                               size_t embedding_size,
                                               NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!router) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embedding) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* var = get_router(router);

    // Strict: only works on float32 router
    if (!std::holds_alternative<Nordlys<float>>(*var)) {
      if (error_out) *error_out = NORDLYS_ERROR_TYPE_MISMATCH;
      return nullptr;
    }

    auto& r = std::get<Nordlys<float>>(*var);
    EmbeddingView<float> view{embedding, embedding_size, Device{CpuDevice{}}};
    auto response = r.route(view);
    return build_route_result<float, NordlysRouteResult32>(response);
  } catch (...) {
    if (error_out) *error_out = NORDLYS_ERROR_INTERNAL;
    return nullptr;
  }
}

void nordlys_route_result_free_f32(NordlysRouteResult32* result) {
  if (!result) return;
  cleanup_route_result_contents(result);
  free(result);
}

void nordlys_route_result_free_f64(NordlysRouteResult64* result) {
  if (!result) return;
  cleanup_route_result_contents(result);
  free(result);
}

NordlysRouteResult64* nordlys_router_route_f64(NordlysRouter* router, const double* embedding,
                                               size_t embedding_size,
                                               NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!router) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embedding) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* var = get_router(router);

    // Strict: only works on float64 router
    if (!std::holds_alternative<Nordlys<double>>(*var)) {
      if (error_out) *error_out = NORDLYS_ERROR_TYPE_MISMATCH;
      return nullptr;
    }

    auto& r = std::get<Nordlys<double>>(*var);
    EmbeddingView<double> view{embedding, embedding_size, Device{CpuDevice{}}};
    auto response = r.route(view);
    return build_route_result<double, NordlysRouteResult64>(response);
  } catch (...) {
    if (error_out) *error_out = NORDLYS_ERROR_INTERNAL;
    return nullptr;
  }
}

NordlysBatchRouteResult32* nordlys_router_route_batch_f32(NordlysRouter* router,
                                                          const float* embeddings,
                                                          size_t n_embeddings,
                                                          size_t embedding_size,
                                                          NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!router) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embeddings) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* batch_result
        = static_cast<NordlysBatchRouteResult32*>(malloc(sizeof(NordlysBatchRouteResult32)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results
        = static_cast<NordlysRouteResult32*>(malloc(sizeof(NordlysRouteResult32) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, NordlysRouteResult32{});

    // Route each embedding
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const float* embedding_ptr = embeddings + (result_idx * embedding_size);

      // Call single route
      NordlysErrorCode route_error;
      auto* result = nordlys_router_route_f32(router, embedding_ptr, embedding_size, &route_error);

      if (result) {
        // Transfer ownership of data
        batch_result->results[result_idx] = *result;
        // Free only the result struct, not the data inside
        free(result);
      } else {
        // If routing fails, clean up previously allocated results
        for (size_t cleanup_idx = 0; cleanup_idx < result_idx; ++cleanup_idx) {
          cleanup_route_result_contents(&batch_result->results[cleanup_idx]);
        }
        free(batch_result->results);
        free(batch_result);
        return nullptr;
      }
    }

    return batch_result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

void nordlys_batch_route_result_free_f32(NordlysBatchRouteResult32* result) {
  if (!result) return;

  if (result->results) {
    // Free each individual result's data using ranges
    auto results_span = std::span(result->results, result->count);
    std::ranges::for_each(results_span, [](NordlysRouteResult32& res) {
      free(res.selected_model);
      if (res.alternatives) {
        auto alternatives_span = std::span(res.alternatives, res.alternatives_count);
        std::ranges::for_each(alternatives_span, [](char* str) { free(str); });
        free(res.alternatives);
      }
    });
    free(result->results);
  }
  free(result);
}

void nordlys_batch_route_result_free_f64(NordlysBatchRouteResult64* result) {
  if (!result) return;

  if (result->results) {
    // Free each individual result's data using ranges
    auto results_span = std::span(result->results, result->count);
    std::ranges::for_each(results_span, [](NordlysRouteResult64& res) {
      free(res.selected_model);
      if (res.alternatives) {
        auto alternatives_span = std::span(res.alternatives, res.alternatives_count);
        std::ranges::for_each(alternatives_span, [](char* str) { free(str); });
        free(res.alternatives);
      }
    });
    free(result->results);
  }
  free(result);
}

NordlysBatchRouteResult64* nordlys_router_route_batch_f64(NordlysRouter* router,
                                                          const double* embeddings,
                                                          size_t n_embeddings,
                                                          size_t embedding_size,
                                                          NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!router) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embeddings) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* batch_result
        = static_cast<NordlysBatchRouteResult64*>(malloc(sizeof(NordlysBatchRouteResult64)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results
        = static_cast<NordlysRouteResult64*>(malloc(sizeof(NordlysRouteResult64) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, NordlysRouteResult64{});

    // Route each embedding using the double version
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const double* embedding_ptr = embeddings + (result_idx * embedding_size);

      NordlysErrorCode route_error;
      auto* result = nordlys_router_route_f64(router, embedding_ptr, embedding_size, &route_error);

      if (result) {
        batch_result->results[result_idx] = *result;
        free(result);
      } else {
        // If routing fails, clean up previously allocated results
        for (size_t cleanup_idx = 0; cleanup_idx < result_idx; ++cleanup_idx) {
          cleanup_route_result_contents(&batch_result->results[cleanup_idx]);
        }
        free(batch_result->results);
        free(batch_result);
        return nullptr;
      }
    }

    return batch_result;
  } catch (const std::exception&) {
    return nullptr;
  }
}

void nordlys_string_free(char* str) { free(str); }

size_t nordlys_router_get_n_clusters(NordlysRouter* router) {
  if (!router) return 0;
  try {
    return std::visit([](auto& r) -> size_t { return static_cast<size_t>(r.get_n_clusters()); },
                      *get_router(router));
  } catch (...) {
    return 0;
  }
}

size_t nordlys_router_get_embedding_dim(NordlysRouter* router) {
  if (!router) return 0;
  try {
    return std::visit([](auto& r) -> size_t { return static_cast<size_t>(r.get_embedding_dim()); },
                      *get_router(router));
  } catch (...) {
    return 0;
  }
}

char** nordlys_router_get_supported_models(NordlysRouter* router, size_t* count) {
  if (!router || !count) {
    if (count) *count = 0;
    return nullptr;
  }
  try {
    auto models = std::visit([](auto& r) { return r.get_supported_models(); }, *get_router(router));

    *count = models.size();
    if (models.empty()) {
      return nullptr;
    }

    char** result = static_cast<char**>(malloc(sizeof(char*) * models.size()));
    if (!result) {
      *count = 0;
      return nullptr;
    }

    for (size_t str_idx = 0; str_idx < models.size(); ++str_idx) {
      result[str_idx] = str_duplicate(models[str_idx]);
      if (!result[str_idx]) {
        // Allocation failed, clean up all previously allocated strings
        for (size_t cleanup_idx = 0; cleanup_idx < str_idx; ++cleanup_idx) {
          free(result[cleanup_idx]);
        }
        free(result);
        *count = 0;
        return nullptr;
      }
    }

    return result;
  } catch (...) {
    *count = 0;
    return nullptr;
  }
}

void nordlys_string_array_free(char** strings, size_t count) {
  if (strings) {
    auto strings_span = std::span(strings, count);
    std::ranges::for_each(strings_span, [](char* str) { free(str); });
    free(strings);
  }
}

NordlysPrecision nordlys_router_get_precision(NordlysRouter* router) {
  if (!router) return NORDLYS_PRECISION_UNKNOWN;
  auto* var = get_router(router);
  return std::holds_alternative<Nordlys<double>>(*var) ? NORDLYS_PRECISION_FLOAT64
                                                       : NORDLYS_PRECISION_FLOAT32;
}

}  // extern "C"
