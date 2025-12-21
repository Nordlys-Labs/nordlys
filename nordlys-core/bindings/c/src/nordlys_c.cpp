#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <variant>

#include "adaptive.h"
#include <adaptive_core/router.hpp>
#include <adaptive_core/profile.hpp>

// Type-erased router: either float or double precision
using RouterVariant = std::variant<RouterT<float>, RouterT<double>>;

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
template<typename ResultT>
static void cleanup_route_result_contents(ResultT* result) {
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
static RouterVariant* get_router(AdaptiveRouter* router) {
  return router ? reinterpret_cast<RouterVariant*>(router) : nullptr;
}

// Factory: creates correct router type based on profile dtype
static std::optional<RouterVariant> create_router_variant(RouterProfile profile) {
  if (profile.is_float64()) {
    auto result = RouterT<double>::from_profile(std::move(profile));
    if (!result) return std::nullopt;
    return RouterVariant{std::in_place_type<RouterT<double>>, std::move(result.value())};
  } else {
    auto result = RouterT<float>::from_profile(std::move(profile));
    if (!result) return std::nullopt;
    return RouterVariant{std::in_place_type<RouterT<float>>, std::move(result.value())};
  }
}

// Helper to build precision-specific result from RouteResponseT<Scalar>
template<typename Scalar, typename ResultT>
static ResultT* build_route_result(const RouteResponseT<Scalar>& response) {
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

AdaptiveRouter* adaptive_router_create(const char* profile_path) {
  if (!profile_path) return nullptr;
  try {
    auto profile = RouterProfile::from_json(profile_path);
    auto variant = create_router_variant(std::move(profile));
    if (!variant) return nullptr;
    return reinterpret_cast<AdaptiveRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

AdaptiveRouter* adaptive_router_create_from_json(const char* json_str) {
  if (!json_str) return nullptr;
  try {
    auto profile = RouterProfile::from_json_string(json_str);
    auto variant = create_router_variant(std::move(profile));
    if (!variant) return nullptr;
    return reinterpret_cast<AdaptiveRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

AdaptiveRouter* adaptive_router_create_from_binary(const char* path) {
  if (!path) return nullptr;
  try {
    auto profile = RouterProfile::from_binary(path);
    auto variant = create_router_variant(std::move(profile));
    if (!variant) return nullptr;
    return reinterpret_cast<AdaptiveRouter*>(new RouterVariant(std::move(*variant)));
  } catch (...) {
    return nullptr;
  }
}

void adaptive_router_destroy(AdaptiveRouter* router) {
  delete get_router(router);
}

AdaptiveRouteResult32* adaptive_router_route_f32(AdaptiveRouter* router, const float* embedding,
                                                      size_t embedding_size, float cost_bias,
                                                      AdaptiveErrorCode* error_out) {
  if (error_out) *error_out = ADAPTIVE_OK;

  if (!router) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embedding) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* var = get_router(router);

    // Strict: only works on float32 router
    if (!std::holds_alternative<RouterT<float>>(*var)) {
      if (error_out) *error_out = ADAPTIVE_ERROR_TYPE_MISMATCH;
      return nullptr;
    }

    auto& r = std::get<RouterT<float>>(*var);
    auto response = r.route(embedding, embedding_size, cost_bias);
    return build_route_result<float, AdaptiveRouteResult32>(response);
  } catch (...) {
    if (error_out) *error_out = ADAPTIVE_ERROR_INTERNAL;
    return nullptr;
  }
}



void adaptive_route_result_free_f32(AdaptiveRouteResult32* result) {
   if (!result) return;
   cleanup_route_result_contents(result);
   free(result);
}

void adaptive_route_result_free_f64(AdaptiveRouteResult64* result) {
   if (!result) return;
   cleanup_route_result_contents(result);
   free(result);
}

AdaptiveRouteResult64* adaptive_router_route_f64(AdaptiveRouter* router, const double* embedding,
                                                     size_t embedding_size, float cost_bias,
                                                     AdaptiveErrorCode* error_out) {
  if (error_out) *error_out = ADAPTIVE_OK;

  if (!router) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embedding) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* var = get_router(router);

    // Strict: only works on float64 router
    if (!std::holds_alternative<RouterT<double>>(*var)) {
      if (error_out) *error_out = ADAPTIVE_ERROR_TYPE_MISMATCH;
      return nullptr;
    }

    auto& r = std::get<RouterT<double>>(*var);
    auto response = r.route(embedding, embedding_size, cost_bias);
    return build_route_result<double, AdaptiveRouteResult64>(response);
  } catch (...) {
    if (error_out) *error_out = ADAPTIVE_ERROR_INTERNAL;
    return nullptr;
  }
}

AdaptiveBatchRouteResult32* adaptive_router_route_batch_f32(
    AdaptiveRouter* router,
    const float* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    AdaptiveErrorCode* error_out) {

   if (error_out) *error_out = ADAPTIVE_OK;

   if (!router) {
     if (error_out) *error_out = ADAPTIVE_ERROR_NULL_ROUTER;
     return nullptr;
   }
   if (!embeddings) {
     if (error_out) *error_out = ADAPTIVE_ERROR_NULL_EMBEDDING;
     return nullptr;
   }

   try {
     auto* batch_result = static_cast<AdaptiveBatchRouteResult32*>(malloc(sizeof(AdaptiveBatchRouteResult32)));
     if (!batch_result) {
       return nullptr;
     }

     batch_result->count = n_embeddings;
     batch_result->results = nullptr;

     if (n_embeddings == 0) {
       return batch_result;
     }

     batch_result->results = static_cast<AdaptiveRouteResult32*>(malloc(sizeof(AdaptiveRouteResult32) * n_embeddings));
     if (!batch_result->results) {
       free(batch_result);
       return nullptr;
     }

     // Initialize all results to zero
     std::fill_n(batch_result->results, n_embeddings, AdaptiveRouteResult32{});

    // Route each embedding
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const float* embedding_ptr = embeddings + (result_idx * embedding_size);

      // Call single route
      AdaptiveErrorCode route_error;
      auto* result = adaptive_router_route_f32(router, embedding_ptr, embedding_size, cost_bias, &route_error);

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

void adaptive_batch_route_result_free_f32(AdaptiveBatchRouteResult32* result) {
   if (!result) return;

   if (result->results) {
      // Free each individual result's data using ranges
      auto results_span = std::span(result->results, result->count);
      std::ranges::for_each(results_span, [](AdaptiveRouteResult32& res) {
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

void adaptive_batch_route_result_free_f64(AdaptiveBatchRouteResult64* result) {
   if (!result) return;

   if (result->results) {
      // Free each individual result's data using ranges
      auto results_span = std::span(result->results, result->count);
      std::ranges::for_each(results_span, [](AdaptiveRouteResult64& res) {
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

AdaptiveBatchRouteResult64* adaptive_router_route_batch_f64(
    AdaptiveRouter* router,
    const double* embeddings,
    size_t n_embeddings,
    size_t embedding_size,
    float cost_bias,
    AdaptiveErrorCode* error_out) {

  if (error_out) *error_out = ADAPTIVE_OK;

  if (!router) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_ROUTER;
    return nullptr;
  }
  if (!embeddings) {
    if (error_out) *error_out = ADAPTIVE_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* batch_result = static_cast<AdaptiveBatchRouteResult64*>(malloc(sizeof(AdaptiveBatchRouteResult64)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results = static_cast<AdaptiveRouteResult64*>(malloc(sizeof(AdaptiveRouteResult64) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, AdaptiveRouteResult64{});

    // Route each embedding using the double version
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const double* embedding_ptr = embeddings + (result_idx * embedding_size);

      AdaptiveErrorCode route_error;
      auto* result = adaptive_router_route_f64(router, embedding_ptr, embedding_size, cost_bias, &route_error);

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

void adaptive_string_free(char* str) { free(str); }

size_t adaptive_router_get_n_clusters(AdaptiveRouter* router) {
  if (!router) return 0;
  try {
    return std::visit([](auto& r) -> size_t {
      return static_cast<size_t>(r.get_n_clusters());
    }, *get_router(router));
  } catch (...) {
    return 0;
  }
}

size_t adaptive_router_get_embedding_dim(AdaptiveRouter* router) {
  if (!router) return 0;
  try {
    return std::visit([](auto& r) -> size_t {
      return static_cast<size_t>(r.get_embedding_dim());
    }, *get_router(router));
  } catch (...) {
    return 0;
  }
}

char** adaptive_router_get_supported_models(AdaptiveRouter* router, size_t* count) {
  if (!router || !count) {
    if (count) *count = 0;
    return nullptr;
  }
  try {
    auto models = std::visit([](auto& r) {
      return r.get_supported_models();
    }, *get_router(router));

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

void adaptive_string_array_free(char** strings, size_t count) {
  if (strings) {
    auto strings_span = std::span(strings, count);
    std::ranges::for_each(strings_span, [](char* str) { free(str); });
    free(strings);
  }
}

AdaptivePrecision adaptive_router_get_precision(AdaptiveRouter* router) {
  if (!router) return ADAPTIVE_PRECISION_UNKNOWN;
  auto* var = get_router(router);
  return std::holds_alternative<RouterT<double>>(*var)
      ? ADAPTIVE_PRECISION_FLOAT64
      : ADAPTIVE_PRECISION_FLOAT32;
}

}  // extern "C"
