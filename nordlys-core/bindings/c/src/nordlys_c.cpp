#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <nordlys/checkpoint/checkpoint.hpp>
#include <nordlys/common/device.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/routing/nordlys.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <variant>

#include "nordlys.h"

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
static void cleanup_route_result_contents(NordlysRouteResult* result) {
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

// Get Nordlys C++ class pointer from opaque C handle
static ::Nordlys* get_nordlys(Nordlys* nordlys) {
  return nordlys ? reinterpret_cast<::Nordlys*>(nordlys) : nullptr;
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

// Factory: creates nordlys from checkpoint
static std::optional<::Nordlys> create_nordlys(NordlysCheckpoint profile, Device device) {
  auto result = ::Nordlys::from_checkpoint(std::move(profile), device);
  if (!result) return std::nullopt;
  return std::move(result.value());
}

// Helper to build route result
static NordlysRouteResult* build_route_result(const RouteResult& response) {
  auto* result = static_cast<NordlysRouteResult*>(malloc(sizeof(NordlysRouteResult)));
  if (!result) return nullptr;

  result->selected_model = str_duplicate(response.selected_model);
  if (!result->selected_model) {
    free(result);
    return nullptr;
  }

  result->cluster_id = response.cluster_id;
  result->cluster_distance = response.cluster_distance;  // float

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

Nordlys* nordlys_create(const char* profile_path, NordlysDevice device) {
  if (!profile_path) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_json(profile_path);
    auto dev = device_to_device(device);
    auto nordlys = create_nordlys(std::move(profile), dev);
    if (!nordlys) return nullptr;
    return reinterpret_cast<Nordlys*>(new ::Nordlys(std::move(*nordlys)));
  } catch (...) {
    return nullptr;
  }
}

Nordlys* nordlys_create_from_json(const char* json_str, NordlysDevice device) {
  if (!json_str) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_json_string(json_str);
    auto dev = device_to_device(device);
    auto nordlys = create_nordlys(std::move(profile), dev);
    if (!nordlys) return nullptr;
    return reinterpret_cast<Nordlys*>(new ::Nordlys(std::move(*nordlys)));
  } catch (...) {
    return nullptr;
  }
}

Nordlys* nordlys_create_from_msgpack(const char* path, NordlysDevice device) {
  if (!path) return nullptr;
  try {
    auto profile = NordlysCheckpoint::from_msgpack(path);
    auto dev = device_to_device(device);
    auto nordlys = create_nordlys(std::move(profile), dev);
    if (!nordlys) return nullptr;
    return reinterpret_cast<Nordlys*>(new ::Nordlys(std::move(*nordlys)));
  } catch (...) {
    return nullptr;
  }
}

void nordlys_destroy(Nordlys* nordlys) { delete get_nordlys(nordlys); }

NordlysRouteResult* nordlys_route(Nordlys* nordlys, const float* embedding,
                                  size_t embedding_size, NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!nordlys) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_NORDLYS;
    return nullptr;
  }
  if (!embedding) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* n = get_nordlys(nordlys);
    EmbeddingView view{embedding, embedding_size, Device{CpuDevice{}}};
    auto response = n->route(view);
    return build_route_result(response);
  } catch (...) {
    if (error_out) *error_out = NORDLYS_ERROR_INTERNAL;
    return nullptr;
  }
}

void nordlys_route_result_free(NordlysRouteResult* result) {
  if (!result) return;
  cleanup_route_result_contents(result);
  free(result);
}

NordlysBatchRouteResult* nordlys_route_batch(Nordlys* nordlys, const float* embeddings,
                                             size_t n_embeddings, size_t embedding_size,
                                             NordlysErrorCode* error_out) {
  if (error_out) *error_out = NORDLYS_OK;

  if (!nordlys) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_NORDLYS;
    return nullptr;
  }
  if (!embeddings) {
    if (error_out) *error_out = NORDLYS_ERROR_NULL_EMBEDDING;
    return nullptr;
  }

  try {
    auto* batch_result
        = static_cast<NordlysBatchRouteResult*>(malloc(sizeof(NordlysBatchRouteResult)));
    if (!batch_result) {
      return nullptr;
    }

    batch_result->count = n_embeddings;
    batch_result->results = nullptr;

    if (n_embeddings == 0) {
      return batch_result;
    }

    batch_result->results
        = static_cast<NordlysRouteResult*>(malloc(sizeof(NordlysRouteResult) * n_embeddings));
    if (!batch_result->results) {
      free(batch_result);
      return nullptr;
    }

    // Initialize all results to zero
    std::fill_n(batch_result->results, n_embeddings, NordlysRouteResult{});

    // Route each embedding
    for (size_t result_idx = 0; result_idx < n_embeddings; ++result_idx) {
      const float* embedding_ptr = embeddings + (result_idx * embedding_size);

      // Call single route
      NordlysErrorCode route_error;
      auto* result = nordlys_route(nordlys, embedding_ptr, embedding_size, &route_error);

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

void nordlys_batch_route_result_free(NordlysBatchRouteResult* result) {
  if (!result) return;

  if (result->results) {
    // Free each individual result's data using ranges
    auto results_span = std::span(result->results, result->count);
    std::ranges::for_each(results_span, [](NordlysRouteResult& res) {
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

void nordlys_string_free(char* str) { free(str); }

size_t nordlys_get_n_clusters(Nordlys* nordlys) {
  if (!nordlys) return 0;
  try {
    return static_cast<size_t>(get_nordlys(nordlys)->get_n_clusters());
  } catch (...) {
    return 0;
  }
}

size_t nordlys_get_embedding_dim(Nordlys* nordlys) {
  if (!nordlys) return 0;
  try {
    return static_cast<size_t>(get_nordlys(nordlys)->get_embedding_dim());
  } catch (...) {
    return 0;
  }
}

char** nordlys_get_supported_models(Nordlys* nordlys, size_t* count) {
  if (!nordlys || !count) {
    if (count) *count = 0;
    return nullptr;
  }
  try {
    auto models = get_nordlys(nordlys)->get_supported_models();

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

}  // extern "C"
